local assert = assert
local ipairs = ipairs
local math = math
local pairs = pairs
local string = string
local table = table
local type = type
--
local april_assert = april_assert
local get_table_fields = get_table_fields
local iterator = iterator
local mop = matrix.op
local md = matrix.dict

local MAX_UPDATES_WITHOUT_PRUNE = ann.optimizer.MAX_UPDATES_WITHOUT_PRUNE

------------------------------------
--------- RMSPROP METHOD ----------
------------------------------------

-- Tieleman, T. and Hinton, G. (2012),
-- Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
-- http://climin.readthedocs.org/en/latest/rmsprop.html
--
-- TODO:
-- RMSProp and equilibrated adaptive learning rates for non-convex optimization
-- http://arxiv.org/pdf/1502.04390v1.pdf

local rmsprop, rmsprop_methods = class("ann.optimizer.rmsprop", ann.optimizer)
ann.optimizer.rmsprop = rmsprop -- global environment

function rmsprop:constructor(g_options, l_options, count, Eupdates, Erms)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
                              {"learning_rate", "Global learning rate (0.01)"},
                              {"momentum", "Nesterov momentum (0.0)"},
			      {"decay", "Decay rate (0.99)"},
			      {"epsilon", "Epsilon constant (1e-06)"},
                              {"weight_decay", "Weight L2 regularization (0.0)"},
                              {"max_norm_penalty", "Weight max norm upper bound (0)"},
			    },
			    g_options,
			    l_options,
			    count)
  self.Eupdates = Eupdates or {}
  self.Erms = Erms or {}
  if not g_options then
    -- default values
    self:set_option("learning_rate", 0.01)
    self:set_option("momentum", 0.0)
    self:set_option("decay", 0.99)
    self:set_option("epsilon", 1e-06)
    self:set_option("weight_decay", 0.0)
    self:set_option("max_norm_penalty", 0.0)
  end
end

function rmsprop_methods:execute(eval, weights)
  local table = table
  local assert = assert
  -- apply momentum to weights
  for wname,w in pairs(weights) do
    local mt = self:get_option_of(wname, "momentum")
    if mt > 0.0 then
      local Eupdate = self.Eupdates[wname] or matrix.as(w):zeros()
      w:axpy(-mt, Eupdate)
      self.Eupdates[wname] = Eupdate
    end
  end
  -- compute gradients
  local arg = table.pack( eval(weights) )
  local tr_loss,gradients = table.unpack(arg)
  
  -- the gradient computation could fail returning nil, it is important to take
  -- this into account
  if not gradients then return nil end
  --
  local count = self:get_count()
  for wname,w in pairs(weights) do
    local Eupdate     = self.Eupdates[wname] or matrix.as(w):zeros()
    local Erms        = self.Erms[wname] or matrix.as(w):zeros()
    local grad        = april_assert(gradients[wname],
                                     "Not found gradients of %s", wname)
    -- learning options
    local lr          = self:get_option_of(wname, "learning_rate")
    local mt          = self:get_option_of(wname, "momentum")
    local decay       = self:get_option_of(wname, "decay")
    local eps         = self:get_option_of(wname, "epsilon")
    local l2          = self:get_option_of(wname, "weight_decay")
    local mnp         = self:get_option_of(wname, "max_norm_penalty")
    
    -- L2 regularization
    if l2 > 0.0 then grad:axpy(l2, w) end
    -- apply RMSProp with Nesterov momentum rules    
    Erms:scal(decay):axpy(1 - decay, mop.cmul(grad, grad))
    if mt > 0.0 then
      local tmp = (Erms + eps):sqrt():div(lr):cmul(grad)
      Eupdate:scal(mt):axpy(1.0, tmp)
    else
      Eupdate:copy(Erms):scalar_add(eps):sqrt():div(lr):cmul(grad)
    end
    -- apply update step
    w:axpy(-1.0, Eupdate)
    -- constraints
    if mnp > 0.0 then ann.optimizer.utils.max_norm_penalty(w, mnp) end
    -- weights normality check
    if count % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      w:prune_subnormal_and_check_normal()
    end
    --
    self.Erms[wname] = Erms
    if mt > 0.0 then self.Eupdates[wname] = Eupdate end
  end
  -- count one more update iteration
  self:count_one()
  -- returns the same as returned by eval()
  return table.unpack(arg)
end

function rmsprop_methods:clone()
  local obj = ann.optimizer.rmsprop()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  obj.Eupdates          = md.clone( self.Eupdates )
  obj.Erms        = md.clone( self.Erms )
  return obj
end

function rmsprop_methods:ctor_name()
  return "ann.optimizer.rmsprop"
end
function rmsprop_methods:ctor_params()
  return self.global_options,
  self.layerwise_options,
  self.count,
  self.Eupdates,
  self.Erms
end

local rmsprop_properties = {
  gradient = true
}
function rmsprop_methods:needs_property(property)
  return rmsprop_properties[property]
end
