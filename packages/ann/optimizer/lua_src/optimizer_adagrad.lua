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

-----------------------------------
--------- ADAGRAD METHOD ----------
-----------------------------------

-- http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/

local adagrad, adagrad_methods = class("ann.optimizer.adagrad", ann.optimizer)
ann.optimizer.adagrad = adagrad -- global environment

function adagrad:constructor(g_options, l_options, count, Egradients)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
                              {"learning_rate", "Global learning rate (1.0)"},
			      {"decay", "Decay rate (0.95)"},
			      {"epsilon", "Epsilon constant (1e-06)"},
                              {"weight_decay", "Weight L2 regularization (0.0)"},
                              {"max_norm_penalty", "Weight max norm upper bound (0)"},
			    },
			    g_options,
			    l_options,
			    count)
  self.Egradients = Egradients or {}
  if not g_options then
    -- default values
    self:set_option("learning_rate", 1.0)
    self:set_option("decay", 0.95)
    self:set_option("epsilon", 1e-06)
    self:set_option("weight_decay", 0.0)
    self:set_option("max_norm_penalty", 0.0)
  end
end

function adagrad_methods:execute(eval, weights)
  local table = table
  local assert = assert
  --
  local origw = weights
  local arg = table.pack( eval(origw) )
  local tr_loss,gradients = table.unpack(arg)
  -- the gradient computation could fail returning nil, it is important to take
  -- this into account
  if not gradients then return nil end
  --
  local count = self:get_count()
  for wname,w in pairs(weights) do
    local Egradient   = self.Egradients[wname] or matrix.as(w):zeros()
    local grad        = gradients[wname]
    -- learning options
    local lr          = self:get_option_of(wname, "learning_rate")
    local decay       = self:get_option_of(wname, "decay")
    local eps         = self:get_option_of(wname, "epsilon")
    local l2          = self:get_option_of(wname, "weight_decay")
    local mnp         = self:get_option_of(wname, "max_norm_penalty")
    -- L2 regularization
    if l2 > 0.0 then grad:axpy(l2, w) end
    -- accumulate gradients
    if count == 0 then
      Egradient[{}] = grad^2
    else
      Egradient[{}] = decay*Egradient + (1-decay)*grad^2
    end
    -- compute update on grad matrix
    local update = mop.cmul(grad, 1 / (eps + mop.sqrt(Egradient)))
    -- apply update matrix to the weights
    w:axpy(-lr, update)
    -- constraints
    if mnp > 0.0 then ann.optimizer.utils.max_norm_penalty(w, mnp) end
    -- weights normality check
    if count % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      w:prune_subnormal_and_check_normal()
    end
    --
    self.Egradients[wname] = Egradient
  end
  -- count one more update iteration
  self:count_one()
  -- returns the same as returned by eval()
  return table.unpack(arg)
end

function adagrad_methods:clone()
  local obj = ann.optimizer.adagrad()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  obj.Egradients        = md.clone( self.Egradients )
  return obj
end

function adagrad_methods:to_lua_string(format)
  local format = format or "binary"
  local str_t = { "ann.optimizer.adagrad(",
		  table.tostring(self.global_options),
		  ",",
		  table.tostring(self.layerwise_options),
		  ",",
		  tostring(self.count),
		  ",",
		  util.to_lua_string(self.Egradients, format),
		  ")" }
  return table.concat(str_t, "")
end

local adagrad_properties = {
  gradient = true
}
function adagrad_methods:needs_property(property)
  return adagrad_properties[property]
end
