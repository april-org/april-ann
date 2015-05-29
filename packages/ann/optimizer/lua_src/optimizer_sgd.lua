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

------------------------------------------------
--------- STOCHASTIC GRADIENT DESCENT ----------
------------------------------------------------

local sgd, sgd_methods = class("ann.optimizer.sgd", ann.optimizer)
ann.optimizer.sgd = sgd -- global environment

function sgd:constructor(g_options, l_options, count, update)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
			      {"learning_rate", "Learning speed factor (0.01)"},
			      {"momentum", "Learning inertia factor (0.0)"},
                              {"decay", "Decay of hyper-parameters (1e-05), global option"},
                              {"weight_decay", "Weight L2 regularization (0.0)"},
                              {"L1_norm", "Weight L1 regularization (0.0)"},
                              {"max_norm_penalty", "Weight max norm upper bound (0)"},
			    },
			    g_options,
			    l_options,
			    count)
  self.update = update or {}
  if not g_options then
    -- default values
    self:set_option("learning_rate", 0.01)
    self:set_option("momentum", 0.0)
    self:set_option("decay", 1e-05)
    self:set_option("weight_decay", 0.0)
    self:set_option("L1_norm", 0.0)
    self:set_option("max_norm_penalty", 0.0)
  end
end

function sgd_methods:execute(eval, weights)
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
  local d0 = self:get_option("decay")
  local decay = 1.0 / (1.0 + d0 * self:get_count())
  --
  for wname,w in pairs(weights) do
    local update      = self.update[wname] or matrix.as(w):zeros()
    local grad        = gradients[wname]
    -- learning options
    local lr          = self:get_option_of(wname, "learning_rate")
    local lrd         = lr * decay
    local mt          = self:get_option_of(wname, "momentum")
    local l1          = self:get_option_of(wname, "L1_norm")
    local l2          = self:get_option_of(wname, "weight_decay")
    local mnp         = self:get_option_of(wname, "max_norm_penalty")
    assert(self:get_option_of(wname, "decay") == d0,
           "decay option cannot be defined layerwise, only globally")
    -- L2 regularization
    if l2 > 0.0 then grad:axpy(l2, w) end
    -- momentum
    if mt > 0.0 then update:scal(mt) else update:zeros() end
    -- apply back-propagation learning rule to update matrix
    update:axpy(lrd, grad)
    -- apply update matrix to the weights
    w:axpy(-1.0, update)
    -- L1 regularization, truncated gradient implementation
    if l1 > 0.0 then ann.optimizer.utils.l1_truncate_gradient(w, lrd*l1,
                                                              update) end
    -- constraints
    if mnp > 0.0 then ann.optimizer.utils.max_norm_penalty(w, mnp) end
    -- weights normality check
    if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      w:prune_subnormal_and_check_normal()
    end
    --
    self.update[wname] = update
  end
  -- count one more update iteration
  self:count_one()
  -- returns the same as returned by eval()
  return table.unpack(arg)
end

function sgd_methods:clone()
  local obj = ann.optimizer.sgd()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  obj.update            = md.clone( self.update )
  return obj
end

function sgd_methods:ctor_name()
  return "ann.optimizer.sgd"
end
function sgd_methods:ctor_params()
  return self.global_options,
  self.layerwise_options,
  self.count,
  self.update
end

local sgd_properties = {
  gradient = true
}
function sgd_methods:needs_property(property)
  return sgd_properties[property]
end
