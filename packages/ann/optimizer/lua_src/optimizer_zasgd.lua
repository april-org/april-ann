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
local wrap_matrices = matrix.dict.wrap_matrices

local MAX_UPDATES_WITHOUT_PRUNE = ann.optimizer.MAX_UPDATES_WITHOUT_PRUNE

---------------------------------------------------------
--------- AVERAGED STOCHASTIC GRADIENT DESCENT ----------
---------------------------------------------------------

-- extracted from: http://research.microsoft.com/pubs/192769/tricks-2012.pdf
-- Leon Bottou, Stochastic Gradient Descent Tricks, Microsoft Research, 2012
local asgd, asgd_methods = class("ann.optimizer.asgd", ann.optimizer)
ann.optimizer.asgd = asgd

function asgd:constructor(g_options, l_options, count, aw)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
			      {"learning_rate", "Learning speed factor (0.01)"},
			      {"lr_decay", "Learning decay factor (0.75)"},
			      {"t0", "Average starts at bunch t0, good values are data size or data dimension (0)"},
                              {"weight_decay", "Weight L2 regularization (0.0)"},
			    },
			    g_options,
			    l_options,
			    count)
  self.aw = wrap_matrices(aw or matrix.dict())
  if not g_options then
    -- default values
    self:set_option("learning_rate", 0.01)
    self:set_option("lr_decay", 0.75)
    self:set_option("t0", 0)
    self:set_option("weight_decay", 0.0)
  end
end

function asgd_methods:execute(eval, weights)
  local wrap_matrices = wrap_matrices
  local table = table
  local assert = assert
  --
  local weights = wrap_matrices(weights)
  local arg = table.pack( eval(weights) )
  local tr_loss,gradients = table.unpack(arg)
  -- the gradient computation could fail returning nil, it is important to take
  -- this into account
  if not gradients then return nil end
  gradients = wrap_matrices(gradients)
  local t = self:get_count()
  for wname,w in pairs(weights) do
    local aw          = self.aw(wname) or w:clone():zeros()
    local grad        = gradients(wname)
    -- learning options
    local lr          = self:get_option_of(wname, "learning_rate")
    local lr_decay    = self:get_option_of(wname, "lr_decay")
    local t0          = self:get_option_of(wname, "t0")
    local l2          = self:get_option_of(wname, "weight_decay")
    -- effective values at time t
    local lr_t        = lr / ((1.0 + l2 * lr * t)^(lr_decay)) -- learning rate factor
    local mu_t        = 1.0 / math.max(1, t - t0)             -- average factor
    -- L2 regularization
    if l2 > 0.0 then grad:axpy(l2, w) end
    -- apply back-propagation learning rule
    w:axpy(-lr_t, grad)
    if mu_t ~= 1 then
      -- compute averaged weights
      aw:axpy(mu_t, w - aw)
    else
      -- just copy last weight values
      aw:copy(w)
    end
    -- weights normality check
    if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      w:prune_subnormal_and_check_normal()
    end
    --
    self.aw[wname] = aw
  end
  -- count one more update iteration
  self:count_one()
  -- returns the same as returned by eval() plus the averaged weights
  table.insert(arg, self.aw)
  return table.unpack(arg)
end

function asgd_methods:clone()
  local obj = ann.optimizer.asgd()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  obj.aw                = self.aw:clone()
  return obj
end

function asgd_methods:to_lua_string(format)
  local format = format or "binary"
  local str_t = { "ann.optimizer.asgd(",
		  table.tostring(self.global_options),
		  ",",
		  table.tostring(self.layerwise_options),
		  ",",
		  tostring(self.count),
		  ",",
		  self.aw:to_lua_string(format),
		  ")" }
  return table.concat(str_t, "")
end

local asgd_needs_properties = {
  gradient = true,
}
function asgd_methods:needs_property(property)
  return asgd_needs_properties[property]
end

local asgd_has_properties = {
  average = true
}
function asgd_methods:has_property(property)
  return asgd_has_properties[property]
end

function asgd_methods:get_averaged_weights()
  return self.aw
end

