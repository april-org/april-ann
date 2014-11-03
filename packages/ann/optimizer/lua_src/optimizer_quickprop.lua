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

------------------------------
--------- QUICKPROP ----------
------------------------------

local quickprop, quickprop_methods = class("ann.optimizer.quickprop",
                                           ann.optimizer)
ann.optimizer.quickprop = quickprop

function quickprop:constructor(g_options, l_options, count, update, lastg)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
                              {"learning_rate", "Learning speed factor (0.01)"},
                              {"mu", "Maximum growth factor (1.75)"},
                              {"epsilon", "Bootstrap factor (1e-04)"},
			      {"max_step", "Maximum step value (1000)"},
                              {"decay", "Decay of hyper-parameters (1e-05), global option"},
                              {"weight_decay", "Weight L2 regularization (0.0)"},
                              {"L1_norm", "Weight L1 regularization (0.0)"},
                              {"max_norm_penalty", "Weight max norm upper bound (0)"},
                            },
			    g_options,
			    l_options,
			    count)
  -- default values
  if not g_options then
    self:set_option("learning_rate", 0.01)
    self:set_option("mu", 1.75)
    self:set_option("epsilon", 1e-04)
    self:set_option("max_step", 1000)
    self:set_option("decay", 1e-05)
    self:set_option("weight_decay", 0.0)
    self:set_option("L1_norm", 0.0)
    self:set_option("max_norm_penalty", 0.0)
  end
  self.update = wrap_matrices(update or matrix.dict())
  self.lastg  = wrap_matrices(lastg  or matrix.dict())
end

function quickprop_methods:execute(eval, weights)
  local wrap_matrices = wrap_matrices
  local table = table
  local assert = assert
  local math = math
  --
  local origw = weights
  local weights = wrap_matrices(weights)
  local arg = table.pack( eval(origw) )
  local tr_loss,gradients = table.unpack(arg)
  -- the gradient computation could fail returning nil, it is important to take
  -- this into account
  if not gradients then return nil end
  gradients = wrap_matrices(gradients)
  local d0 = self:get_option("decay")
  local decay = 1.0 / (1.0 + d0 * self:get_count())
  for wname,w in pairs(weights) do
    local update      = self.update(wname)
    local lastg       = self.lastg(wname)
    local grad        = gradients(wname)
    -- learning options
    local lr          = self:get_option_of(wname, "learning_rate")
    local lrd         = lr * decay
    local mu          = self:get_option_of(wname, "mu")
    local epsilon     = self:get_option_of(wname, "epsilon")
    local max_step    = self:get_option_of(wname, "max_step")
    local l1          = self:get_option_of(wname, "L1_norm")
    local l2          = self:get_option_of(wname, "weight_decay")
    local mnp         = self:get_option_of(wname, "max_norm_penalty")
    assert(self:get_option_of(wname, "decay") == d0,
           "decay option cannot be defined layerwise, only globally")
    if not update then
      -- compute standard back-propagation learning rule
      update = w:clone()
      lastg  = grad:clone()
      update:copy(grad)
    else
      local shrink = mu / (1.0 + mu)
      -- compute quickprop update
      update:map(lastg, grad,
		 function(prev_step, prev_slope, slope)
		   local step=0
		   if math.abs(prev_step) > 1e-03 then
		     if math.sign(slope) == math.sign(prev_step) then
		       step = step + epsilon * slope
		     end
		     if ( (prev_step > 0 and slope > shrink*prev_slope) or
                       (prev_step < 0 and slope < shrink*prev_slope) ) then
		       step = step + mu * prev_step
		     else
		       step = step + (prev_step*slope) / (prev_slope - slope)
		     end
		   else
		     step = step + epsilon * slope
		   end
		   if step > max_step then step = max_step
		   elseif step < -max_step then
		     step = -max_step
		   end
		   return step
      end)
      lastg:copy(grad)
    end
    --
    self.update[wname] = update
    self.lastg[wname]  = lastg
    -- L2 regularization
    if l2 > 0.0 then update:axpy(l2, w) end
    -- apply update matrix to the weights
    w:axpy(-lrd, update)
    -- L1 regularization, truncated gradient implementation
    if l1 > 0.0 then ann.optimizer.utils.l1_truncate_gradient(w, lrd*l1) end
    -- constraints
    if mnp > 0.0 then ann.optimizer.utils.max_norm_penalty(w, mnp) end
    -- weights normality check
    if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      w:prune_subnormal_and_check_normal()
    end
  end
  -- count one more update iteration
  self:count_one()
  -- returns the same as returned by eval()
  return table.unpack(arg)
end

function quickprop_methods:clone()
  local obj = ann.optimizer.quickprop()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  obj.update            = self.update:clone()
  obj.lastg             = self.lastg:clone()
  return obj
end

function quickprop_methods:to_lua_string(format)
  local format = format or "binary"
  local str_t = { "ann.optimizer.quickprop(",
		  table.tostring(self.global_options),
		  ",",
		  table.tostring(self.layerwise_options),
		  ",",
		  tostring(self.count),
		  ",",
		  self.update:to_lua_string(format),
		  ",",
		  self.lastg:to_lua_string(format),
		  ")" }
  return table.concat(str_t, "")
end

local quickprop_properties = {
  gradient = true
}
function quickprop_methods:needs_property(property)
  return quickprop_properties[property]
end

