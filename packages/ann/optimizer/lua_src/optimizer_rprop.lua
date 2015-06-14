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
--------- RESILIENT PROP ----------
-----------------------------------

local rprop, rprop_methods = class("ann.optimizer.rprop", ann.optimizer)
ann.optimizer.rprop = rprop

function rprop:constructor(g_options, l_options, count, steps, old_signs)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
			      {"initial_step", "Initial weight update value (0.1)"},
			      {"eta_plus", "Update value up by this factor (1.2)"},
			      {"eta_minus", "Update value down by this factor (0.5)"},
			      {"max_step", "Maximum value of update step (50)"},
			      {"min_step", "Minimum value of update step (1e-05)"},
			      {"niter", "Number of iterations (1)"},
                              {"weight_decay", "Weight L2 regularization (0.0)"},
                              {"L1_norm", "Weight L1 regularization (0.0)"},
                              {"max_norm_penalty", "Weight max norm upper bound (0)"},
			    },
			    g_options,
			    l_options,
			    count)
  self.steps     = steps or {}
  self.old_signs = old_signs or {}
  if not g_options then
    -- default values
    self:set_option("initial_step",  0.1)
    self:set_option("eta_plus",      1.2)
    self:set_option("eta_minus",     0.5)
    self:set_option("max_step",      50)
    self:set_option("min_step",      1e-05)
    self:set_option("niter",         1)
    self:set_option("weight_decay",  0.0)
    self:set_option("L1_norm",       0.0)
    self:set_option("max_norm_penalty", 0.0)
  end
end

function rprop_methods:execute(eval, weights)
  local table = table
  local assert = assert
  --
  local origw         = weights
  local initial_step  = self:get_option("initial_step")
  local eta_plus      = self:get_option("eta_plus")
  local eta_minus     = self:get_option("eta_minus")
  local max_step      = self:get_option("max_step")
  local min_step      = self:get_option("min_step")
  local niter         = self:get_option("niter")
  local steps         = self.steps
  local old_signs     = self.old_signs
  local arg
  for i=1,niter do
    arg = table.pack( eval(origw, i-1) )
    local tr_loss,gradients = table.unpack(arg)
    -- the gradient computation could fail returning nil, it is important to
    -- take this into account
    if not gradients then return nil end
    --
    for wname,w in pairs(weights) do
      local grad        = gradients[wname]
      local old_sign    = old_signs[wname]
      local step        = steps[wname] or w:clone():fill(initial_step)
      -- learning options
      local l1          = self:get_option_of(wname, "L1_norm")
      local l2          = self:get_option_of(wname, "weight_decay")
      local mnp         = self:get_option_of(wname, "max_norm_penalty")
      --
      local sign  = mop.sign(grad)
      -- compute rprop learning step
      if old_sign then
	ann_optimizer_utils.rprop.step(step, old_sign, sign,
                                       eta_minus, eta_plus)
      end
      step:clamp(min_step, max_step)
      -- apply sign to rprop learning step
      local update = mop.cmul(sign, step)
      -- L2 regularization
      if l2 then update:axpy(l2, mop.cmul(step, w)) end
      -- apply update to weights
      w:axpy(-1.0, update)
      -- L1 regularization, truncated gradient implementation
      if l1 > 0.0 then
        ann.optimizer.utils.l1_truncate_gradient(w, mop.scal(step, l1))
      end
      -- constraints
      if mnp > 0.0 then ann.optimizer.utils.max_norm_penalty(w, mnp) end
      -- keep matrices for the next iteration
      old_signs[wname] = sign
      steps[wname]     = step
      -- weights normality check
      if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
	w:prune_subnormal_and_check_normal()
      end
    end
    -- count one more update iteration
    self:count_one()
  end
  -- returns the same as returned by eval()
  return table.unpack(arg)
end

function rprop_methods:clone()
  local obj = ann.optimizer.rprop()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  if self.steps then
    obj.steps = md.clone( self.steps )
  end
  if self.old_signs then
    obj.old_signs = md.clone( self.old_signs )
  end
  return obj
end

function rprop_methods:ctor_name()
  return "ann.optimizer.rprop"
end
function rprop_methods:ctor_params()
  return self.global_options,
  self.layerwise_options,
  self.count,
  self.steps,
  self.old_signs
end

local rprop_properties = {
  gradient = true
}
function rprop_methods:needs_property(property)
  return rprop_properties[property]
end
