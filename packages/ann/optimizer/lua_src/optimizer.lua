local optimizer_methods,
optimizer_class_metatable = class("ann.optimizer")

function optimizer_class_metatable:__call()
  return class_instance({
			  global_options = {},
			  layerwise_options = {},
			  count = 0,
			},
			self)
end

function optimizer_methods:set_option(name,value)
  self.global_options[name] = value
end

function optimizer_methods:get_option(name,value)
  return self.global_options[name]
end

function optimizer_methods:set_layerwise_option(layer_name,name,value)
  self.layerwise_options[layer_name] = self.layerwise_options[layer_name] or {}
  self.layerwise_options[layer_name][name] = value
end

function optimizer_methods:get_layerwise_option(layer_name,name)
  return (self.layerwise_options[layer_name] or {})[name]
end

function optimizer_methods:get_option_of(layer_name,name)
  return ( (self.layerwise_options[layer_name] or {})[name] or
	     self.global_options[layer_name] )
end

-- eval is a function which returns the loss, the function result (a matrix) and
-- its derivatives, and second derivatives (if needed)
--
-- cnn_table is a dictionary of connections objects, indexed by its names.
function optimizer_methods:execute(eval, cnn_table)
  self:count_one()
  return eval()
end

function optimizer_methods:count_one()
  self.count = self.count + 1
end

function optimizer_methods:get_count()
  return self.count
end

------------------------------------------------
--------- STOCHASTIC GRADIENT DESCENT ----------
------------------------------------------------

local sgd_methods, sgd_class_metatable = class("ann.optimizer.sgd",ann.optmizer)

function sgd_class_metatable:__call()
  local obj = ann.optimizer()
  return class_instance(obj, self)
end

function sgd_methods:execute(eval, cnn_table)
  local loss,output,gradients = eval()
  local bunch_size = output:dim(1) -- mini-batch or bunch size
  for cname,cnn in ipairs(cnn_table) do
    local w,oldw     = cnn:matrix()
    local grad       = gradients[cname]
    local N          = cnn:get_shared_count()
    local lr         = self:get_option_of(cname, "learning_rate")
    local mt         = self:get_option_of(cname, "momentum")     or 0.0
    local wd         = self:get_option_of(cname, "weight_decay") or 0.0
    local cwd        = 1.0 - wd
    local beta       = 1.0
    if mt > 0.0 then
      -- intertia is computed between current and old weight matrices, but with
      -- inverted sign
      oldw:axpy(-1.0, w)
      -- apply momentum term to inertia computation, with inverted sign
      oldw:scal(-mt)
      -- sum current weights by applying the complementary of weight decay term
      oldw:axpy(cwd, w)
    else
      -- if no momentum, oldw is a copy of current weights
      oldw:copy(w)
      if cwd < 1.0 then
	-- apply weight decay if needed
	oldw:scal(cwd)
      end
    end
    -- apply backpropagation learning rule
    local norm_lr_rate = -1.0/math.sqrt( N * bunch_size ) * lr
    oldw:axpy(norm_lr_rate, grad)
    -- swap current and old weight matrices
    cnn:swap()
  end
  -- count one more update iteration
  self:count_one()
  return loss
end
