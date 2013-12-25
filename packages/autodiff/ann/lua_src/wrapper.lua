local AD = autodiff

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

local component_methods,
component_class_metatable = class("autodiff.ann.component",
				  "ann.components.base")

function component_class_metatable:__call(name, net, input,
					  input_size, output_size)
  local obj = {
    name        = name,
    net         = net,
    input       = input,
    input_size  = input_size or 0,
    output_size = output_size or 0,
    seed        = AD.matrix(string.format("seed-%s", name)),
    cache       = {},
  }
  return class_instance(obj, self)
end

function component_methods:build(tbl)
  self.weights = assert(tbl.weights, "Needs a table with weights field")
  local diff_tbl = { }
  for k,w in pairs(self.weights) do
    local s = AD.get(k)
    s:set_dims(w:dim())
    table.insert(diff_tbl, s)
  end
  self.diff_tbl = diff_tbl
  -- forward function
  self.func,self.program = AD.func(self.net, {self.input}, self.weights)
  -- diffentiation
  self.dw_tbl = table.pack( AD.diff(self.net, diff_tbl, self.seed) )
  -- backward function
  self.dw_func,self.dw_program = AD.func(self.dw_tbl, {self.input,self.seed},
					 self.weights)
  --
  self.input_size  = tbl.input  or self.input_size
  self.output_size = tbl.output or self.output_size
  return self,tbl.weights,{ [self.name] = self }
end

function component_methods:backprop(seed_token)
  assert(self.input_matrix, "Call forward before backprop")
  local seed_matrix = seed_token:get_matrix():transpose()
  self.seed_token   = seed_token
  local aux = table.pack( self.dw_func(self.input_matrix, seed_matrix,
				       self.cache) )
  self.grads = {}
  for i,s in ipairs(self.diff_tbl) do self.grads[s.name] = aux[i] end
  -- return nil
end

function component_methods:calculate(input_token)
  return self:forward(input_token)
end

function component_methods:clone()
  local new_obj = AD.ann.component(self.name, self.net, self.input,
				   self.input_size, self.output_size)
  new_obj:build{
    weights = iterator(pairs(self.weights)):
    map(function(name,m)return name,m:clone() end):table()
  }
  return new_obj
end

function component_methods:compute_gradients()
  assert(self.grads, "Call backprop before compute_gradients")
  return self.grads
end

function component_methods:copy_components()
  return { [self.name] = self }
end

function component_methods:copy_weights()
  return self.weights
end

function component_methods:debug_info()
end

function component_methods:forward(input_token,during_training)
  local input_matrix  = input_token:get_matrix():transpose()
  self.input_matrix   = input_matrix
  self.input_token    = input_token
  local output_matrix = self.func(input_matrix, self.cache)
  self.output_token   = tokens.matrix(output_matrix:transpose())
  return self.output_token
end

function component_methods:get_component()
  return self
end

function component_methods:get_error_input()
  return self.seed_token
end

function component_methods:get_input()
  return self.input_token
end

function component_methods:get_output()
  return self.output_token
end

function component_methods:get_input_size()
  return self.input_size
end

function component_methods:get_output_size()
  return self.input_size
end

function component_methods:get_is_built()
  return self.weights
end

function component_methods:get_name()
  return self.name
end

function component_methods:get_use_cuda()
  return false
end

function component_methods:get_weights_name()
  return
end

function component_methods:has_weights_name()
  return false
end

function component_methods:precompute_output_size()
  return self.output_size
end

function component_methods:reset(count)
  local count = count or 0
  if count == 0 then table.clear(self.cache) end
end

function component_methods:set_use_cuda()
end

function component_methods:to_lua_string()
  error("Not implemented")
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

local loss_methods,
loss_class_metatable = class("autodiff.ann.loss", "ann.loss")

function loss_class_metatable:__call(loss, input, target)
  local obj = {
    loss     = loss,
    target   = target,
    input    = input,
    mean_var = stats.mean_var(),
    cache    = (net or {}).cache or {},
  }
  return class_instance(obj, self)
end

function loss_methods:compile(weights, net)
  if net then self.cache = net.cache or self.cache end
  self.weights = weights
  -- forward
  self.func,self.program = AD.func(self.loss, {self.input,self.target},
				   self.weights)
  -- differentiation
  self.dL = AD.diff(self.loss, self.input)
  -- backward
  self.dL_func,self.dL_program = AD.func(self.dL, {self.input,self.target},
					 self.weights)
  --
end

function loss_methods:accum_loss(loss,loss_matrix)
  if not loss_matrix then loss_matrix = loss end
  for i=1,loss_matrix:size() do self.mean_var:add(loss_matrix:get(i)) end
  if loss~=loss_matrix then
    return loss,loss_matrix
  else
    return loss_matrix
  end
end

function loss_methods:clone()
  local new_obj = AD.ann.loss(self.loss, self.input, self.target)
  return new_obj
end

function loss_methods:compute_loss(input,target)
  self.input_token   = input
  self.target_token  = target
  self.input_matrix  = input:get_matrix():transpose()
  self.target_matrix = target:get_matrix():transpose()
  local loss_matrix  = matrix.col_major(1)
  local loss         = self.func(self.input_matrix, self.target_matrix,
				 self.cache)
  loss_matrix:set(1,loss)
  return loss,loss_matrix
end

function loss_methods:get_accum_loss()
  return self.mean_var:compute()
end

function loss_methods:gradient(input,target)
  if self.input_token ~= input then self:compute_loss(input,target) end
  for i,v in pairs(self.cache) do print(i) end
  self.grads = self.dL_func(self.input_matrix, self.target_matrix, self.cache)
  return tokens.matrix(self.grads:transpose())
end

function loss_methods:reset()
  table.clear(self.cache)
  self.mean_var:clear()
end

function loss_methods:to_lua_string()
  error("Not implemented")
end
