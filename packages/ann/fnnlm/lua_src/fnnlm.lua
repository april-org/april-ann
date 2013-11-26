local ann_fnnlm_methods, ann_fnnlm_class_metatable = class("ann.fnnlm")

function ann_fnnlm_class_metatable:__call(t)
  local params = get_table_fields(
    {
      factors = {
	mandatory=true, type_match="table",
	getter = get_table_fields_ipairs{
	  name   = { mandatory = true, type_match = "string" },
	  layers = { mandatory = true, type_match = "table",
		     getter = get_table_fields_ipairs{
		       size = { mandatory = true,
				type_match = "number" },
		       actf = { mandatory = false,
				type_match = "string",
				default="linear" },
		     },
	  },
	  order = { mandatory = true, type_match = "number" },
	  convolution = { mandatory=false, type_match="number",
			  default=0 },
	},
      },
      cache_size  = { mandatory = false, type_match = "number", default = 0 },
      output_size = { mandatory = true,  type_match = "number" },
      -- convolution = { mandatory = false, type_match = "table",
      -- 		      getter = get_table_fields_recursive{
      -- 			size  = { mandatory = true, type_match = "number" },
      -- 			actf  = { mandatory = true, type_match = "number" },
      -- 			order = { mandatory = true, type_match = "number" },
      -- 		      }
      -- },
      hidden_actf = { mandatory = false, type_match = "string", default="tanh" },
      hidden_size = { mandatory = true,  type_match = "number" },
      bunch_size  = { mandatory = false, type_match = "number", default = 32 },
    }, t)
  --
  local obj = { 
    factor_names      = {},
    factor_components = {},
    cache_component   = {},
    input_component   = ann.components.join{  name="factors_join" },
    hidden_component  = ann.components.stack{ name="hidden_stack" },
    output_component  = ann.components.stack{ name="output_stack" },
    ann_component     = ann.components.stack{ name="FNNLM" },
    trainer           = {},
    params            = table.deep_copy(params),
  }
  local projection_layer_size = 0
  for _,factor in ipairs(params.factors) do
    table.insert(obj.factor_names, factor.name)
    local join = ann.components.join{ name=factor.name.."-join" }
    local size
    for pos=1,factor.order do
      local stack = ann.components.stack{ name=factor.name.."-stack-"..pos }
      for i=2,#factor.layers do
	local prevsize = factor.layers[i-1].size
	size = factor.layers[i].size
	local actf = factor.layers[i].actf
	local prefixc = "factor_" .. factor.name .. "_" .. pos .. "_" .. (i-1) .. "_"
	local prefixw = "factor_" .. factor.name .. "_" .. (i-1) .. "_"
	stack:push( ann.components.hyperplane{
		      input  = prevsize,
		      output = size,
		      name   = prefixc .. "layer",
		      dot_product_name    = prefixc .. "w",
		      bias_name           = prefixc .. "b",
		      dot_product_weights = prefixw .. "w",
		      bias_weights        = prefixw .. "b", })
	stack:push(ann.components.actf[actf]{name=prefixc.."actf"})
      end
      join:add(stack)
    end
    projection_layer_size = projection_layer_size + size*factor.order
    table.insert(obj.factor_components, join)
    obj.input_component:add(join)
  end
  --
  if params.cache_size > 0 then
    projection_layer_size = projection_layer_size + params.cache_size
    obj.cache_component   = ann.components.base{ size=params.cache_size,
						 name="cache" }
    obj.input_component:add( obj.cache_component )
  end
  --
  obj.hidden_component:push( ann.components.hyperplane{
			       input  = projection_layer_size,
			       output = params.hidden_size,
			       name   = "hidden_layer",
			       dot_product_name    = "hidden_w",
			       bias_name           = "hidden_b",
			       dot_product_weights = "hidden_w",
			       bias_weights        = "hidden_b", })
  obj.hidden_component:push(ann.components.actf[params.hidden_actf]{name="hidden_actf"})
  --
  obj.output_component:push( ann.components.hyperplane{
			       input  = params.hidden_size,
			       output = params.output_size,
			       name   = "output_layer",
			       dot_product_name    = "output_w",
			       bias_name           = "output_b",
			       dot_product_weights = "output_w",
			       bias_weights        = "output_b", })
  obj.output_component:push(ann.components.actf.log_softmax{name="output_actf"})
  --
  obj.ann_component:push(obj.input_component)
  obj.ann_component:push(obj.hidden_component)
  obj.ann_component:push(obj.output_component)
  --
  obj = class_instance(obj, self)
  obj = class_wrapper(obj.ann_component, obj)
  --
  return obj
end

function ann_fnnlm_methods:get_trainer()
  return trainable.supervised_trainer(self,
				      ann.loss.multi_class_cross_entropy(self.params.output_size),
				      self.params.bunch_size)
end

function ann_fnnlm_methods:set_dropout(value)
  for name,component in obj.trainer:iterate_components("^.*actf.*$") do
    if not name:match("^factor_.*_1_actf$") then
      component:set_option("dropout",value)
    end
  end
end

function ann_fnnlm_methods:clone()
  local obj = ann.fnnlm(self.params)
  obj:build{ weights = table.map(self:copy_weights(),
				 function(cnn) return cnn:clone() end) }
  return obj
end

function ann_fnnlm_methods:forward(t)
  if type(t) == "table" then
    t = (type(t[1]) == "table" and t) or { t }
    local bunch_token = tokens.vector.bunch(#t)
    for b=1,#t do
      local k = 1
      local factors_token = tokens.vector.bunch(#self.params.factors)
      for f=1,#self.params.factors do
	local token = tokens.vector.bunch(self.params.factors[f].order)
	for i=1,self.params.factors[f].order do
	  assert(t[b][k], "Found nil value at table position " .. k)
	  local sparse = tokens.vector.sparse(1):set(1, t[b][k], 1.0)
	  token:set(i, sparse)
	  k = k + 1
	end
	factors_token:set(f, token)
      end
      bunch_token:set(b, factors_token)
    end
    t = bunch_token
  end
  return self.ann_component:forward(t)
end
