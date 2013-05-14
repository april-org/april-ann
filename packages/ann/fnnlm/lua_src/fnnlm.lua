class("ann.fnnlm")

function ann.fnnlm:__call(t)
  local params = get_table_fields(
    {
      factors = { mandatory=true, type_match="table",
		  getter = get_table_fields_ipairs{
		    name   = { mandatory = true, type_match = "string" },
		    layers = { mandatory = true, type_match = "table",
			       getter = get_table_fields_ipairs{
				 size = { mandatory = true, type_match = "number" },
				 actf = { mandatory = true, type_match = "string" },
			       },
		    },
		    order  = { mandatory = true, type_match = "number" },
		  },
      },
      cache_size  = { mandatory = false, type_match = "number", default = 0 },
      output_size = { mandatory = true,  type_match = "number" },
      hidden_actf = { mandatory = true,  type_match = "string" },
      hidden_size = { mandatory = true,  type_match = "number" },
      bunch_size  = { mandatory = false, type_match = "number", default = 32 },
    }, t)
  local obj = { 
    factor_names      = {},
    factor_components = {},
    cache_component   = {}
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
    local stack = ann.components.stack{ name=factor.name.."-stack" }
    local size
    for i=2,#factor.layers do
      local prevsize = factor.layers[i-1].size
      size = factor.layers[i].size
      local actf = factor.layers[i].actf
      local prefix = "factor_" .. factor.name .. "_" .. (i-1) .. "_"
      stack:push( ann.components.hyperplane{
		    input  = prevsize,
		    output = size,
		    name   = prefix .. "layer",
		    dot_product_name    = prefix .. "w",
		    bias_name           = prefix .. "b",
		    dot_product_weights = prefix .. "w",
		    bias_weights        = prefix .. "b", })
      stack:push(ann.components.actf[actf]{name=prefix.."actf"})
    end
    projection_layer_size = projection_layer_size + size*(factor.order-1)
    table.insert(obj.factor_components, stack)
    obj.input_component:add(stack)
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
  obj.output_component:push(ann.components.actf.softmax{name="output_actf"})
  --
  obj.ann_component:push(obj.input_component)
  obj.ann_component:push(obj.hidden_component)
  obj.ann_component:push(obj.output_component)
  --
  obj.trainer =
    trainable.supervised_trainer(obj.ann_component,
				 ann.loss.multi_class_cross_entropy(params.output_size),
				 params.bunch_size)
end
--
obj = class_instance(obj, fnnlm, true)
return obj
end

function ann.fnnlm:get_component()
  return self.ann_component
end

function ann.fnnlm:set_dropout(value)
  for name,component in obj.trainer:iterate_components("^.*actf.*$") do
    if not name:match("^factor_.*_1_actf$") then
      component:set_option("dropout",value)
    end
  end
end
