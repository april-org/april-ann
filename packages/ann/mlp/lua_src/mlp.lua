ann = ann or {}
ann.mlp = ann.mlp or {}

function ann.mlp.add_layers(t)
  local new_net = t.ann:clone()
  new_net:release_output()
  local prev_units = new_net:get_layer_activations(new_net:get_layer_activations_size())
  for i=1,#t.new_layers do
    if i == #t.new_layers then type = "outputs" end
    local actf   = ann.activations.from_string(t.new_layers[i][2])
    local size   = t.new_layers[i][1]
    local input  = prev_units
    local output = ann.units.real_cod{ ann = new_net,
				       size = size,
				       type = type }
    local bias   = ann.connections.bias{ ann  = new_net,
					 size = size }
    local weights = ann.connections.all_all{
      ann         = new_net,
      input_size  = input:num_neurons(),
      output_size = size }
    ann.actions.forward_bias{ ann = new_net,
			      output = output,
			      connections = bias }
    ann.actions.dot_product{ ann = new_net,
			     input = input,
			     output = output,
			     connections = weights }
    if actf then
      ann.actions.activations{ ann = new_net,
			       output = output,
			       actfunc = actf }
    end
    bias:randomize_weights{ random=t.random,
			    inf=t.inf,
			    sup=t.sup,
			    use_fanin = false }
    weights:randomize_weights{ random=t.random,
			       inf=t.inf,
			       sup=t.sup,
			       use_fanin = false}
    prev_units = units
  end
  return new_net
end
