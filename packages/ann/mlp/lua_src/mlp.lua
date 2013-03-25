ann = ann or {}
ann.mlp = ann.mlp or {}

function ann.mlp.convert_to_all_all(t)
  local net        = t.ann or error ("Needs an ann field")
  local topology   = t.topology or error ("Needs a topology string field")
  local bunch_size = t.bunch_size or error ("Needs a bunch_size field")
  local all_all_net = ann.mlp.all_all.generate{ bunch_size = bunch_size,
						topology   = topology,
						random     = random(5825276) }
  if net:get_layer_connections_size() ~= all_all_net:get_layer_connections_size() then
    error("Incorrect topology description")
  end
  for i=1,net:get_layer_connections_size() do
    local old_w = net:get_layer_connections(i)
    local new_w = all_all_net:get_layer_connections(i)
    if old_w:size() ~= new_w:size() then
        error("Mismatch between connection sizes!! check that "..
              "bias weights are registered before matrix weights")
    end
    new_w:load{ w = old_w:weights() }
  end
  return all_all_net
end

function ann.mlp.add_layers(t)
  local new_net = t.ann:clone()
  new_net:release_output()
  local prev_units = new_net:get_layer_activations(new_net:get_layer_activations_size())
  for i=1,#t.new_layers do
    local layer_type = "hidden"
    if i == #t.new_layers then layer_type = "outputs" end
    local actf   = ann.activations.from_string(t.new_layers[i][2])
    local size   = t.new_layers[i][1]
    local input  = prev_units
    local output = ann.units.real_cod{ ann = new_net,
				       size = size,
				       type = layer_type }
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
    if t.random then
      bias:randomize_weights{ random=t.random,
			      inf=t.inf,
			      sup=t.sup,
			      use_fanin = false }
      weights:randomize_weights{ random=t.random,
				 inf=t.inf,
				 sup=t.sup,
				 use_fanin = false}
    end
    if t.bias_table then bias:load{ w=t.bias_table[i] } end
    if t.weights_table then weights:load{ w=t.weights_table[i] } end
    prev_units = units
  end
  return new_net
end
