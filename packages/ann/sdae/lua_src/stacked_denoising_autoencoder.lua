ann.autoencoders = ann.autoencoders or {}

-- This function builds a codifier from the weights of the first layer of a
-- restricted autoencoder
function ann.autoencoders.build_two_layered_codifier_from_weights(bunch_size,
								  input_size,
								  input_actf,
								  cod_size,
								  cod_actf,
								  bias_mat,weights_mat)
  local codifier = ann.mlp{ bunch_size = bunch_size }
  local input_layer  = ann.units.real_cod{ ann  = codifier,
					   size = input_size,
					   type = "inputs" }
  local cod_layer = ann.units.real_cod{ ann  = codifier,
					size = cod_size,
					type = "outputs" }
  local cod_bias  = ann.connections.bias{ ann  = codifier,
					  size = cod_size }
  local cod_weights = ann.connections.all_all{ ann         = codifier,
					       input_size  = input_size,
					       output_size = cod_size }
  ann.actions.forward_bias { ann         = codifier,
			     output      = cod_layer,
			     connections = cod_bias }
  ann.actions.dot_product { ann         = codifier,
			    input       = input_layer,
			    output      = cod_layer,
			    connections = cod_weights,
			    transpose   = false }
  if cod_actf then
    ann.actions.activations { ann     = codifier,
			      actfunc = cod_actf,
			      output  = cod_layer }
  end
  cod_bias:load{ w=bias_mat }
  cod_weights:load{ w=weights_mat }
  return codifier
end

-- This function builds an autoencoder of two layers (input-hidden-output) where
-- the hidden-output uses the same weights as input-hidden, so it is
-- simetric. The weights are initilitzed randomly but in the range
-- [-1/fanin, 1/fanin]
function ann.autoencoders.build_autoencoder_from_sizes_and_actf(bunch_size,
								input_size,
								input_actf,
								cod_size,
								cod_actf,
								weights_random)
  local autoencoder  = ann.mlp{ bunch_size = bunch_size }
  local input_layer  = ann.units.real_cod{ ann  = autoencoder,
					   size = input_size,
					   type = "inputs" }
  local hidden_layer = ann.units.real_cod{ ann  = autoencoder,
					   size = cod_size,
					   type = "hidden" }
  local output_layer = ann.units.real_cod{ ann  = autoencoder,
					   size = input_size,
					   type = "outputs" }
  local hidden_bias  = ann.connections.bias{ ann  = autoencoder,
					     size = cod_size }
  local inv_fanin = 1.0/input_size
  hidden_bias:randomize_weights{ random=weights_random,
				 inf=-inv_fanin,
				 sup= inv_fanin }
  local hidden_weights  = ann.connections.all_all{ ann         = autoencoder,
						   input_size  = input_size,
						   output_size = cod_size }
  hidden_weights:randomize_weights{ random=weights_random,
				    inf=-inv_fanin,
				    sup= inv_fanin }
  local output_bias = ann.connections.bias{ ann  = autoencoder,
					    size = input_size }
  local inv_fanin = 1.0/cod_size
  output_bias:randomize_weights{ random=weights_random,
				 inf=-inv_fanin,
				 sup= inv_fanin }
  ann.actions.forward_bias { ann         = autoencoder,
			     output      = hidden_layer,
			     connections = hidden_bias }
  ann.actions.dot_product { ann         = autoencoder,
			    input       = input_layer,
			    output      = hidden_layer,
			    connections = hidden_weights,
			    transpose   = false }
  if cod_actf then
    ann.actions.activations { ann     = autoencoder,
			      actfunc = cod_actf,
			      output  = hidden_layer }
  end
  ann.actions.forward_bias { ann         = autoencoder,
			     output      = output_layer,
			     connections = output_bias }
  ann.actions.dot_product { ann         = autoencoder,
			    input       = hidden_layer,
			    output      = output_layer,
			    connections = hidden_weights,
			    transpose   = true }
  if input_actf then
    ann.actions.activations { ann     = autoencoder,
			      actfunc = input_actf,
			      output  = output_layer }
  end
  return autoencoder
end

-- This functions receives layer sizes and layer weights and bias arrays. It
-- returns a fully connected stacked denoising autoencoder ANN.
function ann.autoencoders.build_full_autoencoder(bunch_size,
						 layers,
						 weights_mat,
						 bias_mat)
  local sdae = ann.mlp{ bunch_size = bunch_size }
  local neuron_layers = {}
  local actfs         = {}
  local weights_sdae  = {}
  local bias_sdae     = {}
  table.insert(neuron_layers, ann.units.real_cod{
		 ann  = sdae,
		 size = layers[1].size,
		 type = "inputs" })
  for i=2,#layers do
    table.insert(neuron_layers, ann.units.real_cod{
		   ann  = sdae,
		   size = layers[i].size,
		   type = "hidden" })
    table.insert(actfs, ann.activations.from_string(layers[i].actf))
    table.insert(bias_sdae, ann.connections.bias{
		   ann  = sdae,
		   size = layers[i].size })
    bias_sdae[#bias_sdae]:load{ w=bias_mat[i-1][1] }
    table.insert(weights_sdae, ann.connections.all_all{
		   ann = sdae,
		   input_size  = layers[i-1].size,
		   output_size = layers[i].size })
    weights_sdae[#bias_sdae]:load{ w=weights_mat[i-1] }
    ann.actions.forward_bias{ ann=sdae,
			      output=neuron_layers[#neuron_layers],
			      connections=bias_sdae[#bias_sdae] }
    ann.actions.dot_product{ ann=sdae,
			     input=neuron_layers[#neuron_layers-1],
			     output=neuron_layers[#neuron_layers],
			     connections=weights_sdae[#weights_sdae],
			     transpose = false }
    ann.actions.activations{ ann=sdae,
			     actfunc=actfs[#actfs],
			     output=neuron_layers[#neuron_layers] }
  end
  for i=#layers-1,1,-1 do
    table.insert(neuron_layers, ann.units.real_cod{
		   ann  = sdae,
		   size = layers[i].size,
		   type = (i>1 and "hidden") or "outputs" })
    table.insert(actfs, ann.activations.from_string(layers[i].actf))
    table.insert(bias_sdae, ann.connections.bias{
		   ann  = sdae,
		   size = layers[i].size })
    bias_sdae[#bias_sdae]:load{ w=bias_mat[i][2] }
    ann.actions.forward_bias{ ann=sdae,
			      output=neuron_layers[#neuron_layers],
			      connections=bias_sdae[#bias_sdae] }
    ann.actions.dot_product{ ann=sdae,
			     input=neuron_layers[#neuron_layers-1],
			     output=neuron_layers[#neuron_layers],
			     connections=weights_sdae[i],
			     transpose = true }
    ann.actions.activations{ ann=sdae,
			     actfunc=actfs[#actfs],
			     output=neuron_layers[#neuron_layers] }
  end
  return sdae
end

function ann.autoencoders.generate_training_table_configuration_from_params(current_dataset_params,
									    params)
  local data = {}
  if current_dataset_params.input_dataset then
    -- The input is perturbed with gaussian noise
    data.input_dataset = dataset.perturbation{
      dataset  = current_dataset_params.input_dataset,
      mean     = 0,
      variance = params.var,
      random   = params.perturbation_random }
    data.output_dataset = current_dataset_params.input_dataset
  end -- if params.input_dataset
  if current_dataset_params.distribution then
    data.distribution = {}
    for _,v in ipairs(current_dataset_params.distribution) do
      table.insert(data.distribution, {
		     -- The input is perturbed with gaussian noise
		     input_dataset = dataset.perturbation{
		       dataset  = v.input_dataset,
		       mean     = 0,
		       variance = params.var,
		       random   = params.perturbation_random },
		     output_dataset = v.input_dataset,
		     probability = v.prob })
    end -- for _,v in ipairs(params.distribution)
  end -- if params.distribution
  data.shuffle     = params.shuffle_random
  data.replacement = params.replacement
  return data
end

-- Params is a table which could contain:
--   * input_dataset => dataset with input (and output) for AE
--   * val_input_dataset => for validation
--   * distribution => a table which contains a list of {input_dataset=...., prob=....}
--   * replacement => replacement value for training
--   * shuffle_random => random number generator
--   * weights_random => random number generator
--   * perturbation_random => random number generator
--   * var => variance of gaussian noise
--   * layers => table which contains a list of { size=...., actf=....}, being
--               size a number and actf a string = "logistic"|"tanh"|"linear"
--   * bunch_size => size of mini-batch
--   * learning_rate
--   * momentum
--   * weight_decay
--   * max_epochs
--   * max_epochs_wo_improvement
--
-- This function returns a Stacked Denoising Auto-Encoder parameters table,
-- pretrained following algorithm of:
--
-- [CITE]
--
-- If you train an auto-encoder for a topology of 256 128 64
-- the WHOLE auto-encoder will had this topology:
-- 256 - 128 - 64 - 128 - 256
-- So it has four layers: (1) 256-128, (2) 128-64, (3) 64-128, (4) 128-256
--
-- Two arrays store weights and bias, in this order:
-- bias[1] => 128      bias of layer (1)
-- bias[2] =>  64      bias of layer (2)
-- bias[3] => 128      bias of layer (3)
-- bias[4] => 256      bias of layer (4)
-- weights[1] => 256*128  weights of layer (1)
-- weights[2] => 128*64   weights of layer (2)
function ann.autoencoders.stacked_denoising_pretraining(params)
  local check_mandatory_param = function(params, name)
    if not params[name] then error ("Parameter " .. name .. " is mandatory") end
  end
  local valid_params = table.invert{ "shuffle_random", "distribution",
				     "perturbation_random", "replacement",
				     "var", "layers", "bunch_size",
				     "learning_rate",
				     "max_epochs", "max_epochs_wo_improvement",
				     "momentum", "weight_decay", "val_input_dataset",
				     "weights_random"}
  for name,v in pairs(valid_params) do
    if not valid_params[name] then
      error("Incorrect param name '"..name.."'")
    end
  end
  -- Error checking in params table --
  if params.input_dataset and params.distribution then
    error("The input_dataset and distribution parameters are forbidden together")
  end
  if params.distribution and not params.replacement then
    error("The replacement parameter is mandatary if distribution")
  end
  for _,name in ipairs({ "shuffle_random", "perturbation_random",
			 "var", "layers", "bunch_size", "learning_rate",
			 "max_epochs", "max_epochs_wo_improvement",
			 "momentum", "weight_decay", "val_input_dataset",
			 "weights_random"}) do
    check_mandatory_param(params, name)
  end
  --------------------------------------

  -- copy dataset params to auxiliar table
  local current_dataset_params = {
    input_dataset = params.input_dataset,
    distribution  = params.distribution
  }
  local current_val_dataset_params = {
    input_dataset  = params.val_input_dataset,
    output_dataset = params.val_input_dataset
  }
  -- output weights and bias matrices
  local weights = {}
  local bias    = {}
  -- loop for each pair of layers
  for i=2,#params.layers do
    local input_size = params.layers[i-1].size
    local cod_size   = params.layers[i].size
    printf("# Training of layer %d--%d--%d (number %d)\n",
	  input_size, cod_size, input_size, i-1)
    local input_actf = ann.activations.from_string(params.layers[i-1].actf)
    local cod_actf   = ann.activations.from_string(params.layers[i].actf)
    local val_data = current_val_dataset_params
    local data
    data = ann.autoencoders.generate_training_table_configuration_from_params(current_dataset_params,
									      params)
    local dae
    dae = ann.autoencoders.build_autoencoder_from_sizes_and_actf(params.bunch_size,
								 input_size,
								 input_actf,
								 cod_size,
								 cod_actf,
								 params.weights_random)
    dae:set_option("learning_rate", params.learning_rate)
    dae:set_option("momentum", params.momentum)
    dae:set_option("weight_decay", params.weight_decay)
    collectgarbage("collect")
    dae:set_error_function(ann.error_functions.full_cross_entropy())
    local best_val_error = 111111111
    local best_net       = dae:clone()
    local best_epoch     = 0
    for epoch=1,params.max_epochs do
      local train_error = dae:train_dataset(data)
      local val_error   = dae:validate_dataset(val_data)
      local _,m = dae:get_layer_connections(2):weights()
      if val_error < best_val_error then
	best_val_error = val_error
	best_epoch     = epoch
	best_net       = dae:clone()
      end
      printf("%4d %10.6f %10.6f  (best %10.6f at epoch %4d)\n",
	     epoch, train_error, val_error, best_val_error, best_epoch)
      collectgarbage("collect")
      -- convergence criteria
      if epoch - best_epoch > params.max_epochs_wo_improvement then break end
    end
    local _,b1mat = best_net:get_layer_connections(1):weights()
    local _,b2mat = best_net:get_layer_connections(3):weights()
    local _,wmat  = best_net:get_layer_connections(2):weights()
    table.insert(weights, wmat)
    table.insert(bias, { b1mat, b2mat })
    if i ~= #params.layers then
      -- generation of new input patterns using only the first part of
      -- autoencoder except at last loop iteration
      local codifier
      codifier = ann.autoencoders.build_two_layered_codifier_from_weights(params.bunch_size,
									  input_size,
									  input_actf,
									  cod_size,
									  cod_actf,
									  b1mat, wmat)
      -- auxiliar function
      local generate_codification = function(codifier, ds)
	local output_mat = matrix(ds:numPatterns(), cod_size)
	local output_ds  = dataset.matrix(output_mat)
	codifier:use_dataset{ input_dataset = ds, output_dataset = output_ds }
	return output_ds
      end
      if current_dataset_params.distribution then
	-- compute code for each distribution dataset
	for _,v in ipairs(current_dataset_params.distribution) do
	  v.input_dataset = generate_codification(codifier, v.input_dataset)
	end
      else
	-- compute code for input dataset
	local ds = generate_codification(codifier,
					 current_dataset_params.input_dataset)
	current_dataset_params.input_dataset = ds
      end
      -- compute code for validation input dataset
      local ds = generate_codification(codifier,
				       current_val_dataset_params.input_dataset)
      current_val_dataset_params.input_dataset  = ds
      current_val_dataset_params.output_dataset = ds
    end -- if i ~= params.layers
  end -- for i=2,#params.layers
  return {weights=weights, bias=bias}
end



-- Receive an autoencoder table with bias and weights, pretrained with previous
-- function
function ann.autoencoders.stacked_denoising_finetunning(sdae_table, params)
  local check_mandatory_param = function(params, name)
    if not params[name] then error ("Parameter " .. name .. " is mandatory") end
  end
  local valid_params = table.invert{ "shuffle_random", "distribution",
				     "perturbation_random", "replacement",
				     "var", "layers", "bunch_size",
				     "learning_rate",
				     "max_epochs", "max_epochs_wo_improvement",
				     "momentum", "weight_decay", "val_input_dataset",
				     "weights_random"}
  for name,v in pairs(valid_params) do
    if not valid_params[name] then
      error("Incorrect param name '"..name.."'")
    end
  end
  -- Error checking in params table --
  if params.input_dataset and params.distribution then
    error("The input_dataset and distribution parameters are forbidden together")
  end
  if params.distribution and not params.replacement then
    error("The replacement parameter is mandatary if distribution")
  end
  for _,name in ipairs({ "shuffle_random", "perturbation_random",
			 "var", "layers", "bunch_size", "learning_rate",
			 "max_epochs", "max_epochs_wo_improvement",
			 "momentum", "weight_decay", "val_input_dataset",
			 "weights_random"}) do
    check_mandatory_param(params, name)
  end
  --------------------------------------
  local weights = sdae_table.weights
  local bias    = sdae_table.bias
  -- FINETUNING
  print("# Begining of fine-tuning")
  local sdae = ann.autoencoders.build_full_autoencoder(params.bunch_size,
						       params.layers,
						       weights, bias)
  sdae:set_option("learning_rate", params.learning_rate)
  sdae:set_option("momentum", params.momentum)
  sdae:set_option("weight_decay", params.weight_decay)
  collectgarbage("collect")
  local data
  data = ann.autoencoders.generate_training_table_configuration_from_params(params,
									    params)
  local val_data = { input_dataset  = params.val_input_dataset,
		     output_dataset = params.val_input_dataset }
  local best_val_error = 111111111
  local best_net       = sdae:clone()
  local best_epoch     = 0
  for epoch=1,params.max_epochs do
    local train_error = sdae:train_dataset(data)
    local val_error   = sdae:validate_dataset(val_data)
    if val_error < best_val_error then
      best_val_error = val_error
      best_epoch     = epoch
      best_net       = sdae:clone()
    end
    printf("%4d %10.6f %10.6f  (best %10.6f at epoch %4d)\n",
	   epoch, train_error, val_error, best_val_error, best_epoch)
    collectgarbage("collect")
    -- convergence criteria
    if epoch - best_epoch > params.max_epochs_wo_improvement then break end
  end
  return best_net
end

function ann.autoencoders.build_codifier_from_sdae(sdae,
						   bunch_size,
						   layers)
  local sdae_connections = sdae:get_layer_connections_vector()
  local sdae_activations = sdae:get_layer_activations_vector()
  local codifier_net = ann.mlp{ bunch_size = bunch_size }
  local codifier_connections = {}
  local codifier_activations = {}
  for i=1,(#layers-1)*2 do
    table.insert(codifier_connections, sdae_connections[i]:clone(codifier_net))
  end
  local type = "inputs"
  for i=1,#layers-1 do
    table.insert(codifier_activations, sdae_activations[i]:clone(codifier_net,
								 type))
    type = "hidden"
  end
  table.insert(codifier_activations, sdae_activations[#layers]:clone(codifier_net,
								     "outputs"))
  local k=1
  for i=2,#layers do
    local actf    = ann.activations.from_string(layers[i].actf)
    local input   = codifier_activations[i-1]
    local output  = codifier_activations[i]
    local bias    = codifier_connections[k]
    local weights = codifier_connections[k+1]
    ann.actions.forward_bias{ ann         = codifier_net,
			      output      = output,
			      connections = bias }
    ann.actions.dot_product{ ann         = codifier_net,
			     input       = input,
			     output      = output,
			     connections = weights,
			     transpose   = false }
    if actf then
      ann.actions.activations{ ann     = codifier_net,
			       output  = output,
			       actfunc = actf }
    end
    k = k + 2
  end
  return codifier_net
end
  
function ann.autoencoders.compute_encoded_dataset_using_codifier(codifier_net,
								 input_dataset)
  local output_dataset = dataset.matrix(matrix(input_dataset:numPatterns(),
					       codifier_net:get_output_size()))
  codifier_net:use_dataset{ input_dataset  = input_dataset,
			    output_dataset = output_dataset }
  return output_dataset
end
