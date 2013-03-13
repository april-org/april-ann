ann.autoencoders = ann.autoencoders or {}

-- AUXILIAR LOCAL FUNCTIONS --

-- This function builds a codifier from the weights of the first layer of a
-- restricted autoencoder
local function build_two_layered_codifier_from_weights(bunch_size,
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
  cod_bias:load{ w=bias_mat }
  local cod_weights = ann.connections.all_all{ ann         = codifier,
					       input_size  = input_size,
					       output_size = cod_size }
  cod_weights:load{ w=weights_mat }
  -- 
  codifier:push_back_all_all_layer{
    input   = input_layer,
    output  = cod_layer,
    bias    = cod_bias,
    weights = cod_weights,
    actfunc = cod_actf }
  return codifier
end

-- This function builds an autoencoder of two layers (input-hidden-output) where
-- the hidden-output uses the same weights as input-hidden, so it is
-- simetric.
local function build_two_layered_autoencoder_from_sizes_and_actf(bunch_size,
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
  -- first layer
  local hidden_bias,hidden_weights
  hidden_bias,hidden_weights = autoencoder:push_back_all_all_layer{
    input   = input_layer,
    output  = hidden_layer,
    actfunc = cod_actf }
  -- second layer (weights transposed)
  autoencoder:push_back_all_all_layer{
    input     = hidden_layer,
    output    = output_layer,
    weights   = hidden_weights,
    transpose = true,
    actfunc   = input_actf }
  -- randomize weights
  autoencoder:randomize_weights{ random=weights_random,
				 inf=-math.sqrt(6 / (input_size + cod_size)),
				 sup= math.sqrt(6 / (input_size + cod_size)),
				 use_fanin = false}
  return autoencoder
end

-- Generate the data table for training a two-layered auto-encoder
local function generate_training_table_configuration(current_dataset_params,
						     replacement,
						     shuffle_random,
						     noise_pipeline,
						     output_datasets,
						     zero)
  local data = {}
  if current_dataset_params.input_dataset then
    if output_datasets and #output_datasets > 1 then
      error("Without distribution only one dataset is allowed "..
	    "at output_datasets table")
    end
    data.input_dataset  = current_dataset_params.input_dataset
    data.output_dataset = ((output_datasets or {})[1] or
			   current_dataset_params.input_dataset)
    -- The input is perturbed with noise
    for _,noise_builder in ipairs(noise_pipeline) do
      data.input_dataset = noise_builder(data.input_dataset)
    end
  end -- if params.input_dataset
  if current_dataset_params.distribution then
    data.distribution = {}
    for idx,v in ipairs(current_dataset_params.distribution) do
      local ds = v.input_dataset
      -- The input is perturbed with noise
      for _,noise_builder in ipairs(noise_pipeline) do
	data.input_dataset = noise_builder(data.input_dataset)
      end
      if output_datasets and not output_datasets[idx] then
	error("Incorrect number of output_datasets")
      end
      table.insert(data.distribution, {
		     input_dataset  = ds,
		     output_dataset = (output_datasets or {})[idx] or ds,
		     probability    = v.prob })
    end -- for _,v in ipairs(distribution)
  end -- if distribution
  data.shuffle     = shuffle_random
  data.replacement = replacement
  return data
end

-- PUBLIC FUNCTIONS --

-- This functions receives layer sizes and sdae_table with weights and bias
-- arrays. It returns a fully connected stacked denoising autoencoder ANN.
function ann.autoencoders.build_full_autoencoder(bunch_size,
						 layers,
						 sdae_table)
  local weights_mat = sdae_table.weights
  local bias_mat    = sdae_table.bias
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
		   size = layers[i].size,
		   w    = bias_mat[i-1][1] })
    table.insert(weights_sdae, ann.connections.all_all{
		   ann = sdae,
		   input_size  = layers[i-1].size,
		   output_size = layers[i].size,
		   w           = weights_mat[i-1] })
    sdae:push_back_all_all_layer{ input   = neuron_layers[#neuron_layers-1],
				  output  = neuron_layers[#neuron_layers],
				  weights = weights_sdae[#weights_sdae],
				  bias    = bias_sdae[#bias_sdae],
				  actfunc = actfs[#actfs] }
  end
  for i=#layers-1,1,-1 do
    table.insert(neuron_layers, ann.units.real_cod{
		   ann  = sdae,
		   size = layers[i].size,
		   type = (i>1 and "hidden") or "outputs" })
    table.insert(actfs, ann.activations.from_string(layers[i].actf))
    table.insert(bias_sdae, ann.connections.bias{
		   ann  = sdae,
		   size = layers[i].size,
		   w    = bias_mat[i][2] })
    sdae:push_back_all_all_layer{ input     = neuron_layers[#neuron_layers-1],
				  output    = neuron_layers[#neuron_layers],
				  weights   = weights_sdae[i],
				  bias      = bias_sdae[#bias_sdae],
				  actfunc   = actfs[#actfs],
				  transpose = true }
  end
  return sdae
end

-- Params is a table which could contain:
--   * input_dataset => dataset with input (and output) for AE
--   * distribution => a table which contains a list of {input_dataset=...., prob=....}
--   * replacement => replacement value for training
--   * shuffle_random => random number generator
--   * weights_random => random number generator
--   * perturbation_random => random number generator
--   * var => variance of gaussian noise
--   * layers => table which contains a list of { size=...., actf=....}, being
--               size a number and actf a string = "logistic"|"tanh"|"linear"
--   * supervised_layer => size and actf
--   * output_datasets => a table with output datasets
--   * bunch_size => size of mini-batch
--   * learning_rate
--   * momentum
--   * weight_decay
--   * max_epochs
--   * min_epochs
--   * stopping_criterion => function
--   * pretraining_percentage_stopping_criterion
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
function ann.autoencoders.stacked_denoising_pretraining(t)
  error("Deprecated, use ann.autoencoders.greedy_layerwise_pretraining")
end

function ann.autoencoders.greedy_layerwise_pretraining(params)
  local check_mandatory_param = function(params, name)
    if not params[name] then error ("Parameter " .. name .. " is mandatory") end
  end
  local valid_params = table.invert{ "shuffle_random", "distribution",
				     "input_dataset", "output_datasets",
				     "supervised_layer",
				     "replacement",
				     "layers", "bunch_size",
				     "training_options",
				     "weights_random" }
  for name,v in pairs(params) do
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
  for _,name in ipairs({ "shuffle_random",
			 "layers", "bunch_size", "training_options",
			 "weights_random" }) do
    check_mandatory_param(params, name)
  end
  if (params.supervised_layer~=nil) == (not params.output_datasets) then
    error("Params output_datasets and "..
	  "supervised_layer must be present together")
  end
  if params.output_datasets and not type(params.output_datasets) == "table" then
    error("Param output_datasets must be an array of datasets")
  end
  params.training_options.global    = params.training_options.global    or { ann_options }
  params.training_options.layerwise = params.training_options.layerwise or {}
  --------------------------------------

  -- copy dataset params to auxiliar table
  local current_dataset_params = {
    input_dataset = params.input_dataset,
    distribution  = params.distribution
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
    io.stdout:flush()
    local input_actf_str = params.layers[i-1].actf
    local input_actf = ann.activations.from_string(input_actf_str)
    local cod_actf   = ann.activations.from_string(params.layers[i].actf)
    local global_options    = table.deep_copy(params.training_options.global)
    local layerwise_options = params.training_options.layerwise[i-1] or {}
    layerwise_options.ann_options = layerwise_options.ann_options or {}
    local lookup = function(name) return layerwise_options[name] or global_options[name] end
    local data
    data = generate_training_table_configuration(current_dataset_params,
						 params.replacement,
						 params.shuffle_random,
						 lookup("noise_pipeline") or {},
						 nil,
						 (((input_actf_str == "tanh" or input_actf_str == "softsign") and -1.0) or 0.0))
    local dae
    dae = build_two_layered_autoencoder_from_sizes_and_actf(params.bunch_size,
							    input_size,
							    input_actf,
							    cod_size,
							    cod_actf,
							    params.weights_random)
    for key,value in pairs(global_options.ann_options) do
      if layerwise_options.ann_options[key] == nil then
	dae:set_option(key, value)
      end
    end
    for key,value in pairs(layerwise_options.ann_options) do
      dae:set_option(key, value)
    end
    collectgarbage("collect")
    if (params.layers[i-1].actf == "logistic" or
	params.layers[i-1].actf == "softmax") then
      dae:set_error_function(ann.error_functions.full_logistic_cross_entropy())
    else
      dae:set_error_function(ann.error_functions.mse())
    end
    if params.layers[i-1].actf=="linear" or params.layers[i].actf=="linear" then
      -- if activation is linear, the derivative slope is so high, so we use
      -- learning_rate to reduce its impact
      local ratio = 1/math.sqrt(cod_size+input_size)
      dae:set_option("learning_rate", dae:get_option("learning_rate")*ratio)
    end
    ---------- TRAIN THE AUTOENCODER WO VALIDATION ----------
    local best_net = ann.train_wo_validation{
      ann            = dae,
      min_epochs     = lookup("min_epochs"),
      max_epochs     = lookup("max_epochs"),
      training_table = data,
      percentage_stopping_criterion = lookup("pretraining_percentage_stopping_criterion"),
      update_function = function(t)
	printf("%4d %10.6f  (improvement %.4f)\n",
	       t.current_epoch, t.train_error, t.train_improvement)
	io.stdout:flush()	
      end }
    ---------------------------------------------------------
    local b1mat = best_net:get_layer_connections(1):weights()
    local b2mat = best_net:get_layer_connections(3):weights()
    local wmat  = best_net:get_layer_connections(2):weights()
    table.insert(weights, wmat)
    table.insert(bias, { b1mat, b2mat })
    if i ~= #params.layers or params.supervised_layer then
      -- generation of new input patterns using only the first part of
      -- autoencoder except at last loop iteration
      local codifier
      codifier = build_two_layered_codifier_from_weights(params.bunch_size,
							 input_size,
							 input_actf,
							 cod_size,
							 cod_actf,
							 b1mat, wmat)
      if current_dataset_params.distribution then
	-- compute code for each distribution dataset
	for _,v in ipairs(current_dataset_params.distribution) do
	  v.input_dataset = ann.autoencoders.encode_dataset(codifier,
							    v.input_dataset)
	end
      else
	-- compute code for input dataset
	local ds =
	  ann.autoencoders.encode_dataset(codifier,
					  current_dataset_params.input_dataset)
	  current_dataset_params.input_dataset = ds
      end
    end -- if i ~= params.layers
  end -- for i=2,#params.layers
  local sdae_table = {weights=weights, bias=bias}
  -- Train a supervised layer
  local full_ann = nil
  if params.supervised_layer then
    printf("# Training of supervised layer %d--%d (number %d)\n",
	   params.layers[#layers].size, params.supervised_layer.size,
	   #params.layers+1)
    local input_size     = params.layers[#layers].size
    local input_actf_str = params.layers[#layers].actf
    local global_options    = table.deep_copy(params.training_options.global)
    local layerwise_options = (params.training_options.layerwise[#layers] or
			       { ann_options = {} })
    layerwise_options.ann_options = layerwise_options.ann_options or {}
    local lookup = function(name) return layerwise_options[name] or global_options[name] end
    local data
    data = generate_training_table_configuration(current_dataset_params,
						 params.replacement,
						 params.shuffle_random,
						 lookup("noise_pipeline") or {},
						 params.output_datasets,
						 (((input_actf_str == "tanh" or input_actf_str == "softsign") and -1.0) or 0.0))
    local thenet = ann.mlp.all_all.generate{
      topology = string.format("%d inputs %d %s",
			       input_size,
			       params.supervised_layer.size,
			       params.supervised_layer.actf),
      bunch_size = params.bunch_size,
      random     = params.weights_random,
      inf=-math.sqrt(6 / (input_size + params.supervised_layer.size)),
      sup= math.sqrt(6 / (input_size + params.supervised_layer.size)),
      use_fanin = false }
    for key,value in pairs(global_options.ann_options) do
      if layerwise_options.ann_options[key] == nil then
	thenet:set_option(key, value)
      end
    end
    for key,value in pairs(layerwise_options.ann_options) do
      thenet:set_option(key, value)
    end
    if (params.supervised_layer.actf == "softmax" or
	params.supervised_layer.actf == "logistic") then
      thenet:set_error_function(ann.error_functions.logistic_cross_entropy())
    else
      thenet:set_error_function(ann.error_functions.mse())
    end
    local best_net = ann.train_wo_validation{
      ann            = thenet,
      min_epochs     = lookup("min_epochs"),
      max_epochs     = lookup("max_epochs"),
      training_table = data,
      percentage_stopping_criterion = lookup("pretraining_percentage_stopping_criterion"),
      update_function = function(t)
	printf("%4d %10.6f  (improvement %.4f)\n",
	       t.current_epoch, t.train_error, t.train_improvement)
	io.stdout:flush()	
      end }
    local codifier = ann.autoencoders.build_codifier_from_sdae_table(sdae_table,
								     params.bunch_size,
								     params.layers)
    full_ann = ann.mlp.add_layers{
      ann           = codifier,
      new_layers    = { { params.supervised_layer.size,
			  params.supervised_layer.actf } },
      bunch_size    = params.bunch_size,
      bias_table    = { best_net:get_layer_connections(1):weights() },
      weights_table = { best_net:get_layer_connections(2):weights() } }
  end
  return sdae_table,full_ann
end

-- This function returns a MLP formed by the codification part of a full stacked
-- auto encoder
function ann.autoencoders.build_codifier_from_sdae_table(sdae_table,
							 bunch_size,
							 layers)
  local weights_mat   = sdae_table.weights
  local bias_mat      = sdae_table.bias
  local codifier_net  = ann.mlp{ bunch_size = bunch_size }
  local neuron_layers = {}
  local actfs         = {}
  local weights_codifier_net  = {}
  local bias_codifier_net     = {}
  table.insert(neuron_layers, ann.units.real_cod{
		 ann  = codifier_net,
		 size = layers[1].size,
		 type = "inputs" })
  for i=2,#layers do
    table.insert(neuron_layers, ann.units.real_cod{
		   ann  = codifier_net,
		   size = layers[i].size,
		   type = ((i < #layers and "hidden") or "outputs") })
    table.insert(actfs, ann.activations.from_string(layers[i].actf))
    table.insert(bias_codifier_net, ann.connections.bias{
		   ann  = codifier_net,
		   size = layers[i].size,
		   w    = bias_mat[i-1][1] })
    table.insert(weights_codifier_net, ann.connections.all_all{
		   ann = codifier_net,
		   input_size  = layers[i-1].size,
		   output_size = layers[i].size,
		   w           = weights_mat[i-1] })
    codifier_net:push_back_all_all_layer{
      input   = neuron_layers[#neuron_layers-1],
      output  = neuron_layers[#neuron_layers],
      bias    = bias_codifier_net[#bias_codifier_net],
      weights = weights_codifier_net[#weights_codifier_net],
      actfunc = actfs[#actfs] }
  end
  return codifier_net
end

-- This function returns a MLP formed by the codification part of a full stacked
-- auto encoder
function ann.autoencoders.build_codifier_from_sdae(sdae, bunch_size, layers)
  local sdae_connections = sdae:get_layer_connections_vector()
  local sdae_activations = sdae:get_layer_activations_vector()
  local codifier_net = ann.mlp{ bunch_size = bunch_size }
  local codifier_connections = {}
  local codifier_activations = {}
  for i=1,(#layers-1)*2 do
    table.insert(codifier_connections, sdae_connections[i]:clone(codifier_net))
  end
  local layer_type = "inputs"
  for i=1,#layers-1 do
    table.insert(codifier_activations, sdae_activations[i]:clone(codifier_net,
								 layer_type))
    layer_type = "hidden"
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
    codifier_net:push_back_all_all_layer{
      input   = input,
      output  = output,
      bias    = bias,
      weights = weights,
      actfunc = actf }
    k = k + 2
  end
  return codifier_net
end

-- Returns a dataset with the codification of input dataset patterns  
function ann.autoencoders.compute_encoded_dataset_using_codifier()
  error("Deprecated, use ann.autoencoders.encode_dataset")
end
function ann.autoencoders.encode_dataset(codifier_net,
					 input_dataset)
  local output_dataset = dataset.matrix(matrix(input_dataset:numPatterns(),
					       codifier_net:get_output_size()))
  codifier_net:use_dataset{ input_dataset  = input_dataset,
			    output_dataset = output_dataset }
  return output_dataset
end

function ann.autoencoders.stacked_denoising_pretraining(t)
  error("Deprecated, use ann.autoencoders.greedy_layerwise_pretraining")
end
