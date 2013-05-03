ann.autoencoders = ann.autoencoders or {}

-- AUXILIAR LOCAL FUNCTIONS --

-- This function builds a codifier from the weights of the first layer of a
-- restricted autoencoder
local function build_two_layered_codifier_from_weights(input_size,
						       input_actf,
						       cod_size,
						       cod_actf,
						       bias_mat,
						       weights_mat)
  local codifier_component = ann.components.stack()
  codifier_component:push( ann.components.hyperplane{
			     dot_product_weights = "weights",
			     bias_weights        = "bias",
			     input  = input_size,
			     output = cod_size, } )
  codifier_component:push(ann.components[cod_actf]())
  local weights_table = codifier_component:build()
  weights_table["weights"]:load{ w = weights_mat }
  weights_table["bias"]:load{ w = bias_mat }
  return codifier_component
end

-- This function builds an autoencoder of two layers (input-hidden-output) where
-- the hidden-output uses the same weights as input-hidden, so it is
-- simetric.
local function build_two_layered_autoencoder_from_sizes_and_actf(input_size,
								 input_actf,
								 cod_size,
								 cod_actf,
								 weights_random)
  local autoencoder_component = ann.components.stack()
  autoencoder_component:push( ann.components.hyperplane{
				name                = "layer1",
				dot_product_weights = "weights",
				bias_weights        = "bias1",
				input  = input_size,
				output = cod_size, } )
  autoencoder_component:push(ann.components[cod_actf]{ name="actf1" })
  autoencoder_component:push( ann.components.hyperplane{
				name                = "layer2",
				dot_product_weights = "weights",
				bias_weights        = "bias2",
				input  = cod_size,
				output = input_size,
				transpose = true} )
  autoencoder_component:push(ann.components[input_actf]{ name="actf2" })
  local weights_table = autoencoder_component:build()
  for _,wname in ipairs({ "weights", "bias1", "bias2" }) do
    weights_table[wname]:randomize_weights{
      random = weights_random,
      inf    = -math.sqrt(6 / (input_size + cod_size)),
      sup    =  math.sqrt(6 / (input_size + cod_size)) }
  end
  return autoencoder_component
end

--auxiliar function to generate a replacement
function get_replacement_dataset(randObject, replacementSize, ...)
  local resul = {}
  if #arg > 0 then
    local mat = matrix(replacementSize)
    local numPat = arg[1]:numPatterns()
    for i=1,replacementSize do
      mat:setElement(i,randObject:randInt(1,numPat))
    end
    local ds = dataset.matrix(mat)
    for i,v in ipairs(arg) do
      if v ~= nil then
	if v:numPatterns() ~= numPat then
	  error("Datasets have differnet number of patterns")
	end
	table.insert(resul,dataset.indexed(ds,{v}))
      end
    end
  end
  return resul
end


-- Generate the data table for training a two-layered auto-encoder
local function generate_training_table_configuration(current_dataset_params,
						     replacement,
						     shuffle_random,
						     noise_pipeline,
						     output_datasets)
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



local function
    generate_training_table_configuration_on_the_fly(current_dataset_params,
						     replacement,
						     shuffle_random,
						     noise_pipeline,
						     mlp_final_trainer,
						     output_datasets)
  return function()
    -- Take the original datasets (input, and output)
    local input_dataset  = current_dataset_params.input_dataset
    local output_dataset = (output_datasets or {})[1]
    -- Generate a replacement of each dataset
    local input_repl_ds, output_repl_ds
    -- if autoencoder, only generate one corpus replacement
    input_repl_ds, output_repl_ds = unpack( get_replacement_dataset(shuffle_random, replacement, input_dataset, output_dataset) )
    -- generate the last layer dataset
    local input_layer_dataset = mlp_final_trainer:use_dataset{
      input_dataset = input_repl_ds
    }
    -- The output is the same than the imput
    output_dataset = (output_repl_ds or input_layer_dataset)
    -- Add the noise
    for _,noise_builder in ipairs(noise_pipeline) do
      input_dataset = noise_builder(input_layer_dataset)
    end
    return {
      input_dataset  = input_dataset,
      output_dataset = output_dataset
    }
  end
end

-- PUBLIC FUNCTIONS --

-- This functions receives layer sizes and sdae_table with weights and bias
-- arrays. It returns a fully connected stacked denoising autoencoder ANN.
function ann.autoencoders.build_full_autoencoder(layers,
						 sdae_table)
  local weights_mat   = sdae_table.weights
  local bias_mat      = sdae_table.bias
  local sdae          = ann.components.stack()
  local prev_size     = layers[1].size
  local weights_table = {}
  local k = 1
  for i=2,#layers do
    local size , actf   = layers[i].size,layers[i].actf
    local wname , bname = "weights" .. (i-1) , "bias" .. k
    sdae:push( ann.components.hyperplane{
		 input               = prev_size,
		 output              = size,
		 dot_product_weights = wname,
		 bias_weights        = bname })
    sdae:push( ann.components[actf]() )
    --
    weights_table[wname] = ann.connections{ input=prev_size,
					    output=size,
					    w=weights_mat[i-1] }
    weights_table[bname] = ann.connections{ input=1,
					    output=size,
					    w=weights_mat[i-1] }
    prev_size = size
    k = k+1
  end
  for i=#layers-1,1,-1 do
    local size , actf   = layers[i].size,layers[i].actf
    local wname , bname = "weights" .. i , "bias" .. k
    sdae:push( ann.components.hyperplane{
		 input               = prev_size,
		 output              = size,
		 transpose           = true,
		 dot_product_weights = wname,
		 bias_weights        = bname })
    sdae:push( ann.components[actf]() )
    --
    weights_table[bname] = ann.connections{ input=1,
					    output=size,
					    w=weights_mat[i-1] }
    prev_size = size
    k = k+1
  end
  dae:build{ weights=weights_table }
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
--   * neuron_squared_length_upper_bound
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
function ann.autoencoders.greedy_layerwise_pretraining(t)
  local params = get_table_fields(
    {
      shuffle_random   = { mandatory=true,  isa_match=random },
      weights_random   = { mandatory=true,  isa_match=random },
      layers           = { mandatory=true, type_match="table",
			   getter=get_table_fields_ipairs{
			     actf = { mandatory=true, type_match="string" },
			     size = { mandatory=true, type_match="number" },
			   }, },
      bunch_size       = { mandatory=true, type_match="number", },
      training_options = { mandatory=true, type_match="table" },
      --
      input_dataset    = { mandatory=false },
      output_datasets  = { mandatory=false, type_match="table", default=nil },
      distribution     = { mandatory=false, type_match="table", default=nil,
			   getter=get_table_fields_ipairs{
			     input_dataset   = { mandatory=true },
			     output_datasets = { mandatory=true },
			     probability     = { mandatory=true },
			   }, },
      replacement      = { mandatory=false, type_match="number", default=nil },
      supervised_layer = { mandatory=false, type_match="table", default=nil,
			   getter=get_table_fields_recursive{
			     actf = { mandatory=true, type_match="string" },
			     size = { mandatory=true, type_match="number" },
			   }, },
    }, t)
  -- Error checking in params table --
  if params.input_dataset and params.distribution then
    error("The input_dataset and distribution parameters are forbidden together")
  end
  if params.distribution and not params.replacement then
    error("The replacement parameter is mandatary if distribution")
  end
  if (params.supervised_layer~=nil) == (not params.output_datasets) then
    error("Params output_datasets and "..
	    "supervised_layer must be present together")
  end
  params.training_options.global    = params.training_options.global    or { ann_options }
  params.training_options.layerwise = params.training_options.layerwise or {}
  --------------------------------------
  -- on the fly. Do not generate all the dataset for each layer
  local on_the_fly = params.replacement
  if on_the_fly and params.distribution then
    error("On the fly mode is not working with dataset distribution")
  end
  -- copy dataset params to auxiliar table
  local current_dataset_params = {
    input_dataset = params.input_dataset,
    distribution  = params.distribution
  }
  -- output weights and bias matrices
  local weights = {}
  local bias    = {}
  -- incremental mlp
  local mlp_final_weights = {}
  local mlp_final = ann.components.stack()
  -- loop for each pair of layers
  for i=2,#params.layers do
    local input_size = params.layers[i-1].size
    local cod_size   = params.layers[i].size
    printf("# Training of layer %d--%d--%d (number %d)\n",
	   input_size, cod_size, input_size, i-1)
    io.stdout:flush()
    local input_actf = params.layers[i-1].actf
    if input_actf == "logistic" then input_actf = "log_logistic"
    elseif input_actf == "softmax" then input_actf = "log_softmax"
    end
    local cod_actf   = params.layers[i].actf
    local global_options    = table.deep_copy(params.training_options.global)
    local layerwise_options = params.training_options.layerwise[i-1] or {}
    layerwise_options.ann_options = layerwise_options.ann_options or {}
    local lookup = function(name) return layerwise_options[name] or global_options[name] end
    local data
    if not on_the_fly or i == 2 then
      data = generate_training_table_configuration(current_dataset_params,
						   params.replacement,
						   params.shuffle_random,
						   lookup("noise_pipeline") or {},
						   nil)
    else
      local mlp_final_trainer = trainer(mlp_final:clone(), nil, params.bunch_size)
      data = generate_training_table_configuration_on_the_fly(current_dataset_params,
							      params.replacement,
							      params.shuffle_random,
							      lookup("noise_pipeline") or {},
							      mlp_final_trainer,
							      nil)
    end
    local dae
    dae = build_two_layered_autoencoder_from_sizes_and_actf(input_size,
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
    local loss_function
    if input_size > 1 and (input_actf == "log_logistic" or
			 input_actf == "log_softmax") then
      loss_function = ann.loss.multi_class_cross_entropy(input_size)
    elseif input_size == 1 and input_actf == "log_logistic" then
      loss_function = ann.loss.cross_entropy(input_size)
    else
      loss_function = ann.loss.mse(input_size)
    end
    if input_actf=="linear" or cod_actf=="linear" then
      -- if activation is linear, the derivative slope is so high, so we use
      -- learning_rate to reduce its impact
      local ratio = 1/math.sqrt(cod_size+input_size)
      dae:set_option("learning_rate", dae:get_option("learning_rate")*ratio)
    end
    ---------- TRAIN THE AUTOENCODER WO VALIDATION ----------
    local trainer = trainable.supervised_trainer(dae, loss_function,
						 params.bunch_size)
    local best_net = trainer:train_wo_validation{
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
    local b1obj = best_net:weights("bias1"):clone()
    local b2obj = best_net:weights("bias2"):clone()
    local wobj  = best_net:weights("weights"):clone()
    local b1mat = b1obj:weights()
    local b2mat = b2obj:weights()
    local wmat  = wobj:weights()
    table.insert(weights, wmat)
    table.insert(bias, { b1mat, b2mat })
    --
    mlp_final:push( ann.components.hyperplane{
		      input  = input_size,
		      output = cod_size,
		      name = "layer" .. (i-1),
		      dot_product_name = "weights" .. (i-1),
		      bias_name = "bias" .. (i-1),
		      dot_product_weights = "weights" .. (i-1),
		      bias_weights = "bias" .. (i-1), })
    mlp_final:push( ann.components[cod_actf]{ name="actf" .. (i-1) } )
    mlp_final_weights["weights" .. (i-1)] = wobj
    mlp_final_weights["bias"    .. (i-1)] = b1obj
    --
    --insert the information
    if not on_the_fly then
      if i ~= #params.layers or params.supervised_layer then
	-- generation of new input patterns using only the first part of
	-- autoencoder except at last loop iteration (only if not
	-- supervised_layer)
	local codifier
	codifier = build_two_layered_codifier_from_weights(input_size,
							   input_actf,
							   cod_size,
							   cod_actf,
							   b1mat, wmat)
	local cod_trainer = trainable.supervised_trainer(codifier,
							 nil,
							 params.bunch_size)
	if current_dataset_params.distribution then
	  -- compute code for each distribution dataset
	  for _,v in ipairs(current_dataset_params.distribution) do
	    v.input_dataset = cod_trainer:use_dataset{
	      input_dataset = v.input_dataset
	    }
	  end
	else
	  -- compute code for input dataset
	  local ds = cod_trainer:use_dataset{
	    input_dataset = current_dataset_params.input_dataset
	  }
	  current_dataset_params.input_dataset = ds
	end -- if distribution ... else
      end -- if i ~= params.layers
    end -- for i=2,#params.layers
  end -- if not on_the_fly
  local sdae_table = {weights=weights, bias=bias}
  -- Train a supervised layer
  if params.supervised_layer then
    printf("# Training of supervised layer %d--%d (number %d)\n",
	   params.layers[#params.layers].size, params.supervised_layer.size,
	   #params.layers+1)
    local input_size     = params.layers[#params.layers].size
    local input_actf     = params.layers[#params.layers].actf
    local global_options    = table.deep_copy(params.training_options.global)
    local layerwise_options = (params.training_options.layerwise[#params.layers] or
				 { ann_options = {} })
    layerwise_options.ann_options = layerwise_options.ann_options or {}
    local lookup = function(name) return layerwise_options[name] or global_options[name] end
    local data

    if not on_the_fly then
      data = generate_training_table_configuration(current_dataset_params,
						   params.replacement,
						   params.shuffle_random,
						   lookup("noise_pipeline") or {},
						   params.output_datasets)

    else
      local mlp_final_trainer = trainer(mlp_final:clone(), nil, params.bunch_size)
      data = generate_training_table_configuration_on_the_fly(current_dataset_params,
							      params.replacement,
							      params.shuffle_random,
							      lookup("noise_pipeline") or {},
							      mlp_final_trainer,
							      params.output_datasets)

    end
    local thenet = ann.mlp.all_all.generate(
      string.format("%d inputs %d %s",
		    input_size,
		    params.supervised_layer.size,
		    params.supervised_layer.actf))
    for key,value in pairs(global_options.ann_options) do
      if layerwise_options.ann_options[key] == nil then
	thenet:set_option(key, value)
      end
    end
    for key,value in pairs(layerwise_options.ann_options) do
      thenet:set_option(key, value)
    end
    local loss_function
    if params.supervised_layer.size > 1 and
      (params.supervised_layer.actf == "log_logistic" or
       params.supervised_layer.actf == "log_softmax") then
	loss_function = ann.loss.multi_class_cross_entropy(params.supervised_layer.size)
    elseif (params.supervised_layer.size == 1 and
	    params.supervised_layer.actf == "log_logistic") then
      loss_function = ann.loss.cross_entropy(cod_size)
    else
      loss_function = ann.loss.mse(cod_size)
    end
    local thenet_trainer = trainable.supervised_trainer(thenet,
							loss_function,
							params.bunch_size)
    thenet_trainer:build()
    thenet_trainer:randomize_weights{
      random     = params.weights_random,
      inf=-math.sqrt(6 / (input_size + params.supervised_layer.size)),
      sup= math.sqrt(6 / (input_size + params.supervised_layer.size))
    }
    local best_net_trainer = thenet_trainer:train_wo_validation{
      min_epochs     = lookup("min_epochs"),
      max_epochs     = lookup("max_epochs"),
      training_table = data,
      percentage_stopping_criterion = lookup("pretraining_percentage_stopping_criterion"),
      update_function = function(t)
	printf("%4d %10.6f  (improvement %.4f)\n",
	       t.current_epoch, t.train_error, t.train_improvement)
	io.stdout:flush()	
      end
    }
    local wobj  = best_net_trainer:weights("w1"):clone()
    local bobj  = best_net_trainer:weights("b1"):clone()
    local lastn = #params.layers
    mlp_final:push( ann.components.hyperplane{
		      input  = input_size,
		      output = params.supervised_layer.size,
		      name = "layer" .. lastn,
		      dot_product_name = "weights" .. lastn,
		      bias_name = "bias" .. lastn,
		      dot_product_weights = "weights" .. lastn,
		      bias_weights = "bias" .. lastn, })
    mlp_final:push( ann.components[params.supervised_layer.actf]{
		      name="actf" .. lastn } )
    mlp_final_weights["weights" .. lastn] = wobj
    mlp_final_weights["bias"    .. lastn] = bobj
  end

  mlp_final:build{ weights = mlp_final_weights }
  
  return sdae_table, mlp_final
end

-- This function returns a MLP formed by the codification part of a full stacked
-- auto encoder
function ann.autoencoders.build_codifier_from_sdae_table(sdae_table,
							 layers)
  local weights_mat   = sdae_table.weights
  local bias_mat      = sdae_table.bias
  local codifier_net  = ann.components.stack()
  local weights_table = {}
  for i=2,#layers do
    local bname = "bias"..(i-1)
    local wname = "weights"..(i-1)
    codifier_net:push( ann.components.hyperplane{
			 input  = layers[i-1].size,
			 output = layers[i].size,
			 dot_product_weights = wname,
			 bias_weights        = bname })
    codifier_net:push( ann.components[layers[i].actf]() )
    weights_table[wname] = ann.connections{ input=layers[i-1].size,
					    output=layers[i].size,
					    w = weights_mat[i-1] }
    weights_table[bname] = ann.connections{ input=1, output=layers[i].size,
					    w = bias_mat[i-1][1] }
  end
  codifier_net:build{ weights = weights_table }
  return codifier_net
end

-- -- This function returns a MLP formed by the codification part of a full stacked
-- -- auto encoder
-- function ann.autoencoders.build_codifier_from_sdae(sdae, bunch_size, layers)
--   local sdae_connections = sdae:get_layer_connections_vector()
--   local sdae_activations = sdae:get_layer_activations_vector()
--   local codifier_net = ann.mlp{ bunch_size = bunch_size }
--   local codifier_connections = {}
--   local codifier_activations = {}
--   for i=1,(#layers-1)*2 do
--     table.insert(codifier_connections, sdae_connections[i]:clone(codifier_net))
--   end
--   local layer_type = "inputs"
--   for i=1,#layers-1 do
--     table.insert(codifier_activations, sdae_activations[i]:clone(codifier_net,
-- 								 layer_type))
--     layer_type = "hidden"
--   end
--   table.insert(codifier_activations, sdae_activations[#layers]:clone(codifier_net,
-- 								     "outputs"))
--   local k=1
--   for i=2,#layers do
--     local actf    = ann.activations.from_string(layers[i].actf)
--     local input   = codifier_activations[i-1]
--     local output  = codifier_activations[i]
--     local bias    = codifier_connections[k]
--     local weights = codifier_connections[k+1]
--     codifier_net:push_back_all_all_layer{
--       input   = input,
--       output  = output,
--       bias    = bias,
--       weights = weights,
--       actfunc = actf }
--     k = k + 2
--   end
--   return codifier_net
-- end

-- Returns a dataset with the codification of input dataset patterns  
function ann.autoencoders.compute_encoded_dataset_using_codifier()
  error("Deprecated, use trainable.supervised_trainer.use_dataset")
end
