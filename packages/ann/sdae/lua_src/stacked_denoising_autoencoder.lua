ann.autoencoders = ann.autoencoders or {}

function ann.autoencoders.build_autoencoder_from_sizes_and_actf(input_size,
								input_actf,
								cod_size,
								cod_actf)
  local autoencoder  = ann.mlp(params.bunch_size)
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
  local hidden_weights  = ann.connections.all_all{ ann         = autoencoder,
						   input_size  = input_size,
						   output_size = cod_size }
  local output_bias = ann.connections.bias{ ann  = autoencoder,
					    size = input_size }
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
  return autoencoder,hidden_weights,hidden_bias,output_bias
end

-- Params is a table which could contain:
--   * input_dataset => dataset with input (and output) for AE
--   * val_input_dataset => for validation
--   * distribution => a table which contains a list of {input_dataset=...., prob=....}
--   * replacement => replacement value for training
--   * shuffle_random => random number generator
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
-- This function returns two tables: weights and bias
-- { w1, w2, ..., wN } where wi are matrices of April
-- { {b1, b2},  {b3, b4}, ... } where bi are matrices of April
--
-- If you train an auto-encoder for a topology of 256 128 64
-- the WHOLE auto-encoder will had this topology:
-- 256 - 128 - 64 - 128 - 256
-- So it has four layers: (1) 256-128, (2) 128-64, (3) 64-128, (4) 128-256
-- and this function will return:
-- w1 => layer (1) and (4)
-- w2 => layer (2) and (3)
-- b1 => 128 at layer (1)
-- b2 => 256 at layer (4)
-- b3 => 64 at layer (2)
-- b4 => 128 at layer (3)
function ann.autoencoders.stacked_denoising_pretraining(params)
  local check_mandatory_param = function(params, name)
    if not params[name] then error ("Parameter " .. name .. " is mandatory") end
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
			 "momentum", "weight_decay", "val_input_dataset"}) do
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
    local input_actf = ann.activations.from_string(params.layers[i-1].actf)
    local cod_actf   = ann.activations.from_string(params.layers[i].actf)
    local data = { shuffle = params.shuffle_random }
    local val_data = current_val_dataset_params
    if current_dataset_params.input_dataset then
      data = {
	-- The input is perturbed with gaussian noise
	input_dataset = dataset.perturbation{
	  dataset  = current_dataset_params.input_dataset,
	  mean     = 0,
	  variance = params.var,
	  random   = params.perturbation_random },
	output_dataset = current_dataset_params.input_dataset,
      }
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
    local dae
    dae,weights,bias1,bias2 = ann.autoencoders.build_autoencoder_from_sizes_and_actf(input_size,
										     input_actf,
										     cod_size,
										     cod_actf)
    collectgarbage("collect")
    dae:set_error_function(ann.error_functions.full_cross_entropy())
    local best_val_error = 111111111
    local best_net       = dae:clone()
    local best_epoch     = 0
    for epoch=1,params.max_epochs do
      local train_error = dae:train_dataset(data)
      local val_error   = dae:validate_dataset(val_data)
      if val_error < best_val_error then
	best_val_error = val_error
	best_epoch     = epoch
	best_net       = dae:clone()
      end
      printf("%4d %10.6f %10.6f  (best %10.6f at epoch %4d)\n",
	     train_error, val_error, best_val_error, best_epoch)
      collectgarbage("collect")
      -- convergence criteria
      if epoch - best_epoch > params.max_epochs_wo_improvement then break end
      local wmat  = weights:weights()
      local b1mat = bias1:weights()
      local b2mat = bias2:weights()
      table.insert(weights, wmat)
      table.insert(bias, { b1mat, b2mat })
    end
  end -- for i=2,#params.layers
  return weights,bias
end
