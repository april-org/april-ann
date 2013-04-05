m1 = ImageIO.read("digits.png"):to_grayscale():invert_colors():matrix()

bunch_size = 8

train_input = dataset.matrix(m1,
			     {
			       patternSize = {16,16},
			       offset      = {0,0},
			       numSteps    = {80,10},
			       stepSize    = {16,16},
			       orderStep   = {1,0}
			     })

val_input  = dataset.matrix(m1,
			    {
			      patternSize = {16,16},
			      offset      = {1280,0},
			      numSteps    = {20,10},
			      stepSize    = {16,16},
			      orderStep   = {1,0}
			    })

m2 = matrix(10,{1,0,0,0,0,0,0,0,0,0})
train_output = dataset.matrix(m2,
			      {
				patternSize = {10},
				offset      = {0},
				numSteps    = {800},
				stepSize    = {-1},
				circular    = {true}
			      })

val_output   = dataset.matrix(m2,
			      {
				patternSize = {10},
				offset      = {0},
				numSteps    = {200},
				stepSize    = {-1},
				circular    = {true}
			      })

layers_table = {
  { size= 256, actf="logistic"}, -- INPUT LAYER
  { size= 256, actf="logistic"}, -- FIRST HIDDEN LAYER
  { size= 128, actf="logistic"}, -- SECOND HIDDEN LAYER
  { size=  32, actf="logistic"}, -- THIRD HIDDEN LAYER
}

params_pretrain = {
  input_dataset         = train_input,
  replacement           = nil,
  shuffle_random        = random(1234),
  weights_random        = random(7890),
  
  layers                = layers_table,
  -- OUTPUT LAYER (SUPERVISED): classification task, 10 classes (10 digits)
  supervised_layer      = { size = 10, actf = "softmax" },
  output_datasets       = { train_output },
  
  bunch_size            = bunch_size,
  
  -- training parameters
  training_options      = {
    global = {
      ann_options = { learning_rate = 0.1,
		      momentum      = 0.02,
		      weight_decay  = 1e-05 },
      noise_pipeline = { function(ds) return dataset.perturbation{
			     dataset  = ds, -- WARNING: the function argument
			     mean     = 0,
			     variance = 0.01,
			     random   = random(8249824) } end,
			 function(ds) return dataset.salt_noise{
			     dataset  = ds, -- WARNING: the function argument
			     vd       = 0.10,
			     zero     = 0.0,
			     random   = random(8249824) } end },
      min_epochs            = 4,
      max_epochs            = 200,
      pretraining_percentage_stopping_criterion = 0.01,
    },
    layerwise = { { min_epochs=50 },
		  { min_epochs=20 },
		  { ann_options = { learning_rate = 0.4,
				    momentum      = 0.02,
				    weight_decay  = 4e-05 },
		    min_epochs=20 },
		  { min_epochs=10 }, },
  }
}

sdae_table,deep_classifier = ann.autoencoders.greedy_layerwise_pretraining(params_pretrain)
codifier_net = ann.autoencoders.build_codifier_from_sdae_table(sdae_table,
							       bunch_size,
							       layers_table)
deep_classifier = ann.mlp.convert_to_all_all{
  ann = deep_classifier,
  topology = "256 inputs 256 logistic 128 logistic 32 logistic 10 softmax",
  bunch_size = bunch_size }

local outf = io.open("data", "w")
encoded_dataset = ann.autoencoders.encode_dataset(codifier_net,
						  train_input)
for ipat,pat in encoded_dataset:patterns() do
  fprintf(outf, "Pattern %d %s\n", ipat, table.concat(pat, " "))
end

encoded_dataset = ann.autoencoders.encode_dataset(codifier_net,
						  val_input)
for ipat,pat in encoded_dataset:patterns() do
  fprintf(outf, "Pattern %d %s\n", ipat, table.concat(pat, " "))
end
outf:close()

--
shallow_classifier = ann.mlp.all_all.generate{
  topology = "256 inputs 256 tanh 128 tanh 10 softmax",
  random   = random(1234),
  inf      = -1,
  sup      =  1,
  bunch_size = bunch_size,
  use_fanin = true}
shallow_classifier:set_error_function(ann.error_functions.logistic_cross_entropy())

-- deep_classifier = ann.mlp.add_layers{
--   ann        = codifier_net,
--   new_layers = { { 10, "softmax" } },
--   bunch_size = bunch_size,
--   random     = random(1234),
--   inf        = -0.1,
--   sup        = 0.1 }
-- deep_classifier:set_error_function(ann.error_functions.logistic_cross_entropy())

deep_classifier_wo_pretraining = ann.mlp.all_all.generate{
  topology = "256 inputs 256 logistic 128 logistic 32 logistic 10 softmax",
  random   = random(1234),
  inf      = -1,
  sup      =  1,
  use_fanin = true,
  bunch_size = bunch_size }
deep_classifier_wo_pretraining:set_error_function(ann.error_functions.logistic_cross_entropy())

train_input = dataset.salt_noise{
  dataset = train_input,
  vd      = 0.2,
  zero    = 0.0,
  random  = random(95285) }

datosentrenar_deep = {
  input_dataset = train_input,
  output_dataset = train_output,
  shuffle = random(8569)
}
datosentrenar_shallow = {
  input_dataset = train_input,
  output_dataset = train_output,
  shuffle = random(8569)
}
datosentrenar_deep_wo = {
  input_dataset = train_input,
  output_dataset = train_output,
  shuffle = random(8569)
}

datosvalidar = {
  input_dataset = val_input,
  output_dataset = val_output
}

deep_classifier:set_option("learning_rate", 1.0)
deep_classifier:set_option("momentum", 0.2)
deep_classifier:set_option("weight_decay", 0.0)
deep_classifier:set_option("neuron_squared_length_upper_bound", 15.0);
deep_classifier:set_option("dropout", 0.5)
deep_classifier:set_error_function(ann.error_functions.logistic_cross_entropy())

shallow_classifier:set_option("learning_rate", 0.4)
shallow_classifier:set_option("momentum",
			      deep_classifier:get_option("momentum"))
shallow_classifier:set_option("weight_decay",
			      deep_classifier:get_option("weight_decay"))
shallow_classifier:set_option("neuron_squared_length_upper_bound",
			      deep_classifier:get_option("neuron_squared_length_upper_bound"))
shallow_classifier:set_option("dropout",
			      deep_classifier:get_option("dropout"))
shallow_classifier:set_error_function(ann.error_functions.logistic_cross_entropy())

deep_classifier_wo_pretraining:set_option("learning_rate",
					  shallow_classifier:get_option("learning_rate"))
deep_classifier_wo_pretraining:set_option("momentum",
					  deep_classifier:get_option("momentum"))
deep_classifier_wo_pretraining:set_option("weight_decay",
					  deep_classifier:get_option("weight_decay"))
deep_classifier:set_option("neuron_squared_length_upper_bound",
			   deep_classifier:get_option("neuron_squared_length_upper_bound"))
deep_classifier_wo_pretraining:set_option("dropout",
					  deep_classifier:get_option("dropout"))
deep_classifier_wo_pretraining:set_error_function(ann.error_functions.logistic_cross_entropy())

for i=1,200 do
  local mse_tr_deep = deep_classifier:train_dataset(datosentrenar_deep)
  local mse_tr_deep_wo = deep_classifier_wo_pretraining:train_dataset(datosentrenar_deep_wo)
  local mse_tr_shallow = shallow_classifier:train_dataset(datosentrenar_shallow)
  local mse_val_deep = deep_classifier:validate_dataset(datosvalidar)
  local mse_val_deep_wo = deep_classifier_wo_pretraining:validate_dataset(datosvalidar)
  local mse_val_shallow = shallow_classifier:validate_dataset(datosvalidar)
  printf("%5d %.6f %.6f \t %.6f %.6f \t %.6f %.6f\n", i,
	 mse_tr_deep, mse_val_deep,
	 mse_tr_deep_wo, mse_val_deep_wo,
	 mse_tr_shallow, mse_val_shallow)
  deep_classifier:set_option("learning_rate",
			     deep_classifier:get_option("learning_rate")*0.99)
end

-- classification
local deep_out_ds = dataset.matrix(matrix(val_output:numPatterns(),
					  val_output:patternSize()))
local deep_wo_out_ds = dataset.matrix(matrix(val_output:numPatterns(),
					     val_output:patternSize()))
local shallow_out_ds = dataset.matrix(matrix(val_output:numPatterns(),
					     val_output:patternSize()))

deep_classifier:use_dataset{ input_dataset  = val_input,
			     output_dataset = deep_out_ds }

deep_classifier_wo_pretraining:use_dataset{input_dataset =val_input,
					   output_dataset=deep_wo_out_ds}

shallow_classifier:use_dataset{ input_dataset  = val_input,
				output_dataset = shallow_out_ds }

local errors = {0,0,0}

for ipat,pat in val_input:patterns() do
  local _,class         = table.max(val_output:getPattern(ipat))
  local _,deep_class    = table.max(deep_out_ds:getPattern(ipat))
  local _,deep_wo_class = table.max(deep_wo_out_ds:getPattern(ipat))
  local _,shallow_class = table.max(shallow_out_ds:getPattern(ipat))
  if class ~= deep_class then errors[1] = errors[1] + 1 end
  if class ~= deep_wo_class then errors[2] = errors[2] + 1 end
  if class ~= shallow_class then errors[3] = errors[3] + 1 end
end

print(errors[1]/val_input:numPatterns(),
      errors[2]/val_input:numPatterns(),
      errors[3]/val_input:numPatterns())
