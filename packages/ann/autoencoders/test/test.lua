m1 = ImageIO.read(string.get_path(arg[0]) ..  "digits.png"):to_grayscale():invert_colors():matrix()

bunch_size     = 32
dropout_factor = 0.5
dropout_random = random(123242)

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
  { size=  256, actf="logistic"}, -- INPUT LAYER
  { size=  256, actf="logistic"}, -- FIRST HIDDEN LAYER
}

perturbation_prob = random(9283424)
params_pretrain = {
  input_dataset         = train_input,
  replacement           = nil,
  shuffle_random        = random(1234),
  weights_random        = random(7890),
  
  layers                = layers_table,
  -- OUTPUT LAYER (SUPERVISED): classification task, 10 classes (10 digits)
  supervised_layer      = { size = 10, actf = "log_softmax" },
  output_datasets       = { train_output },
  
  bunch_size            = 256,
  optimizer             = function() return ann.optimizer.cg() end,
  
  -- training parameters
  training_options      = {
    global = {
      ann_options = { weight_decay  = 1e-05, rho=0.0001, sig=0.8 },
      noise_pipeline = { function(ds) return dataset.perturbation{
			     dataset  = ds, -- WARNING: the function argument
			     mean     = 0,
			     variance = 0.01,
			     random   = perturbation_prob } end,
			 function(ds) return dataset.salt_noise{
			     dataset  = ds, -- WARNING: the function argument
			     vd       = 0.20,
			     zero     = 0.0,
			     random   = perturbation_prob } end },
      min_epochs            = 1,
      max_epochs            = 2,
      pretraining_percentage_stopping_criterion = 0.1,
    },
  }
}

loss_name = "multi_class_cross_entropy"
sdae_table,deep_classifier = ann.autoencoders.greedy_layerwise_pretraining(params_pretrain)
codifier_net = ann.autoencoders.build_codifier_from_sdae_table(sdae_table,
							       layers_table)
function set_dropout(stack)
  if dropout_factor > 0.0 then
    local new_stack = ann.components.stack()
    for i=1,stack:size(),2 do
      new_stack:push( stack:get(i),
		      stack:get(i+1) )
      if i < stack:size()-2 then
	new_stack:push( ann.components.dropout{ name="dropout-".. (i+1)/2,
						prob=dropout_factor,
						random=dropout_random } )
      end
    end
    return new_stack
  else
    return stack
  end
end
deep_classifier = set_dropout(deep_classifier)
trainer_deep_classifier = trainable.supervised_trainer(deep_classifier,
						       ann.loss[loss_name](10),
						       bunch_size,
						       ann.optimizer.cg())
trainer_deep_classifier:build()
trainer_deep_classifier:set_option("rho", 0.0001)
trainer_deep_classifier:set_option("sig", 0.8)
--
shallow_classifier = ann.mlp.all_all.generate("256 inputs 256 tanh 128 tanh 10 log_softmax")
shallow_classifier = set_dropout(shallow_classifier)
trainer_shallow_classifier = trainable.supervised_trainer(shallow_classifier,
							  ann.loss[loss_name](10),
							  bunch_size)
trainer_shallow_classifier:build()
trainer_shallow_classifier:randomize_weights {
  random   = random(1234),
  inf      = -0.1,
  sup      =  0.1 }
--
deep_classifier_wo_pretraining = ann.mlp.all_all.generate("256 inputs 256 logistic 256 logistic 256 logistic 10 log_softmax")
deep_classifier_wo_pretraining = set_dropout(deep_classifier_wo_pretraining)
trainer_deep_wo_pretraining = trainable.supervised_trainer(deep_classifier_wo_pretraining,
							   ann.loss[loss_name](10),
							   bunch_size)
trainer_deep_wo_pretraining:build()
trainer_deep_wo_pretraining:randomize_weights{
  random   = random(1234),
  inf      = -0.1,
  sup      =  0.1 }

train_input = dataset.salt_noise{
  dataset = train_input,
  vd      = 0.2,
  zero    = 0.0,
  random  = random(95285) }

datosentrenar_deep = {
  input_dataset = train_input,
  output_dataset = train_output,
  shuffle = random(8569),
  bunch_size = 512,
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

-- we scale the weights before dropout
if dropout_factor > 0.0 then
  for name,cnn in trainer_deep_classifier:iterate_weights("^w.*$") do
    if name ~= "w1" then
      if cnn.matrix then
	local w,ow = cnn:matrix()
	w:scal(1.0/(1.0-dropout_factor))
	ow:scal(1.0/(1.0-dropout_factor))
      else
	cnn:scal(1.0/(1.0-dropout_factor))
      end
    end
  end
end
print(trainer_deep_classifier:validate_dataset(datosvalidar))

--trainer_deep_classifier:set_option("learning_rate", 0.4)
--trainer_deep_classifier:set_option("momentum", 0.0)
trainer_deep_classifier:set_option("weight_decay", 0.0)
trainer_deep_classifier:set_option("max_norm_penalty", 4.0)

trainer_shallow_classifier:set_option("learning_rate", 0.4)
trainer_shallow_classifier:set_option("momentum",
				      0.1)
trainer_shallow_classifier:set_option("weight_decay",
				      trainer_deep_classifier:get_option("weight_decay"))
trainer_shallow_classifier:set_option("max_norm_penalty",
				      trainer_deep_classifier:get_option("max_norm_penalty"))

trainer_deep_wo_pretraining:set_option("learning_rate",
				       trainer_shallow_classifier:get_option("learning_rate"))
trainer_deep_wo_pretraining:set_option("momentum",
				       trainer_shallow_classifier:get_option("momentum"))
trainer_deep_wo_pretraining:set_option("weight_decay",
				       trainer_deep_classifier:get_option("weight_decay"))
trainer_deep_wo_pretraining:set_option("max_norm_penalty",
				       trainer_deep_classifier:get_option("max_norm_penalty"))

trainer_deep_classifier:set_layerwise_option("b.*", "max_norm_penalty",0.0)
trainer_deep_wo_pretraining:set_layerwise_option("b.*", "max_norm_penalty",0.0)
trainer_shallow_classifier:set_layerwise_option("b.*", "max_norm_penalty",0.0)

trainer_deep_classifier:set_layerwise_option("b.*", "weight_decay",0.0)
trainer_deep_wo_pretraining:set_layerwise_option("b.*", "weight_decay",0.0)
trainer_shallow_classifier:set_layerwise_option("b.*", "weight_decay",0.0)

for i=1,2 do
  local mse_tr_deep = trainer_deep_classifier:train_dataset(datosentrenar_deep)
  local mse_tr_deep_wo = trainer_deep_wo_pretraining:train_dataset(datosentrenar_deep_wo)
  local mse_tr_shallow = trainer_shallow_classifier:train_dataset(datosentrenar_shallow)
  local mse_val_deep = trainer_deep_classifier:validate_dataset(datosvalidar)
  local mse_val_deep_wo = trainer_deep_wo_pretraining:validate_dataset(datosvalidar)
  local mse_val_shallow = trainer_shallow_classifier:validate_dataset(datosvalidar)
  printf("%5d %.6f %.6f \t %.6f %.6f \t %.6f %.6f\n", i,
	 mse_tr_deep, mse_val_deep,
	 mse_tr_deep_wo, mse_val_deep_wo,
	 mse_tr_shallow, mse_val_shallow)
  if trainer_deep_classifier:has_option("learning_rate") then
    trainer_deep_classifier:set_option("learning_rate",
				       trainer_deep_classifier:get_option("learning_rate")*0.99)
  end
end

-- classification
print(trainer_deep_classifier:validate_dataset{
	input_dataset  = val_input,
	output_dataset = val_output,
	loss = ann.loss.zero_one() })
print(trainer_deep_wo_pretraining:validate_dataset{
	input_dataset  = val_input,
	output_dataset = val_output,
	loss = ann.loss.zero_one() })
print(trainer_shallow_classifier:validate_dataset{
	input_dataset  = val_input,
	output_dataset = val_output,
	loss = ann.loss.zero_one() })

local img = ann.connections.input_filters_image(trainer_deep_classifier:weights("w1"),
						{16,16})
ImageIO.write(img, "/tmp/filters.png")
