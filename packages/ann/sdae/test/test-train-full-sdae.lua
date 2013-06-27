m1 = ImageIO.read("test/digits.png"):to_grayscale():invert_colors():matrix()

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

layers_table = {
  { size=  256, actf="logistic"}, -- INPUT LAYER
  { size= 1024, actf="logistic"}, -- FIRST HIDDEN LAYER
  { size= 1024, actf="logistic"}, -- SECOND HIDDEN LAYER
  { size=   32, actf="logistic"}, -- THIRD HIDDEN LAYER
}

perturbation_prob = random(9283424)
params_pretrain = {
  input_dataset         = train_input,
  replacement           = nil,
  shuffle_random        = random(1234),
  weights_random        = random(7890),
  
  layers                = layers_table,
  
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
			     random   = perturbation_prob } end,
			 function(ds) return dataset.salt_pepper_noise{
			     dataset  = ds, -- WARNING: the function argument
			     vd       = 0.30,
			     zero     = 0.0,
			     one      = 1.0,
			     random   = perturbation_prob } end },
      min_epochs            = 10,
      max_epochs            = 200,
      pretraining_percentage_stopping_criterion = 0.01,
    },
  }
}
sdae_table,deep_classifier = ann.autoencoders.greedy_layerwise_pretraining(params_pretrain)
layers_table[1].actf = "log_logistic"
full_sdae = ann.autoencoders.build_full_autoencoder(layers_table, sdae_table)
trainer = trainable.supervised_trainer(full_sdae,
				       ann.loss.multi_class_cross_entropy(layers_table[1].size),
				       bunch_size)
trainer:build()

train_input_wo_noise = train_input
train_input = params_pretrain.training_options.global.noise_pipeline[1](train_input)
train_input = params_pretrain.training_options.global.noise_pipeline[2](train_input)

full_sdae:set_option("learning_rate", 0.00001)
full_sdae:set_option("momentum", 0.00002)
full_sdae:set_option("weight_decay", 0.0)

result = trainer:train_holdout_validation{
  epochs_wo_validation = 2,
  max_epochs = 200,
  min_epochs = 10,
  stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(2),
  training_table = {
    input_dataset  = train_input,
    output_dataset = train_input_wo_noise,
    shuffle        = random(95284),
    replacement    = 100,
  },
  validation_table = {
    input_dataset  = val_input,
    output_dataset = val_input,
  },
  update_function = function(t)
    print(t.current_epoch, t.train_error, t.validation_error)
  end,
}
result.best:save("full-sdae.net", "binary")
