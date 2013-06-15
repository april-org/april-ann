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
			 function(ds) return dataset.salt_noise{
			     dataset  = ds, -- WARNING: the function argument
			     vd       = 0.10,
			     zero     = 0.0,
			     random   = perturbation_prob } end },
      min_epochs            = 10,
      max_epochs            = 200,
      pretraining_percentage_stopping_criterion = 0.01,
    },
  }
}

sdae_table,deep_classifier = ann.autoencoders.greedy_layerwise_pretraining(params_pretrain)
full_sdae = ann.autoencoders.build_full_autoencoder(layers_table, sdae_table)
rnd       = random()
input     = val_input:getPattern(10)
mask      = {}
--for i=1,full_sdae:get_input_size() do input[i] = rnd:rand(0.1) end
for i=1,10 do table.insert(mask, i) end
for i=11,full_sdae:get_input_size() do input[i] = rnd:rand(0.1) end
output = ann.autoencoders.iterative_sampling{
  model   = full_sdae,
  input   = input,
  max     = 1000,
  mask    = mask,
  stop    = 1e-06,
  verbose = false,
}
matrix.saveImage(matrix(16,16,output), "wop.pnm")

output = ann.autoencoders.sgd_sampling{
  model   = full_sdae,
  input   = input,
  max     = 1000,
  mask    = mask,
  stop    = 1e-06,
  verbose = false,
  alpha   = 0.01,
  --clamp   = function(v) return math.max(0, math.min(1,v)) end,
}
matrix.saveImage(matrix(16,16,output), "wop2.pnm")
