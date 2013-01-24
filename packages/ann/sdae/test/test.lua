dofile("../lua_src/stacked_denoising_autoencoder.lua")

m1 = ImageIO.read("digits.png"):to_grayscale():invert_colors():matrix()
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

weights,bias = ann.autoencoders.stacked_denoising_pretraining{
  input_dataset       = train_input,
  val_input_dataset   = val_input,
  replacement         = nil,
  shuffle_random      = random(1234),
  perturbation_random = random(4567),
  weights_random      = random(7890),
  var                 = 0.02,
  layers              = { { size= 256, actf="logistic"},
			  { size=1024, actf="logistic"},
			  { size=  32, actf="logistic"},
			  { size=   2, actf="linear"}},
  bunch_size          = 16,
  learning_rate       = 0.001,
  momentum            = 0.02,
  weight_decay        = 1e-05,
  max_epochs          = 200,
  max_epochs_wo_improvement = 10 }
