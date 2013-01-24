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

layers = { { size= 256, actf="logistic"},
	   { size=1204, actf="logistic"},
	   { size= 512, actf="logistic"},
	   { size= 128, actf="logistic"},
	   { size=   2, actf="logistic"}}

params = {
  input_dataset       = train_input,
  val_input_dataset   = val_input,
  replacement         = nil,
  shuffle_random      = random(1234),
  perturbation_random = random(4567),
  weights_random      = random(7890),
  var                 = 0.02,
  layers              = layers,
  bunch_size          = 8,
  learning_rate       = 0.01,
  momentum            = 0.02,
  weight_decay        = 1e-05,
  max_epochs          = 200,
  max_epochs_wo_improvement = 10 
}

sdae_table = ann.autoencoders.stacked_denoising_pretraining(params)
sdae       = ann.autoencoders.stacked_denoising_finetunning(sdae_table, params)
codifier_net = ann.autoencoders.build_codifier_from_sdae(sdae, 16, layers)

local outf = io.open("data", "w")
encoded_dataset = ann.autoencoders.compute_encoded_dataset_using_codifier(codifier_net,
									  train_input)
for ipat,pat in encoded_dataset:patterns() do
  fprintf(outf, "Pattern %d %s\n", ipat, table.concat(pat, " "))
end

encoded_dataset = ann.autoencoders.compute_encoded_dataset_using_codifier(codifier_net,
									  val_input)
for ipat,pat in encoded_dataset:patterns() do
  fprintf(outf, "Pattern %d %s\n", ipat, table.concat(pat, " "))
end
outf:close()
