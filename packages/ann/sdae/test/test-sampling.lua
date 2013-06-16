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

n         = 54
trainer   = trainable.supervised_trainer.load("full-sdae.net")
full_sdae = trainer:get_component()
rnd       = random()
input     = val_input:getPattern(n)
loss      = ann.loss.multi_class_cross_entropy(full_sdae:get_output_size())
mask      = {}
for i=1,50 do table.insert(mask, i) end
for i=51,full_sdae:get_input_size() do input[i] = rnd:rand(1.0) end
-- for i=1,full_sdae:get_input_size() do input[i] = rnd:rand(1.0) end
output,L = ann.autoencoders.iterative_sampling{
  model   = full_sdae,
  input   = input,
  max     = 1000,
  mask    = mask,
  stop    = 1e-03,
  verbose = false,
  log     = true,
}
matrix.saveImage(matrix(16,16,output), "wop1.pnm")
output = table.imap(output, math.log)
print(L, loss:loss(tokens.memblock(output),
		   tokens.memblock(val_input:getPattern(n))) )

output,L = ann.autoencoders.sgd_sampling{
  model   = full_sdae,
  input   = input,
  max     = 100,
  mask    = mask,
  stop    = 1e-08,
  verbose = false,
  alpha   = 0.004,
  log     = true,
  clamp   = function(v) return math.max(0, math.min(1,v)) end,
}
matrix.saveImage(matrix(16,16,output), "wop2.pnm")
output = table.imap(output, math.log)
print(L, loss:loss(tokens.memblock(output),
		   tokens.memblock(val_input:getPattern(n))) )
