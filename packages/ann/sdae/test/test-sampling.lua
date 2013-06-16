n = tonumber(arg[1] or 1)     -- validation pattern (up to 200)
p = tonumber(arg[2] or 0.5)   -- noise percentage
a = tonumber(arg[3] or 0.004) -- SGD alpha parameter

m1 = ImageIO.read("digits.png"):to_grayscale():invert_colors():matrix()

val_input  = dataset.matrix(m1,
			    {
			      patternSize = {16,16},
			      offset      = {1280,0},
			      numSteps    = {20,10},
			      stepSize    = {16,16},
			      orderStep   = {1,0}
			    })

trainer   = trainable.supervised_trainer.load("full-sdae.net")
full_sdae = trainer:get_component()
rnd       = random()
input     = val_input:getPattern(n)
loss      = ann.loss.cross_entropy(full_sdae:get_output_size())
mask      = {}
for i=1,#input do
  v = rnd:rand(1.0)
  if v < p then input[i] = rnd:rand(1.0)
  else table.insert(mask, i)
  end
end
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
print(loss:loss(tokens.memblock(output),
		tokens.memblock(val_input:getPattern(n))), L)

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
print(loss:loss(tokens.memblock(output),
		tokens.memblock(val_input:getPattern(n))), L)
