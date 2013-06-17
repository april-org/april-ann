pnoise        = tonumber(arg[1] or 0.5)   -- noise percentage
loss_function = arg[2] or "cross_entropy"
alpha         = tonumber(arg[3] or 0.01)  -- SGD alpha parameter
seed          = tonumber(arg[4] or 12345) -- random seed

ipat = 80

num_repetitions = 100
max_iterations  = 100
stop_criterion  = 1e-08

m1 = ImageIO.read("digits.png"):to_grayscale():invert_colors():matrix()

val_input  = dataset.matrix(m1,
			    {
			      patternSize = {16,16},
			      offset      = {1280,0},
			      numSteps    = {20,10},
			      stepSize    = {16,16},
			      orderStep   = {1,0}
			    })
rnd       = random(seed)
trainer   = trainable.supervised_trainer.load("full-sdae.net")
full_sdae = trainer:get_component()
if loss_function == "local_fmeasure" then
  loss = ann.loss.local_fmeasure{ size=1, beta=2 }
  log  = false
  full_sdae:pop()
  full_sdae:push(ann.components.actf.logistic{ name = "actf-output" })
  trainer:build()
elseif loss_function ~= "mse" then
  loss = ann.loss[loss_function](full_sdae:get_output_size())
  log  = true
else
  loss = ann.loss.mse(full_sdae:get_output_size())
  log  = false
  full_sdae:pop()
  full_sdae:push(ann.components.actf.logistic{ name = "actf-output" })
  trainer:build()
end

noise = ann.components.gaussian_noise{ random=random(), mean=0, var=0.4 }
noise:build{ input=256, output=256 }

input = val_input:getPattern(ipat)
mask  = {}
k     = 1
for r=1,16 do
  local aux=math.max(0, math.min(16, math.round(pnoise*16)))
  for c=1,aux do
    input[k] = rnd:rand(1.0)
    k=k+1
  end
  for c=aux+1,16 do
    table.insert(mask,k)
    k=k+1
  end
end
matrix.saveImage(matrix(16,16,input), "wop0.pnm")

output,L,chain = ann.autoencoders.iterative_sampling{
  model   = full_sdae,
  noise   = noise,
  input   = input,
  max     = max_iterations,
  mask    = mask,
  stop    = stop_criterion,
  verbose = false,
  log     = log,
  loss    = loss,
}
for i,output in ipairs(chain) do
  matrix.saveImage(matrix(16,16,output), "wop1-"..string.format("%04d",i)..".pnm")
end
-- output = table.imap(output, math.log)
-- print(loss:loss(tokens.memblock(output),
-- tokens.memblock(val_input:getPattern(n))), L)
ite_L = loss:loss(tokens.memblock(output), tokens.memblock(val_input:getPattern(ipat)))
print(ite_L)

output,L,chain = ann.autoencoders.sgd_sampling{
  model   = full_sdae,
  input   = input,
  max     = max_iterations,
  mask    = mask,
  stop    = stop_criterion,
  verbose = false,
  alpha   = alpha,
  log     = log,
  clamp   = function(v) return math.max(0, math.min(1,v)) end,
  loss    = loss,
}
for i,output in ipairs(chain) do
  matrix.saveImage(matrix(16,16,output), "wop2-"..string.format("%04d",i)..".pnm")
end
-- matrix.saveImage(matrix(16,16,output), "wop2.pnm")
sgd_L = loss:loss(tokens.memblock(output), tokens.memblock(val_input:getPattern(ipat)))
print(sgd_L)
