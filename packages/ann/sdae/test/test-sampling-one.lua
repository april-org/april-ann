pnoise        = tonumber(arg[1] or 0.4)   -- noise percentage
loss_function = arg[2] or "cross_entropy"
alpha         = tonumber(arg[3] or 0.1)  -- SGD alpha parameter
beta          = tonumber(arg[4] or 0.2)
seed          = tonumber(arg[5] or 12345) -- random seed

ipat = 16

max_iterations  = 100
stop_criterion  = 1e-8

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

input = matrix(16, 16, val_input:getPattern(ipat))
mask  = {}
local k = 1
local number_of_blank_cols=math.max(0, math.min(16, math.round(pnoise*16)))
for r=1,16 do
  map(function(c) input:set(r, c, rnd:rand(1.0)) k=k+1 end,
      range, 1, number_of_blank_cols, 1)
  map(function(c) table.insert(mask, k) k=k+1 end,
      range, number_of_blank_cols+1, 16, 1)
end
matrix.saveImage(input, "wop0.pnm")

noise = ann.components.stack():
push(ann.components.gaussian_noise{ random=random(), mean=0, var=0.2 }):
push(ann.components.salt_and_pepper{ random=random(), prob=0.2 })
noise:build{ input=256, output=256 }

output,L,chain = ann.autoencoders.iterative_sampling{
  model   = full_sdae,
  noise   = noise,
  input   = input:rewrap(1,256):clone("col_major"),
  max     = max_iterations,
  mask    = mask,
  stop    = stop_criterion,
  verbose = false,
  log     = log,
  loss    = loss,
}
for i,output in ipairs(chain) do
  matrix.saveImage(output:clone("row_major"):rewrap(16,16),
		   "wop1-"..string.format("%04d",i)..".pnm")
end
if log then output:log() end
ite_L = loss:loss(tokens.matrix(output:rewrap(1,256)),
		  tokens.matrix(matrix.col_major(1,256,val_input:getPattern(ipat))))
print(ite_L)

output,L,chain = ann.autoencoders.sgd_sampling{
  model   = full_sdae,
  noise   = noise,
  input   = input:rewrap(1,256):clone("col_major"),
  max     = max_iterations,
  mask    = mask,
  stop    = stop_criterion,
  verbose = false,
  alpha   = alpha,
  beta    = beta,
  log     = log,
  clamp   = function(mat) mat:clamp(0, 1) end,
  loss    = loss,
}
for i,output in ipairs(chain) do
  matrix.saveImage(output:clone("row_major"):rewrap(16,16),
		   "wop2-"..string.format("%04d",i)..".pnm")
end
if log then output:log() end
sgd_L = loss:loss(tokens.matrix(output:rewrap(1,256)),
		  tokens.matrix(matrix.col_major(1,256,val_input:getPattern(ipat))))
print(sgd_L)
