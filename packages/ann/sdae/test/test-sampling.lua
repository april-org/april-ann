pnoise        = tonumber(arg[1] or 0.4)   -- noise percentage
loss_function = arg[2] or "cross_entropy"
alpha         = tonumber(arg[3] or 0.1)  -- SGD alpha parameter
beta          = tonumber(arg[4] or 0.2)  -- SGD beta parameter
seed          = tonumber(arg[5] or 12345) -- random seed

num_repetitions = 20
max_iterations  = 10
stop_criterion  = 1e-04

rnd = random(seed)

noise = ann.components.stack():
push(ann.components.gaussian_noise{ random=rnd, mean=0, var=0.2 }):
push(ann.components.salt_and_pepper{ random=rnd, prob=0.2 })
noise:build{ input=256, output=256 }

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

local ite_loss_stat = stats.mean_var()
local sgd_loss_stat = stats.mean_var()

local ite_quartiles = {}
local sgd_quartiles = {}

for ipat=1,val_input:numPatterns() do
  fprintf(io.stderr,"\r%3.0f%%", ipat/val_input:numPatterns()*100)
  io.stderr:flush()
  local target_class     = (ipat-1) % 10
  local current_ite_loss = 0
  local current_sgd_loss = 0
  for rep=1,num_repetitions do
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
    output,L = ann.autoencoders.iterative_sampling{
      model   = full_sdae,
      input   = input:rewrap(1,256):clone("col_major"),
      max     = max_iterations,
      mask    = mask,
      stop    = stop_criterion,
      verbose = false,
      log     = log,
      loss    = loss,
      noise   = noise,
    }
    -- matrix.saveImage(matrix(16,16,output), "wop1.pnm")
    -- output = table.imap(output, math.log)
    -- print(loss:loss(tokens.memblock(output),
    -- tokens.memblock(val_input:getPattern(n))), L)
    loss:reset()
    ite_L = loss:loss(tokens.matrix(output),
		      tokens.matrix(matrix.col_major(1,256,val_input:getPattern(ipat))))
    ite_loss_stat:add(ite_L)
    table.insert(ite_quartiles, ite_L)
    
    output,L = ann.autoencoders.sgd_sampling{
      model   = full_sdae,
      input   = input:rewrap(1,256):clone("col_major"),
      max     = max_iterations,
      mask    = mask,
      stop    = stop_criterion,
      verbose = false,
      alpha   = alpha,
      beta    = beta,
      log     = log,
      clamp   = function(mat) mat:clamp(0,1) end,
      loss    = loss,
      noise   = noise,
    }
    -- matrix.saveImage(matrix(16,16,output), "wop2.pnm")
    -- output = table.imap(output, math.log)
    -- print(loss:loss(tokens.memblock(output),
    -- tokens.memblock(val_input:getPattern(n))), L)
    loss:reset()
    sgd_L = loss:loss(tokens.matrix(output),
		      tokens.matrix(matrix.col_major(1,256,val_input:getPattern(ipat))))
    sgd_loss_stat:add(sgd_L)
    table.insert(sgd_quartiles, sgd_L)
  end
end
fprintf(io.stderr,"\n")
io.stderr:flush()
table.sort(ite_quartiles)
table.sort(sgd_quartiles)
local q1 = math.floor(#ite_quartiles*0.25)
local q2 = math.floor(#ite_quartiles*0.5)
local q3 = math.floor(#ite_quartiles*0.75)
local ite_loss_mean,ite_loss_var = ite_loss_stat:compute()
local sgd_loss_mean,sgd_loss_var = sgd_loss_stat:compute()
printf("%.1f  ITE: m= %.4f s2= %.4f Qs= %.4f %.4f %.4f %.4f %.4f\n",
       pnoise,
       ite_loss_mean, ite_loss_var,
       ite_quartiles[1],
       ite_quartiles[q1],
       ite_quartiles[q2],
       ite_quartiles[q3],
       ite_quartiles[#ite_quartiles])
printf("%.1f  SGD: m= %.4f s2= %.4f Qs= %.4f %.4f %.4f %.4f %.4f\n",
       pnoise,
       sgd_loss_mean, sgd_loss_var,
       sgd_quartiles[1],
       sgd_quartiles[q1],
       sgd_quartiles[q2],
       sgd_quartiles[q3],
       sgd_quartiles[#sgd_quartiles])
