pnoise        = tonumber(arg[1] or 0.5)   -- noise percentage
loss_function = arg[2] or "mse"
alpha         = tonumber(arg[3] or 0.01)  -- SGD alpha parameter
seed          = tonumber(arg[4] or 12345) -- random seed

num_repetitions = 100
max_iterations  = 100
stop_criterion  = 1e-04

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
    mask = {}
    for i=1,#input do
      v = rnd:rand(1.0)
      if v < pnoise then input[i] = rnd:rand(1.0)
      else table.insert(mask, i)
      end
    end
    output,L = ann.autoencoders.iterative_sampling{
      model   = full_sdae,
      input   = input,
      max     = max_iterations,
      mask    = mask,
      stop    = stop_criterion,
      verbose = false,
      log     = log,
      loss    = loss,
    }
    -- matrix.saveImage(matrix(16,16,output), "wop1.pnm")
    -- output = table.imap(output, math.log)
    -- print(loss:loss(tokens.memblock(output),
    -- tokens.memblock(val_input:getPattern(n))), L)
    loss:reset()
    ite_L = loss:loss(tokens.memblock(output),
		      tokens.memblock(val_input:getPattern(ipat)))
    ite_loss_stat:add(ite_L)
    table.insert(ite_quartiles, ite_L)
    
    output,L = ann.autoencoders.sgd_sampling{
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
    -- matrix.saveImage(matrix(16,16,output), "wop2.pnm")
    -- output = table.imap(output, math.log)
    -- print(loss:loss(tokens.memblock(output),
    -- tokens.memblock(val_input:getPattern(n))), L)
    loss:reset()
    sgd_L = loss:loss(tokens.memblock(output),
		      tokens.memblock(val_input:getPattern(ipat)))
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
