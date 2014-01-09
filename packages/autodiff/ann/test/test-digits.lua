local learning_rate  = 0.1
local momentum       = 0.1
local weight_decay   = 1e-04
local semilla        = 1234
local rnd            = random(semilla)
local H1             = 256
local H2             = 128
local M              = matrix.col_major
local bunch_size     = 32
--
--------------------------------------------------------------

m1 = ImageIO.read(string.get_path(arg[0]) .. "../../../../TEST/digitos/digits.png"):to_grayscale():invert_colors():matrix()
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
-- una matriz pequenya la podemos cargar directamente
m2 = matrix(10,{1,0,0,0,0,0,0,0,0,0})

-- ojito con este dataset, fijaros que usa una matriz de dim 1 y talla
-- 10 PERO avanza con valor -1 y la considera CIRCULAR en su unica
-- dimension

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
------------------------------------------------------------------------------
local INPUT  = train_input:patternSize()
local OUTPUT = train_output:patternSize()

local weights = {
  b1 = M(H1,1),
  w1 = M(H1,INPUT),
  b2 = M(H2,1),
  w2 = M(H2,H1),
  b3 = M(OUTPUT,1),
  w3 = M(OUTPUT,H2),
}
local weights_list = iterator(pairs(weights)):select(1):table()
table.sort(weights_list)

local AD = autodiff
local op = AD.op
local func = AD.func
local b1,w1,b2,w2,b3,w3,x,y,loss_input = AD.matrix('b1 w1 b2 w2 b3 w3 x y loss_input')
-- it is possible to use AD.scalar('wd') if you want to change weight decay
-- during learning
local wd = weight_decay

x:set_dims(INPUT,0)
y:set_dims(OUTPUT,0)

b1:set_broadcast(false, true)
b2:set_broadcast(false, true)
b3:set_broadcast(false, true)

-- ANN
local net_h1  = AD.ann.relu(b1 + w1 * x)       -- first layer
local net_h2  = AD.ann.relu(b2 + w2 * net_h1)  -- second layer
local net_out = b3 + w3 * net_h2            -- output layer (linear, the softmax
					    -- is added to the loss function)

local net = AD.op.exp( AD.ann.log_softmax(net_out, 2) ) -- softmax (for testing)
-- Loss function: negative cross-entropy with the log-softmax (for training)
local L = AD.op.mean( AD.ann.cross_entropy_log_softmax(net_out, y, 2) )
-- Regularization component
local Lreg = L + 0.5 * wd * (op.sum(w1^2) + op.sum(w2^2) + op.sum(w3^2))
-- Differentiation, plus loss computation
local dw_tbl = table.pack( Lreg, AD.diff(Lreg, {b1, w1, b2, w2, b3, w3}) )

-- Compilation
local L_func = AD.func(L, {x,y}, weights)
local dw_func = AD.func(dw_tbl, {x,y}, weights)
local dw_program = dw_func.program
--
g = io.open("program.lua","w")
g:write(dw_program)
g:close()
--

-- RANDOMIZATION
for _,wname in ipairs(weights_list) do
  local w = weights[wname]
  if wname:match("b.") then
    ann.connections.randomize_weights(w, { inf=0, sup=1, random=rnd })
  else
    ann.connections.randomize_weights(w, { inf=-0.1, sup=0.1, random=rnd })
  end
end

-- OPTIMIZER
local opt = ann.optimizer.sgd()
opt:set_option("learning_rate", learning_rate)
opt:set_option("momentum", momentum)
--

-- WEIGHTS DICTIONARY
local weights_dict = matrix.dict(weights)

local ds_pair_it = trainable.dataset_pair_iterator
-- traindataset
local function train_dataset(in_ds,out_ds)
  local mv = stats.mean_var()
  for input_bunch,output_bunch in ds_pair_it{ input_dataset=in_ds,
					      output_dataset=out_ds,
					      bunch_size=bunch_size,
					      shuffle = rnd, } do
    local loss
    loss = opt:execute(function()
			 local loss,b1,w1,b2,w2,
			 b3,w3 = dw_func(input_bunch:get_matrix():transpose(),
					 output_bunch:get_matrix():transpose())
			 return loss, { b1=b1, w1=w1, b2=b2, w2=w2, b3=b3, w3=w3 }
		       end,
		       weights_dict)
    mv:add(loss)
  end
  return mv:compute()
end

-- validatedataset
local function validate_dataset(in_ds,out_ds)
  local mv = stats.mean_var()
  for input_bunch,output_bunch in ds_pair_it{ input_dataset=in_ds,
					      output_dataset=out_ds,
					      bunch_size=bunch_size } do
    local loss = L_func(input_bunch:get_matrix():transpose(),
			output_bunch:get_matrix():transpose())
    mv:add(loss)
  end
  return mv:compute()
end

-- TRAINING
local train_func = trainable.train_holdout_validation{ min_epochs=100,
						       max_epochs=100 }
while train_func:execute(function()
			   local tr_loss = train_dataset(train_input,
							 train_output)
			   local va_loss = validate_dataset(val_input,
							    val_output)
			   return weights_dict,tr_loss,va_loss
			 end) do
  print(train_func:get_state_string())
end
