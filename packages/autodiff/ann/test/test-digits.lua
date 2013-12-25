local learning_rate  = 0.1
local momentum       = 0.1
local weight_decay   = 1e-04
local semilla        = 1234
local rnd            = random(semilla)
local H1             = 256
local H2             = 128
local M              = matrix.col_major
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

local AD = autodiff
local op = AD.op
local func = AD.func
local b1,w1,b2,w2,b3,w3,x,y,loss_input = AD.matrix('b1 w1 b2 w2 b3 w3 x y loss_input')
-- it is possible to use AD.scalar('wd') if you want to change weight decay
-- during learning
local wd = weight_decay

x:set_dims(INPUT,1)
y:set_dims(OUTPUT,1)

b1:set_broadcast(false, true)
b2:set_broadcast(false, true)
b3:set_broadcast(false, true)

-- ANN
local net_h1  = AD.ann.logistic(b1 + w1 * x)       -- first layer
local net_h2  = AD.ann.logistic(b2 + w2 * net_h1)  -- second layer
local net_out = b3 + w3 * net_h2            -- output layer (linear, the softmax
					    -- is added to the loss function)

-- Loss function: negative cross-entropy with the log-softmax (for training)
local L = AD.ann.cross_entropy_log_softmax(loss_input, y)
-- Regularization
L = L + 0.5 * wd * (op.sum(w1^2) + op.sum(w2^2) + op.sum(w3^2))

-- Compilation using AD.ann helpers
local thenet = AD.ann.component("mynet", net_out, x) -- without softmax layer
local loss   = AD.ann.loss(L, loss_input, y)

-- TRAINER
trainer = trainable.supervised_trainer(thenet, loss, 1)
trainer:build{ weights=weights }
trainer:randomize_weights{ inf=-0.1, sup=0.1, random=rnd }
trainer:set_option("learning_rate", learning_rate)
trainer:set_option("momentum", momentum)
--

-- it is important to give thenet to the loss function, in order to reuse
-- memorized (cached) computations
loss:compile(weights, thenet)
--

local train_func = trainable.train_holdout_validation{ min_epochs=100,
						       max_epochs=100 }
while train_func:execute(function()
			   local tr_loss = trainer:train_dataset{
			     input_dataset  = train_input,
			     output_dataset = train_output,
			     shuffle        = rnd
			   }
			   local va_loss = trainer:validate_dataset{
			     input_dataset  = val_input,
			     output_dataset = val_output,
			   }
			   return trainer,tr_loss,va_loss
			 end) do
  print(train_func:get_state_string())
end
