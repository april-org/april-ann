local learning_rate  = 0.1
local momentum       = 0.0 --0.1
local weight_decay   = 1e-02
local semilla        = 1234
local rnd            = random(semilla)
local H1             = 256
local H2             = 128
local M              = matrix.col_major
--
--------------------------------------------------------------

m1 = ImageIO.read(string.get_path(arg[0]) .. "../../../TEST/digitos/digits.png"):to_grayscale():invert_colors():matrix()
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
  b1 = M(H1,1):uniformf(-0.01,0.01,rnd),
  w1 = M(H1,INPUT):uniformf(-0.01,0.01,rnd),
  b2 = M(H2,1):uniformf(-0.01,0.01,rnd),
  w2 = M(H2,H1):uniformf(-0.01,0.01,rnd),
  b3 = M(OUTPUT,1):uniformf(-0.01,0.01,rnd),
  w3 = M(OUTPUT,H2):uniformf(-0.01,0.01,rnd),
}

local AD = autodiff
local op = AD.op
local func = AD.func
local b1,w1,b2,w2,b3,w3,x,y,seed = AD.matrix('b1 w1 b2 w2 b3 w3 x y seed')
-- it is possible to use AD.scalar('wd') if you want to change weight decay
-- during learning
local wd = weight_decay

b1:set_dims(weights.b1:dim())
w1:set_dims(weights.w1:dim())
b2:set_dims(weights.b2:dim())
w2:set_dims(weights.w2:dim())
b3:set_dims(weights.b3:dim())
w3:set_dims(weights.w3:dim())
x:set_dims(INPUT,1)
y:set_dims(OUTPUT,1)

b1:set_broadcast(false, true)
b2:set_broadcast(false, true)
b3:set_broadcast(false, true)

-- ANN
function logistic(s) return 1/(1 + op.exp(-s)) end
local net_h1  = logistic(b1 + w1 * x)       -- first layer
local net_h2  = logistic(b2 + w2 * net_h1)  -- second layer
local net_out = b3 + w3 * net_h2            -- output layer
local net     = op.exp(net_out) / op.sum(op.exp(net_out)) -- softmax
-- Loss function: negative cross-entropy
local L = -op.sum( op.cmul(y, op.log(net)) )
-- Regularization
L = L + 0.5 * wd * (op.sum(w1^2) + op.sum(w2^2) + op.sum(w2^2))

-- Compilation
local shared_vars = table.deep_copy(weights)
shared_vars.seed  = 1
local f   = func(net, {x}, shared_vars)
local tbl = table.pack( L, AD.diff(L, {b1, w1, b2, w2, b3, w3}) )
local dL_dw,program = func(tbl, {x,y}, shared_vars)
local Lxy = func(L, {x,y}, shared_vars)

AD.dot_graph(tbl[6], "wop.dot")

io.open("program.lua","w"):write(program.."\n")

local opt = ann.optimizer.sgd()
opt:set_option("learning_rate", learning_rate)
opt:set_option("momentum", momentum)

for i=1,100 do
  local idx = rnd:shuffle(train_input:numPatterns())
  local tr_loss = stats.mean_var()
  for j=1,train_input:numPatterns() do
    local input  = M(INPUT,1, train_input:getPattern(idx[j]))
    local output = M(OUTPUT,1, train_output:getPattern(idx[j]))
    local loss = opt:execute(function()
			       local loss,db1,dw1,db2,dw2,db3,dw3 = dL_dw(input,output)
			       return loss, { b1=db1, w1=dw1,
					      b2=db2, w2=dw2,
					      b3=db3, w3=dw3 }
			     end,
			     weights)
    L:eval{ x=input, y=output, b1=weights.b1, w1=weights.w1, b2=weights.b2, w2=weights.w2, b3=weights.b3, w3=weights.w3, seed=seed }
    print(L.last)
    print( (op.cmul(y, op.log(net))).last )
    print( (op.cmul((1-y), op.log(1-net))).last )
    print(net_out.last)
    tr_loss:add(loss)
  end
  local va_loss = stats.mean_var()
  for j=1,val_input:numPatterns() do
    local input  = M(INPUT,1, val_input:getPattern(j))
    local output = M(OUTPUT,1, val_output:getPattern(j))
    local loss   = Lxy(input,output)
    va_loss:add(loss)
  end
  print(i, tr_loss:compute(), va_loss:compute())
end
