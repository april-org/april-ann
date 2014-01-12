local learning_rate  = 0.4
local momentum       = 0.1
local weight_decay   = 1e-05
local semilla        = 1234
local rnd            = random(semilla)
--
local M = matrix.col_major

m_xor = matrix.fromString[[
    4 3
    ascii
    0 0 0
    0 1 1
    1 0 1
    1 1 0
]]

ds_input  = dataset.matrix(m_xor,{patternSize={1,2}})
ds_output = dataset.matrix(m_xor,{offset={0,2},patternSize={1,1}})

m = matrix.fromString[[
    3 3
    ascii col_major
      -0.5 -1.2 1.0
      -2.0 4.0 -4.0
      -1.0 2.0 2.0
]]

local weights = {
  b1 = M(2,1):copy(m("1:2",1)),
  w1 = M(2,2):copy(m("1:2","2:3")),
  b2 = M(1,1):copy(m(3,1)),
  w2 = M(1,2):copy(m(3,"2:3")),
}

local AD = autodiff
local op = AD.op
local func = AD.func
local b1,w1,b2,w2,x,y = AD.matrix('b1 w1 b2 w2 x y')
-- it is possible to use AD.scalar('wd') if you want to change weight decay
-- during learning
local wd = weight_decay
local lr = learning_rate

b1:set_dims(2,1)
w1:set_dims(2,2)
b2:set_dims(1,1)
w2:set_dims(1,2)
x:set_dims(2,4)
y:set_dims(1,4)

b1:set_broadcast(false, true)
b2:set_broadcast(false, true)

-- XOR ANN
function logistic(s) return 1/(1 + op.exp(-s)) end
local xor = logistic(b2 + w2 * logistic(b1 + w1 * x))
-- Loss function: negative cross-entropy
local L = -op.sum(op.cmul(y,op.log(xor)) + op.cmul((1-y),op.log(1-xor)))
-- Regularization
L = L + 0.5 * wd * (op.sum(w1^2) + op.sum(w2^2))

-- Compilation
local shared_vars = weights
local f   = func(xor, {x}, shared_vars)
local tbl = table.pack( L, AD.diff(L, {b1, w1, b2, w2}) )
local dL_dw = func(tbl, {x,y}, shared_vars)
local program = dL_dw.program

AD.dot_graph(tbl[4], "wop.dot")

io.open("program.lua","w"):write(program.."\n")

local opt = ann.optimizer.sgd()
opt:set_option("learning_rate", learning_rate)
opt:set_option("momentum", momentum)

for j=1,4 do
  local input = M(2,1, ds_input:getPattern(j))
  print(table.concat(input:toTable(), " "), f(input):get(1,1))
end
print()

for i=1,30000 do
  local input  = M(2,4)
  local output = M(1,4)
  for j=1,4 do
    input(":",j):copy_from_table(ds_input:getPattern(j))
    output(":",j):copy_from_table(ds_output:getPattern(j))
  end
  local loss = opt:execute(function()
			     local loss,db1,dw1,db2,dw2 = dL_dw(input,output)
			     return loss, { b1=db1, w1=dw1, b2=db2, w2=dw2 }
			   end,
			   weights)
  print(i, loss)
end

print(weights.b1)
print(weights.w1)
print(weights.b2)
print(weights.w2)

for j=1,4 do
  local input = M(2,1, ds_input:getPattern(j))
  print(table.concat(input:toTable(), " "), f(input):get(1,1))
end
print()
