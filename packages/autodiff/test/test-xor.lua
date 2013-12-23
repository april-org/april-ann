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
    1 1 1
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

--
net = ann.mlp.all_all.generate("2 inputs 2 logistic 1 logistic")
net:build{ weights=weights }
--


local AD = autodiff
local op = AD.op
local func = AD.func
local b1,w1,b2,w2,x,y = AD.matrix('b1 w1 b2 w2 x y')

b1:set_dims(2,1)
w1:set_dims(2,2)
b2:set_dims(1,1)
w2:set_dims(1,2)
x:set_dims(2,1)
y:set_dims(1,1)

-- XOR ANN
local xor = b2 + w2 * (b1 + w1 * x)
-- Loss function
local L = op.sum((xor - y)^2)
-- Regularization
-- L = L + 0.5*weight_decay/learning_rate * (op.sum(w1^2) + op.sum(w2^2))

-- Compilation
local f   = func(xor, {x}, weights)
local tbl = table.pack( L, AD.diff(L, {b1, w1, b2, w2}) )
local dL_dw,program = func(tbl, {x,y}, weights)

io.open("program.lua","w"):write(program.."\n")

local opt = ann.optimizer.sgd()
opt:set_option("learning_rate", learning_rate)
opt:set_option("momentum", momentum)

for j=1,4 do
  local input = M(2,1, ds_input:getPattern(j))
  print(table.concat(input:toTable(), " "),
	f(input):get(1,1),
	net:forward(input:transpose()):get_matrix():get(1,1))
end
print()

for i=1,30000 do
  local idx = rnd:shuffle(4)
  local m = stats.mean_var()
  for j=1,4 do
    local input  = M(2,1, ds_input:getPattern(idx[j]))
    local output = M(1,1, ds_output:getPattern(idx[j]))
    local loss = opt:execute(function()
			       local loss,b1,w1,b2,w2 = dL_dw(input,output)
			       print(b1)
			       print(w1)
			       print(b2)
			       print(w2)
			       return loss, { b1=b1, w1=w1, b2=b2, w2=w2 }
			     end,
			     weights)
    m:add(loss)
  end
  print(i, m:compute())
end

print(weights.b1)
print(weights.w1)
print(weights.b2)
print(weights.w2)
