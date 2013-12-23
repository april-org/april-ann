local AD   = autodiff
local op   = AD.op
local func = AD.func
local a,w,w2 = AD.matrix('a w w2')
local b,c = AD.matrix('b c')

local rnd = random(1234)
local M   = matrix.col_major

weights = {
  w  = M(3,4):uniformf(0,1,rnd),
  b  = M(3,1):uniformf(0,1,rnd),
  w2 = M(2,3):uniformf(0,1,rnd),
  c  = M(2,1):uniformf(0,1,rnd),
}

-- if you give the dimensions, during composition of the symbolic expression,
-- the shape of the matrices will be checked
w:set_dims(weights.w:dim())
b:set_dims(weights.b:dim())
w2:set_dims(weights.w2:dim())
c:set_dims(weights.c:dim())
a:set_dims(4,1)

-- a function which returns a symbolic graph with logistic function
function logistic(s)
  return 1/(1 + op.exp(-s))
end

-- the desired equation (neural network with two layers)
f = logistic(w2 * logistic( w * a + b ) + c)

AD.dot_graph(f, "wop.dot")

-- a table with the forward and gradients for the weights
df_dw_tbl = table.pack( f, AD.diff(f, {w, b, w2, c}) )

AD.dot_graph(df_dw_tbl[2], "wop2.dot")

-- compilation of a Lua function which receives an input (a) and has a shared
-- table with the values of the weights; the function returns multiple values,
-- one for the forward, and one for each gradient computation
df_dw,program = AD.func(df_dw_tbl, {a}, weights )

-- write the compiled program to a diskfile
io.open("program.lua","w"):write(program.."\n")

result = table.pack(  df_dw( M(4,1):uniformf(0,1,rnd) ) )
iterator(ipairs(result)):select(2):apply(print)

---------------------------------------------------------------------

-- SYMBOLIC DECLARATION
AD.clear()
-- inputs
local x,s,h = AD.matrix('x s h')
-- target
local target = AD.matrix('target')
-- weights
local wx,ws1,wh1,ws2,wh2,b = AD.matrix('wx ws1 wh1 ws2 wh2 b')
-- gradient seed
local seed = AD.matrix('seed')
-- equation
f = wx*x + ws1*s + wh1*h + (ws2*s + wh2*h) * op.get(x,1,1) + b

-- loss
L = autodiff.op.sum( (f - target)^2 )

-- INSTANTIATION
local rnd = random(1234)
local M   = matrix.col_major

weights = {
  wx  = M(12,3):uniformf(-0.1, 0.1, rnd),
  ws1 = M(12,4):uniformf(-0.1, 0.1, rnd),
  wh1 = M(12,24):uniformf(-0.1, 0.1, rnd),
  ws2 = M(12,4):uniformf(-0.1, 0.1, rnd),
  wh2 = M(12,24):uniformf(-0.1, 0.1, rnd),
  b   = M(12,1):uniformf(-0.1, 0.1, rnd)
}

AD.dot_graph(f, "wop.dot")

df_dw_tbl = table.pack( f, AD.diff(f, {wx, ws1, wh1, ws2, wh2, b}, seed) )

AD.dot_graph(df_dw_tbl[5], "wop2.dot")

df_dw,program = AD.func(df_dw_tbl, {x,s,h,seed}, weights )
io.open("program2.lua", "w"):write(program .. "\n")

result = table.pack(  df_dw( M(3,1):uniformf(0,1,rnd),
			     M(4,1):uniformf(0,1,rnd),
			     M(24,1):zeros():set(10,1,1),
			     M(12,1):uniformf(0,1,rnd) ) )
iterator(ipairs(result)):select(2):apply(print)
