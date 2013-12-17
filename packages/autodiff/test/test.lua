local AD   = autodiff
local op   = AD.op
local func = AD.func
local a,w,b,t = AD.matrix('a w b t')

local rnd = random(1234)
local M   = matrix.col_major

weights = {
  w = M(3,4):uniformf(0,1,rnd),
  b = M(1,3):uniformf(0,1,rnd),
}

function sigmoid(s)
  return 1/(1 + op.exp(-s))
end

d = a * op.transpose(w) + b
c = a * op.transpose(w) -- sigmoid( d )
c = (c - t) * op.transpose(c - t)

print(c)

---------------------------------------------------

local input  = M(1,4):uniformf(0,1,rnd)
local target = M(1,3):uniformf(0,1,rnd)
local cache = {}
f = func(c, {a,t}, weights, cache)
print( f(input,target) )

---------------------------------------------------

aux = c:diff(w)
print(aux)

autodiff.dot_graph(aux, "wop.dot")

df_dw = func(aux, {a,t}, weights, cache)
print( df_dw(input,target) )

---------------------------------------------------

aux = c:diff(a)
print(aux)

df_db = func(aux, {a,t}, weights, cache)
print( df_db(input,target) )
