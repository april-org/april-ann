local AD   = autodiff
local op   = AD.op
local func = AD.func
local a,w,b,w2  = AD.matrix('a w b w2')

local rnd = random(1234)
local M   = matrix.col_major

weights = {
  w  = M(4,3):uniformf(0,1,rnd),
  b  = M(3,1):uniformf(0,1,rnd),
  w2 = M(3,2):uniformf(0,1,rnd),
}

function sigmoid(s)
  return 1/(1 + op.exp(-s))
end

-- c = sigmoid( op.transpose(w) * a + b )
c = op.transpose(w2) * ( op.transpose(w) * a + b )
-- c = op.transpose(w) * a + b

---------------------------------------------------

local input = M(4,1):uniformf(0,1,rnd)
local cache = {}
f = func(c, {a}, weights, cache)
print( f(input) )

---------------------------------------------------

aux = c:diff(w2)
print(aux)

autodiff.dot_graph(aux, "wop2.dot")

df_dw2 = func(aux, {a}, weights, cache)
print( df_dw2( input ) )

---------------------------------------------------

aux = c:diff(w)
print(aux)

autodiff.dot_graph(aux, "wop.dot")

df_dw = func(aux, {a}, weights, cache)
print( df_dw( input ) )

---------------------------------------------------

aux = c:diff(b)
print(aux)

df_db = func(aux, {a}, weights, cache)
print( df_db(input) )
