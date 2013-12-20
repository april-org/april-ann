local AD   = autodiff
local op   = AD.op
local func = AD.func
local a,w,b,c,w2,seed  = AD.matrix('a w b c w2 seed')

local rnd = random(1234)
local M   = matrix.col_major

weights = {
  w  = M(4,3):uniformf(0,1,rnd),
  b  = M(3,1):uniformf(0,1,rnd),
  w2 = M(3,2):uniformf(0,1,rnd),
  c  = M(2,1):uniformf(0,1,rnd),
}

function logistic(s)
  return 1/(1 + op.exp(-s))
end

f         = logistic(op.transpose(w2) * logistic( op.transpose(w) * a + b ) + c)
df_dw_tbl = table.pack( f, AD.diff(f, {w, b, w2, c}) )
df_dw     = AD.func(df_dw_tbl, {a}, weights )

---------------------------------------------------

local input = M(4,1):uniformf(0,1,rnd)
df_dw_result = table.pack( df_dw(input) )

for i,v in ipairs(df_dw_result) do print(v) end

---------------------------------------------------

net = ann.mlp.all_all.generate("4 inputs 3 logistic 2 logistic")
net:build{
  weights = {
    w1 = weights.w:transpose(),
    w2 = weights.w2:transpose(),
    b1 = weights.b(),
    b2 = weights.c(),
  }
}

print( net:forward(input:transpose()):get_matrix() )

net:backprop( M(1,2):ones() )
grads = net:compute_gradients()
for i,v in pairs(grads) do
  print(i)
  if i:match("w.") then v=v:transpose() end
  print(v)
end

