local AD   = autodiff
local op   = AD.op
local func = AD.func
local a,b,c  = AD.scalar('a b c')

f = op.cos(a*b + c)
print(f)
fx = func(f, {a,b,c})
print(fx(3,4,5))

------------------------------------------------

df = f:diff(AD.constant(1))
print(df.a)
AD.dot_graph(df.a, "wop.dot")

df_da = func(df.a, {a,b,c})
print(df_da(3,4,5))
