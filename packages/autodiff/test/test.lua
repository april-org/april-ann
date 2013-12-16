local AD   = autodiff
local op   = AD.op
local func = AD.func
local a,b  = AD.matrix('a b')

weights = {
  b = matrix.col_major(3,4):linear()
}

c = a * op.transpose(b)

---------------------------------------------------

f = func(c, {a}, weights)
print( f(matrix.col_major(2,4):linear(4)) )

---------------------------------------------------

df_db = func( c:diff(b), {a}, weights )
print( df_db(matrix.col_major(2,4):linear(4)) )
