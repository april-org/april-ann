local op   = AD.op
local func = AD.func
local a,b  = AD.scalar('a b')

c = op.blah( op.exp( op.log( a ^ b ) * 2 ) )

c = func(c, a, b)
print( c(3,2) )
