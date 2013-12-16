local AD   = autodiff
local op   = AD.op
local func = AD.func
local a,b  = AD.scalar('a b')

c = a * b

cf = func(c, {a}, {b=2})
print( cf(3) )

dc_db = c:diff(a)

print(dc_db)

f = func(dc_db, {a,b})

print(f(4,5))
