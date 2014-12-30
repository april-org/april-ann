local check = utest.check
local T = utest.test

T("ComplexTest",
  function()
    local a = complex(2,-3)
    check.number_eq(a:real(),2)
    check.number_eq(a:img(),-3)
    local t = table.pack(a:plane())
    check.eq(#t, 2)
    check.number_eq(t[1],2)
    check.number_eq(t[2],-3)
    local t = table.pack(a:polar())
    check.eq(#t, 2)
    check.number_eq(t[1], math.sqrt(2*2 + 3*3))
    check.number_eq(t[2], -0.98279)
    check.number_eq(a:abs(), math.sqrt(2*2 + 3*3))
    check.number_eq(a:angle(), -0.98279)
    check.number_eq(a:sqrt(), 1.674149)
    --
    local b = complex("2-3i")
    check.eq(a,b)
    check.eq(a,a:clone())
    check.eq(a:clone():conj(), complex(2,3))
    check.eq(a:exp(), complex(-3.07493, -1.04274))
    check.eq(tostring(a),tostring(b))
    check.eq(a+b, complex(4,-6))
    check.eq(a-2*b, complex(-2,3))
    check.eq(2*a, complex(4,-6))
    check.eq(a/2, complex(1,-1.5))
    check.eq(complex(2,-3) * complex(-4,5),
             complex(2*-4 + 3*5, 2*5 + 3*4))
    check.eq(-a, complex(-2,3))
end)
