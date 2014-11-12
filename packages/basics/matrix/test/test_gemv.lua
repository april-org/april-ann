local check=utest.check
local T=utest.test

T("GEMVTest", function()

    local t1 = { 1, 2, 3,
		 4, 5, 6 }
    local t2 = { 3, 1, 9 }
    local t3 = { 9, 7 }

    local t1_t2  = matrix(2,{ 32, 71 })
    local t1p_t3 = matrix(3,{ 37, 53, 69 })

    local A = matrix(2,3,t1)
    local B = matrix(t2)
    local C = matrix(t3)

    check.eq(matrix(2):gemv{ A=A, X=B, alpha=1, beta=0 }, t1_t2)
    check.eq(matrix(3):gemv{ A=A:t(), X=C, alpha=1, beta=0 }, t1p_t3)
    
    check.eq(matrix(2):gemv{ A=A:t(), X=B, trans_A=true, alpha=1, beta=0 }, t1_t2)
    check.eq(matrix(3):gemv{ A=A, X=C, trans_A=true, alpha=1, beta=0 }, t1p_t3)
    
    check.errored(function() return matrix(3):gemv{ A=A:t(), X=C, trans_A=true, alpha=1, beta=0 } end)
    check.errored(function() return matrix(3):gemv{ A=A, X=C, alpha=1, beta=0 } end)
end)
