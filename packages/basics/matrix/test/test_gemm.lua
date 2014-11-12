local check=utest.check
local T=utest.test

T("GEMMTest", function()

    local t1 = { 1, 2, 3,
		 4, 5, 6 }
    local t2 = { 3, 4,
		 1, 7,
		 9, 6 }

    local t1_t2   = matrix(2,2,{ 32, 36,
				 71, 87 })
    local t2_t1   = matrix(3,3,{ 19, 26, 33,
				 29, 37, 45,
				 33, 48, 63 })
    local t1p_t2p = matrix(3,3,{ 19, 29, 33,
				 26, 37, 48,
				 33, 45, 63 })
    local t2p_t1p = matrix(2,2,{ 32, 71,
				 36, 87 })

    local A = matrix(2,3,t1)
    local B = matrix(3,2,t2)

    check.eq(A*B, t1_t2)
    check.eq(B*A, t2_t1)
    check.eq(A:t()*B:t(), t1p_t2p)
    check.eq(B:t()*A:t(), t2p_t1p)

    check.eq(matrix(2,2):gemm{ A=A, B=B, alpha=1, beta=0 }, t1_t2)
    check.eq(matrix(3,3):gemm{ A=B, B=A, alpha=1, beta=0 }, t2_t1)
    check.eq(matrix(3,3):gemm{ A=A:t(), B=B:t(), alpha=1, beta=0 }, t1p_t2p)
    check.eq(matrix(2,2):gemm{ A=B:t(), B=A:t(), alpha=1, beta=0 }, t2p_t1p)

    check.eq(matrix(2,2):gemm{ A=A:t(), B=B:t(), trans_A=true, trans_B=true, alpha=1, beta=0 }, t1_t2)
    check.eq(matrix(3,3):gemm{ A=B:t(), B=A:t(), trans_A=true, trans_B=true, alpha=1, beta=0 }, t2_t1)
    check.eq(matrix(3,3):gemm{ A=A, B=B, trans_A=true, trans_B=true, alpha=1, beta=0 }, t1p_t2p)
    check.eq(matrix(2,2):gemm{ A=B, B=A, trans_A=true, trans_B=true, alpha=1, beta=0 }, t2p_t1p)

    check.eq(matrix(2,2):transpose():gemm{ A=A, B=B, alpha=1, beta=0 }, t1_t2)
    check.eq(matrix(3,3):transpose():gemm{ A=B, B=A, alpha=1, beta=0 }, t2_t1)
    check.eq(matrix(3,3):transpose():gemm{ A=A:t(), B=B:t(), alpha=1, beta=0 }, t1p_t2p)
    check.eq(matrix(2,2):transpose():gemm{ A=B:t(), B=A:t(), alpha=1, beta=0 }, t2p_t1p)

    check.eq(matrix(2,2):transpose():gemm{ A=A:t(), B=B:t(), trans_A=true, trans_B=true, alpha=1, beta=0 }, t1_t2)
    check.eq(matrix(3,3):transpose():gemm{ A=B:t(), B=A:t(), trans_A=true, trans_B=true, alpha=1, beta=0 }, t2_t1)
    check.eq(matrix(3,3):transpose():gemm{ A=A, B=B, trans_A=true, trans_B=true, alpha=1, beta=0 }, t1p_t2p)
    check.eq(matrix(2,2):transpose():gemm{ A=B, B=A, trans_A=true, trans_B=true, alpha=1, beta=0 }, t2p_t1p)
    
    check.errored(function() return A*A end)
    check.errored(function() return B*B end)
    
end)
