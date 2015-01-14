local blockf = mathcore.block.float
local blocki = mathcore.block.int32
--
local function dense(m) return (class.is_a(m,matrix.sparse) and m:to_dense()) or m end
local make_eq = function(m1,m2)
  return function()
    local m1,m2=dense(m1),dense(m2)
    return m1:equals(m2)
  end
end
--
local check = utest.check
local T = utest.test
--
local a_dense = matrix(4,5,{
                         1,  0, 0, -1,  2,
                         0,  0, 2,  1,  0,
                         0,  0, 0,  1, -2,
                           -4, 0, 2,  0,  0,
})

local a_sparse_csr,a_sparse_csc
T("SparseConstructorsTest",
  function()
    a_sparse_csr = matrix.sparse.csr(4,5,
                                     blockf{1,-1,2,2,1,1,-2,-4,2},
                                     blocki{0, 3,4,2,3,3, 4, 0,2},
                                     blocki{0,3,5,7,9})
    a_sparse_csc = matrix.sparse.csc(4,5,
                                     blockf{1,-4,2,2,-1,1,1,2,-2},
                                     blocki{0, 3,1,3, 0,1,2,0, 2},
                                     blocki{0,2,2,4,7,9})
    check(make_eq(a_sparse_csr, a_dense), "CSR constructor from blocks")
    check(make_eq(a_sparse_csc, a_dense), "CSC constructor from blocks")
    
    a_sparse_csr = matrix.sparse.csr(a_dense)
    a_sparse_csc = matrix.sparse.csc(a_dense)
    
    check(make_eq(a_sparse_csr, a_dense), "CSR constructor from dense")
    check(make_eq(a_sparse_csc, a_dense), "CSC constructor from dense")

    local aux2 = matrix.sparse.diag(matrix(5,{1,2,3,4,5}),"csc")
    local aux3 = matrix.sparse.diag(blockf({1,2,3,4,5}),"csc")
    local aux4 = matrix.sparse.diag({1,2,3,4,5},"csc")
    check(make_eq(aux2, aux3), "diag CSC constructors 1")
    check(make_eq(aux2, aux4), "diag CSC constructors 2")

    local aux2 = matrix.sparse.diag(matrix(5,{1,2,3,4,5}),"csr")
    local aux3 = matrix.sparse.diag(blockf({1,2,3,4,5}),"csr")
    local aux4 = matrix.sparse.diag({1,2,3,4,5},"csr")
    check(make_eq(aux2, aux3), "diag CSR constructors 1")
    check(make_eq(aux2, aux4), "diag CSR constructors 2")
end)

T("SparseMaxTest",
  function()
    check(function() return a_sparse_csr:max() == 2 end, "CSR max")
    check(function() return a_sparse_csc:max() == 2 end, "CSC max")
end)

T("SparseAXPYTest",
  function()
    local m = matrix(9,1):linspace()
    local m2 = matrix.sparse.csc(3,3,
                                 blockf{10,-4},
                                 blocki{1,0},
                                 blocki{0,1,2,2})
    local aux = m2:as_vector()
    check(function()
        return make_eq(m:clone():axpy(1.0,aux:to_dense()),
                       m:clone():axpy(1.0,aux))()
    end)
end)

T("SparseSerializationTest",
  function()
    local b = matrix.sparse.csr(3,blockf{3},blocki{2})
    local aux3 = matrix.sparse.diag(blockf({1,2,3,4,5}),"csr")
    local str = aux3:toString("ascii")
    check.eq(str, [[5 5 5
ascii csr
1 2 3 4 5 
0 1 2 3 4 
0 1 2 3 4 5 
]],
    "CSR serialization")
    
    local str = aux3:transpose():toString("ascii")
    check.eq(str, [[5 5 5
ascii csc
1 2 3 4 5 
0 1 2 3 4 
0 1 2 3 4 5 
]],
    "CSR transpose + CSC serialization")
end)

local a_csc = matrix.sparse.csc(matrix(3,4,{
                                         1,  2,  3, 0,
                                         0,  0,  2, 0,
                                         0, -1,  0, 1,
}))
local a_csr = matrix.sparse.csr(matrix(3,4,{
                                         1,  2,  3, 0,
                                         0,  0,  2, 0,
                                         0, -1,  0, 1,
}))
local b = matrix(4,2,{
                   2, 1,
                   1, 0,
                     -1, -2,
                   1, 2,
})

T("SparseMMTesT",
  function()
    check(function()
        local c = matrix(3,2):zeros():sparse_mm({
            trans_A=false,
            alpha=1.0,
            A=a_csc,
            B=b,
            beta=0.0,
                                               })
        return make_eq(matrix(3,4,{
                                1,  2,  3, 0,
                                0,  0,  2, 0,
                                0, -1,  0, 1,
                             }) * b,
                       c)()
    end,
    "CSC sparse_mm")
    
    check(function()
        local c = matrix(3,2):zeros():sparse_mm({
            trans_A=false,
            alpha=1.0,
            A=a_csr,
            B=b,
            beta=0.0,
                                               })
        return make_eq(matrix(3,4,{
                                1,  2,  3, 0,
                                0,  0,  2, 0,
                                0, -1,  0, 1,
                             }) * b,
                       c)()
    end,
    "CSR sparse_mm")
    
    check(function()
        local c = matrix(3,2):zeros():sparse_mm({
            trans_A=true,
            alpha=1.0,
            A=a_csc:transpose(),
            B=b,
            beta=0.0,
                                               })
        return make_eq(matrix(3,4,{
                                1,  2,  3, 0,
                                0,  0,  2, 0,
                                0, -1,  0, 1,
                             }) * b,
                       c)()
    end,
    "CSC + transpose sparse_mm")
    
    check(function()
        local c = matrix(3,2):zeros():sparse_mm({
            trans_A=true,
            alpha=1.0,
            A=a_csr:transpose(),
            B=b,
            beta=0.0,
                                               })
        return make_eq(matrix(3,4,{
                                1,  2,  3, 0,
                                0,  0,  2, 0,
                                0, -1,  0, 1,
                             }) * b,
                       c)()
    end,
    "CSR + transpose sparse_mm")
end)

local x = matrix(4):linear()
local y = matrix(3):zeros()

T("SparseGEMVTest",
  function()
    check(function()
        y:gemv({
            trans_A=false,
            alpha=1.0,
            A=a_csc,
            X=x,
            beta=0.0
        })
        return make_eq(a_csc:to_dense()*x, y)()
    end,
    "CSC GEMV")

    check(function()
        y:gemv({
            trans_A=true,
            alpha=1.0,
            A=a_csc:transpose(),
            X=x,
            beta=0.0
        })
        return make_eq(a_csc:to_dense()*x, y)()
    end,
    "CSC transpose + CSR GEMV")
    
    check(function()
        y:gemv({
            trans_A=false,
            alpha=1.0,
            A=a_csr,
            X=x,
            beta=0.0
        })
        return make_eq(a_csc:to_dense()*x, y)()
          end,
      "CSR GEMV")
    
    check(function()
        y:gemv({
            trans_A=true,
            alpha=1.0,
            A=a_csr:transpose(),
            X=x,
            beta=0.0
        })
        return make_eq(a_csc:to_dense()*x, y)()
          end,
      "CSR + transpose + CSC GEMV")
end)

----------------------------------------------------------------------------

T("SparseDotTest",
  function()
    local x = matrix(10,1):uniformf(-10,10,random(1234))
    local y = matrix(1,10,{1,-1,4,0,0,2,0,0,0,-3})
    local z = x:dot(y)
    local y_csr = matrix.sparse.csr(y)
    local y_csc = y:transpose()
    
    -- FIXME: this test is failing in travis :S
    -- check.eq(x:dot(y_csr),z)
    check.eq(x:dot(y_csc),z, "csc dot")
end)

T("SparseDOKBuilder",
  function()
    local d = matrix.sparse.builders.dok()
    
    d:set(1, 1,  1.0)
    d:set(2, 1,  1.0)
    d:set(1, 3, -1.0)
    d:set(3, 2, -1.0)
    
    check.eq(d:build(), matrix.sparse(matrix(3,3,{
                                               1.0, 0.0, -1.0,
                                               1.0, 0.0, 0.0,
                                               0.0, -1.0, 0.0,
    })))
    
    check.eq(d:build(4,4), matrix.sparse(matrix(4,4,{
                                                  1.0, 0.0, -1.0, 0.0,
                                                  1.0, 0.0, 0.0, 0.0,
                                                  0.0, -1.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0, 0.0,
    })))
    
    check.eq(d:build(3,4), matrix.sparse(matrix(3,4,{
                                                  1.0, 0.0, -1.0, 0.0,
                                                  1.0, 0.0, 0.0, 0.0,
                                                  0.0, -1.0, 0.0, 0.0,
    })))
    
    check.eq(d:build(4,3), matrix.sparse(matrix(4,3,{
                                                  1.0, 0.0, -1.0,
                                                  1.0, 0.0, 0.0,
                                                  0.0, -1.0, 0.0,
                                                  0.0, 0.0, 0.0,
    })))

    check.errored(function() d:build(2,3) end)
    check.errored(function() d:build(3,2) end)
    
    check.errored(function() d:set(0, 2, 1.0) end)
    check.errored(function() d:set(2, 0, 1.0) end)
    
    local d = matrix.sparse.builders.dok()
    
    d:set(2, 1,  4)
    d:set(2, 2,  0)
    d:set(3, 5, -1)
    d:set(3, 5,  0) -- remove previous value
    check.eq(d:build(),
             matrix.sparse(matrix(3,5,{
                                    0, 0, 0, 0, 0,
                                    4, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, })))
end)
