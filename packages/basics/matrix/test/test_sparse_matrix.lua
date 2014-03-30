local blockf = mathcore.block.float
local blocki = mathcore.block.int32
--
local function dense(m) return (isa(m,matrix.sparse) and m:to_dense()) or m end
local eq = function(m1,m2)
  return function()
    local m1,m2=dense(m1),dense(m2)
    return m1:equals(m2)
  end
end
--
local check = utest.check
--
local a_dense = matrix(4,5,{
                         1,  0, 0, -1,  2,
                         0,  0, 2,  1,  0,
                         0,  0, 0,  1, -2,
                         -4, 0, 2,  0,  0,
                           })
local a_sparse_csr = matrix.sparse.csr(4,5,
                                       blockf{1,-1,2,2,1,1,-2,-4,2},
                                       blocki{0, 3,4,2,3,3, 4, 0,2},
                                       blocki{0,3,5,7,9})
local a_sparse_csc = matrix.sparse.csc(4,5,
                                       blockf{1,-4,2,2,-1,1,1,2,-2},
                                       blocki{0, 3,1,3, 0,1,2,0, 2},
                                       blocki{0,2,2,4,7,9})
check(eq(a_sparse_csr, a_dense))
check(eq(a_sparse_csc, a_dense))

local a_sparse_csr = matrix.sparse.csr(a_dense)
local a_sparse_csc = matrix.sparse.csc(a_dense)

check(eq(a_sparse_csr, a_dense))
check(eq(a_sparse_csc, a_dense))

local aux2 = matrix.sparse.diag(matrix(5,{1,2,3,4,5}),"csc")
local aux3 = matrix.sparse.diag(blockf({1,2,3,4,5}),"csc")
check(eq(aux2, aux3))

local aux2 = matrix.sparse.diag(matrix(5,{1,2,3,4,5}),"csr")
local aux3 = matrix.sparse.diag(blockf({1,2,3,4,5}),"csr")
check(eq(aux2, aux3))

check(function() return a_sparse_csr:max() == 2 end)
check(function() return a_sparse_csc:max() == 2 end)

local m = matrix(9,1):linspace()
local m2 = matrix.sparse.csc(3,3,
			     blockf{10,-4},
			     blocki{1,0},
			     blocki{0,1,2,2})
local aux = m2:as_vector()
check(eq(m:clone():axpy(1.0,aux:to_dense()),m:clone():axpy(1.0,aux)))

local b = matrix.sparse.csr(3,blockf{3},blocki{2})
local str = aux3:toString("ascii")
check(function() return str == [[5 5 5
ascii csr
1 2 3 4 5 
0 1 2 3 4 
0 1 2 3 4 5 
]]
      end)

local str = aux3:transpose():toString("ascii")
assert(str == [[5 5 5
ascii csc
1 2 3 4 5 
0 1 2 3 4 
0 1 2 3 4 5 
]])

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
local c = matrix(3,2):zeros():sparse_mm({
                                          trans_A=false,
                                          alpha=1.0,
                                          A=a_csc,
                                          B=b,
                                          beta=0.0,
                                        })
assert( (matrix(3,4,{
                  1,  2,  3, 0,
                  0,  0,  2, 0,
                  0, -1,  0, 1,
                    }) * b):equals(c) )
local c = matrix(3,2):zeros():sparse_mm({
                                          trans_A=false,
                                          alpha=1.0,
                                          A=a_csr,
                                          B=b,
                                          beta=0.0,
                                        })
assert( (matrix(3,4,{
                  1,  2,  3, 0,
                  0,  0,  2, 0,
                  0, -1,  0, 1,
                    }) * b):equals(c) )

local c = matrix(3,2):zeros():sparse_mm({
                                          trans_A=true,
                                          alpha=1.0,
                                          A=a_csc:transpose(),
                                          B=b,
                                          beta=0.0,
                                        })
assert( (matrix(3,4,{
                  1,  2,  3, 0,
                  0,  0,  2, 0,
                  0, -1,  0, 1,
                    }) * b):equals(c) )

local c = matrix(3,2):zeros():sparse_mm({
                                          trans_A=true,
                                          alpha=1.0,
                                          A=a_csr:transpose(),
                                          B=b,
                                          beta=0.0,
                                        })
assert( (matrix(3,4,{
                  1,  2,  3, 0,
                  0,  0,  2, 0,
                  0, -1,  0, 1,
                    }) * b):equals(c) )

local x = matrix(4):linear()
local y = matrix(3):zeros()

y:gemv({
         trans_A=false,
         alpha=1.0,
         A=a_csc,
         X=x,
         beta=0.0
       })

assert( (a_csc:to_dense()*x):equals(y) )

y:gemv({
         trans_A=true,
         alpha=1.0,
         A=a_csc:transpose(),
         X=x,
         beta=0.0
       })

print(a_csr:to_dense()*x)
print(y)

assert( (a_csc:to_dense()*x):equals(y) )

print(a_csc:to_dense()*x)
print(a_csr)
print(a_csc)

y:gemv({
         trans_A=false,
         alpha=1.0,
         A=a_csr,
         X=x,
         beta=0.0
       })

assert( (a_csc:to_dense()*x):equals(y) )

y:gemv({
         trans_A=true,
         alpha=1.0,
         A=a_csr:transpose(),
         X=x,
         beta=0.0
       })

assert( (a_csc:to_dense()*x):equals(y) )

