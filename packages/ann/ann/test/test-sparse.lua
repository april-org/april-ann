-- forces the use of CUDA
local use_cuda = false -- util.is_cuda_available()
mathcore.set_use_cuda_default(use_cuda)
--

local check = utest.check
local T = utest.test
local w = matrix(4,3):uniformf(0,1,random(1234))
local input = matrix(5,3,{ 0, 1, 0,
                           1, 1, 1,
                           1, 0, 1,
                           1, 0, 0,
                           0, 0, 1 })
local sparse_input = matrix.sparse.csr(input)
local e = matrix(5,4):uniformf(0,1,random(2384))
--
T("SparseDotProductTest",
  function()
    for _,aux in ipairs({ {w,false},
        {w:transpose():clone(),true}
    }) do
      local w,transpose = table.unpack(aux)
      if not use_cuda or transpose then
        local c = ann.components.dot_product{
          input = 3,
          output = 4,
          weights = "w",
          transpose = transpose,
        }:build{ weights={ w=w } }
        --
        local output = c:forward(input)
        c:backprop(e)
        local grads1 = c:compute_gradients()
        --
        local sparse_output = c:forward(sparse_input)
        c:backprop(e)
        local grads2 = c:compute_gradients()
        --
        check.eq(output,sparse_output)
        check.eq(grads1.w, grads2.w)
      end
    end
end)
