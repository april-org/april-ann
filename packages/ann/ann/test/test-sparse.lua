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
T("JoinAndSparseTest",
  function()
    local j = ann.components.join()
    j:add( ann.components.dot_product{ input=3, output=2, matrix=matrix(2,3):uniformf(-1,1,random(1234)) } )
    j:add( ann.components.dot_product{ input=3, output=2, matrix=matrix(2,3):uniformf(-1,1,random(5825)) } )
    j:build()
    local i1 = matrix.sparse.diag{1,2,3}
    local i2 = matrix.sparse.diag{4,5,6}
    local o = j:forward( tokens.vector.bunch{ i1, i2 } )
    local o2 = j:forward( matrix.sparse(matrix(3,6, {
                                                 1,0,0,4,0,0,
                                                 0,2,0,0,5,0,
                                                 0,0,3,0,0,6, })) )
    check.eq(o, o2)
    check.eq(i1 * j:copy_weights().w1:t(), o[{':','1:2'}])
    check.eq(i2 * j:copy_weights().w2:t(), o[{':','3:4'}])
end)
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
--
T("SparseLogistic",
  function()
    local mop  = matrix.op
    local rnd  = random(1234)
    local beta = 1.0
    local rho  = 0.3
    local dist = stats.dist.binomial(1, 0.6)
    local net  = ann.components.actf.sparse_logistic{ penalty=beta, sparsity=rho }
    local loss = ann.loss.mse()
    for N = 1,4 do -- mini-batch size (bunch size)
      for M = 1,16 do -- number of neurons
        local input  = matrix(N, M):uniformf(-3, 3, rnd)
        local output = net:forward(input)
        local target = matrix(N, M)
        dist:sample( rnd, target:rewrap( target:size(), 1 ) )
        local gradients = net:backprop( loss:gradient(output, target) )
        --
        local EPS = mathcore.limits.float.epsilon()*2
        function sparsity_penalty(output, rho)
          local hat_rho = output:sum(1):scal( 1/output:dim(1) )
          hat_rho:clamp(EPS, 1.0 - EPS)
          local l = rho*mop.log(rho/hat_rho) + (1-rho)*mop.log((1 - rho)/(1-hat_rho))
          return l:sum()
        end
        --
        function compute_loss(output, target)
          local l = loss:compute_loss(output, target)
          return l + beta*sparsity_penalty(output, rho)
        end
        --
        local REL = 0.05 -- relative error
        local E = 0.001  -- difference value
        for i=1,N do
          for j=1,M do
            local aux = input:get(i, j)
            input:set(i, j, aux + E)
            local lp = compute_loss( net:forward(input), target )
            input:set(i, j, aux - E)
            local ln = compute_loss( net:forward(input), target )
            input:set(i, j, aux)
            local hat_g = (lp - ln)/(2*E)
            local g = gradients:get(i, j) / N
            -- printf("%3d  %3d  %.4f  %.4f  %.4f  %.4f\n",
            --        i, j, hat_g, g,
            --        math.abs(hat_g - g)/(math.abs(g) + math.abs(hat_g)),
            --        output:get(i,j), target:get(i,j))
            if hat_g>2*E and g>2*E then
              check.number_eq(hat_g, g, REL, "%.4f  %.4f"%{ hat_g, g})
            end
          end
        end
      end
    end
end)
