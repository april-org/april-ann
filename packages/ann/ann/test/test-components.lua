-- forces the use of CUDA
mathcore.set_use_cuda_default(util.is_cuda_available())
--

local check   = utest.check
local T       = utest.test
local verbose = false
local rnd     = random(1234)

utest.select_tests(arg)

function check_component(component_builder_func,loss_name,i,o,b,desc,norm)
  if verbose then
    fprintf(io.stderr, "\nGradients %s (%d,%d,%d,%s)\n",
            desc,i,o,b,loss_name)
  end
  ann.components.reset_id_counters()
  local c = component_builder_func()
  trainer = trainable.supervised_trainer(c, ann.loss[loss_name](), b)
  trainer:build()
  trainer:randomize_weights{ inf = -1, sup = 1, random = rnd }
  for _,a in trainer:iterate_weights("a.*") do a:fill(0.25) end
  input  = matrix(b, i):uniformf(-1,1,rnd)
  if loss_name == "mse" then
    target = matrix(b, o):uniformf(-1,1,rnd)
  elseif not norm and (loss_name == "batch_fmeasure_micro_avg" or loss_name == "batch_fmeasure_macro_avg") then
    target = matrix(b, o):uniform(0,1,rnd)
  else
    target = matrix(b, o):uniformf(0,1,rnd)
  end
  if norm then
    apply(function(m) m:exp() m:scal(1/m:sum()) end,
      target:sliding_window():iterate())
  end
  result = trainer:grad_check_step(input, target, verbose)
  if not result then
    print("---- WEIGHTS ----")
    for wname,w in pairs(trainer:get_weights_table()) do
      print(wname:upper())
      print(w)
    end
    print("---- INPUT ----")
    print(input)
    print("---- TARGET ----")
    print(target)
    for name,c in trainer:iterate_components() do
      print("---- " .. name .. " ----")
      print("Input matrix")
      print(c:get_input())
      print("Output matrix")
      print(c:get_output())
      print("Error input matrix")
      print(c:get_error_input())
      print("Error output matrix")
      print(c:get_error_output())
    end
    error(string.format("Error at %s (%d,%d,%d,%s) !!!",desc,i,o,b,loss_name))
  end
end

----------
-- BIAS --
----------
T("BIAS TEST",
  function()
    check(function()
        for o=2,4 do
          for b=1,4 do
            check_component(function()
                return ann.components.bias{ size=o }
                            end,
              "mse", o, o, b, "BIAS")
          end
        end
        return true
    end)
end)

------------------------
-- DOT PRODUCT + BIAS --
------------------------
T("DOTPRODUCT + BIAS TEST",
  function()
    check(function()
        for i=2,4 do
          for o=2,4 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i,
                                                      output=o } ):
                    push( ann.components.bias{ size=o } )
                              end,
                "mse", i, o, b, "DOTPRODUCT + BIAS")
            end
          end
        end
        return true
    end)
end)

-- DROPOUT

T("DOTPRODUCT + BIAS + DROPOUT TEST",
  function()
    check(function()
        for i=2,4 do
          for o=2,4 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.bias{ size=o } ):
                    push( ann.components.dropout() )
                              end,
                "mse", i, o, b, "DOTPRODUCT + BIAS + DROPOUT")
            end
          end
        end
        return true
    end)
end)

-----------------------------------
-- DOT PRODUCT + BIAS + FMEASURE --
-----------------------------------

T("DOTPRODUCT + BIAS + LOGISTIC + FM_MICRO_AVG TEST",
  function()
    check(function()
        for i=2,4 do
          for o=1,4 do
            b=32
            check_component(function()
                return ann.components.stack():
                  push( ann.components.dot_product{ input=i, output=o } ):
                  push( ann.components.bias{ size=o } ):
                  push( ann.components.actf.logistic() )
                            end,
              "batch_fmeasure_micro_avg", i, o, b, "DOTPRODUCT + BIAS + FM")
          end
        end
        return true
    end)
end)

T("DOTPRODUCT + BIAS + LOGISTIC + FM_MACRO_AVG TEST",
  function()
    check(function()
        for i=2,4 do
          for o=1,4 do
            b=32
            check_component(function()
                return ann.components.stack():
                  push( ann.components.dot_product{ input=i, output=o } ):
                  push( ann.components.bias{ size=o } ):
                  push( ann.components.actf.logistic() )
                            end,
              "batch_fmeasure_macro_avg", i, o, b, "DOTPRODUCT + BIAS + FM")
          end
        end
        return true
    end)
end)

T("DOTPRODUCT + BIAS + SOFTMAX + FM_MACRO_AVG TEST",
  function()
    check(function()
        for i=2,4 do
          for o=2,4 do
            b=32
            check_component(function()
                return ann.components.stack():
                  push( ann.components.dot_product{ input=i, output=o } ):
                  push( ann.components.bias{ size=o } ):
                  push( ann.components.actf.softmax() )
                            end,
              "batch_fmeasure_macro_avg", i, o, b, "DOTPRODUCT + BIAS + SOFTMAX + FM",
              true)
          end
        end
        return true
    end)
end)

--------------------------------
-- SLICE + DOT PRODUCT + BIAS --
--------------------------------

T("SLICE + DOTPRODUCT + BIAS TEST",
  function()
    check(function()
        for p=2,4 do
          for s=2,4 do
            for o=2,4 do
              for b=1,4 do
                check_component(function()
                    return ann.components.stack():
                      push( ann.components.rewrap{ size={12, 10} } ):
                      push( ann.components.slice{ pos={ p, p+1 },
                                                  size={s+1, s} }):
                      push( ann.components.rewrap{ size={(s+1)*s} } ):
                      push( ann.components.dot_product{ input=(s+1)*s,
                                                        output=o } ):
                      push( ann.components.bias{ size=o } )
                                end,
                  "mse", 120, o, b, "SLICE + DOTPRODUCT + BIAS")
              end
            end
          end
        end
        return true
    end)
end)

------------------
-- AUTO-ENCODER --
------------------

T("LOGISTIC AUTO-ENCODER TEST",
  function()
    check(function()
        for i=2,4 do
          for o=2,4 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ weights="w1",
                                                      input=i, output=o } ):
                    push( ann.components.bias{ size=o } ):
                    push( ann.components.actf.logistic() ):
                    push( ann.components.dot_product{ weights="w1",
                                                      input=o, output=i,
                                                      transpose=true } ):
                    push( ann.components.bias{ size=i } ):
                    push( ann.components.actf.logistic() )
                              end,
                "mse", i, i, b, "AUTO-ENCODER")
            end
          end
        end
        return true
    end)
end)

--------------------------------------
-- CONVOLUTION + ACTF + MAX POOLING --
--------------------------------------

T("REWRAP + CONVOLUTION + LOGISTIC + MAXPOOLING TEST + FLATTEN TEST",
  function()
    check(function()
        for n=1,4 do
          for b=1,4 do
            check_component(function()
                return ann.components.stack():
                  push( ann.components.rewrap{ size={1, 8, 10} } ):
                  push( ann.components.convolution{ kernel={1, 3, 5}, n=n } ):
                  push( ann.components.convolution_bias{ n=n, ndims=3 } ):
                  push( ann.components.actf.logistic() ):
                  push( ann.components.max_pooling{ kernel={n, 2, 3} } ):
                  push( ann.components.flatten() )
                            end,
              "mse", 80, 6, b, "CONVOLUTION "..n)
          end
        end
        return true
    end)
end)

-------------------------------
-- COPY + JOIN + DOT PRODUCT --
-------------------------------

T("COPY + JOIN + DOTPRODUCT TEST",
  function()
    check(function()
        for t=2,4 do
          for i=2,4 do
            for o=2,4 do
              for b=1,4 do
                check_component(function()
                    local j = ann.components.join()
                    for k=1,t do
                      j:add( ann.components.dot_product{ input=i,
                                                         output=o,
                                                         weights="w" } )
                    end
                    return ann.components.stack():
                      push( ann.components.copy{ input=i, times=t } ):
                      push( j )
                                end,
                  "mse", i, o*t, b, "COPY "..t)
              end
            end
          end
        end
        return true
    end)
end)

---------
-- LOG --
---------

T("DOTPRODUCT + LOG TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.log() )
                              end,
                "mse", i, o, b, "LOG")
            end
          end
        end
        return true
    end)
end)

---------
-- EXP --
---------

T("DOTPRODUCT + EXP TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.exp() )
                              end,
                "mse", i, o, b, "EXP")
            end
          end
        end
        return true
    end)
end)

--------------
-- LOGISTIC --
--------------

T("DOTPRODUCT + LOGISTIC TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.logistic() )
                              end,
                "mse", i, o, b, "LOGISTIC")
            end
          end
        end
        return true
    end)
end)

------------------
-- LOG_LOGISTIC --
------------------

T("DOTPRODUCT + LOG_LOGISTIC TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.log_logistic() )
                              end,
                "cross_entropy", i, o, b, "LOG_LOGISTIC")
            end
          end
        end
        return true
    end)
end)

T("DOTPRODUCT + LOGISTIC + CE TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.logistic() )
                              end,
                "non_paired_cross_entropy", i, o, b, "LOGISTIC_CE")
            end
          end
        end
        return true
    end)
end)

----------
-- TANH --
----------

T("DOTPRODUCT + TANH TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.tanh() )
                              end,
                "mse", i, o, b, "TANH")
            end
          end
        end
        return true
    end)
end)

---------------
-- HARD TANH --
---------------

T("DOTPRODUCT + HARDTANH TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,3 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.hardtanh() )
                              end,
                "mse", i, o, b, "HARDTANH")
            end
          end
        end
        return true
    end)
end)

--------------
-- SOFTPLUS --
--------------

T("DOTPRODUCT + SOFTPLUS TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.softplus() )
                              end,
                "mse", i, o, b, "SOFTPLUS")
            end
          end
        end
        return true
    end)
end)

----------
-- RELU --
----------

T("DOTPRODUCT + RELU TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.relu() )
                              end,
                "mse", i, o, b, "RELU")
            end
          end
        end
        return true
    end)
end)

----------------
-- LEAKY RELU --
----------------

T("DOTPRODUCT + LEAKY RELU TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,3 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.leaky_relu() )
                              end,
                "mse", i, o, b, "LEAKY_RELU")
            end
          end
        end
        return true
    end)
end)

-----------
-- PRELU --
-----------

T("DOTPRODUCT + PRELU NON SCALAR TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,3 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.prelu() )
                              end,
                "mse", i, o, b, "NON_SCALAR_PRELU")
            end
          end
        end
        return true
    end)
end)

T("DOTPRODUCT + PRELU SCALAR TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,3 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.prelu{ scalar=true } )
                              end,
                "mse", i, o, b, "SCALAR_PRELU")
            end
          end
        end
        return true
    end)
end)


---------
-- MUL --
---------

T("DOTPRODUCT + MUL NON SCALAR TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,3 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.mul() )
                              end,
                "mse", i, o, b, "NON_SCALAR_MUL")
            end
          end
        end
        return true
    end)
end)

T("DOTPRODUCT + MUL SCALAR TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,3 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.mul{ scalar=true } )
                              end,
                "mse", i, o, b, "SCALAR_MUL")
            end
          end
        end
        return true
    end)
end)

---------------
-- CONST_MUL --
---------------

T("DOTPRODUCT + CONST_MUL NON SCALAR TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,3 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.const_mul{ data=matrix(o):uniformf() } )
                              end,
                "mse", i, o, b, "NON_SCALAR_CONST_MUL")
            end
          end
        end
        return true
    end)
end)

T("DOTPRODUCT + CONST_MUL SCALAR TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=1,3 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.const_mul{ data=matrix(1):uniformf() } )
                              end,
                "mse", i, o, b, "SCALAR_CONST_MUL")
            end
          end
        end
        return true
    end)
end)

---------------------------
-- BATCH STANDARDIZATION --
---------------------------

T("BATCH STANDARDIZATION TEST",
  function()
    local rnd = random(1234)
    local N   = 32
    local SZ  = 100
    local MU  = 2.0
    local SIGMA = 0.1
    local dist = stats.dist.normal(MU, SIGMA*SIGMA)
    local a   = ann.components.batch_standardization{ size=SZ }:build()
    for _=1,8000 do
      local i = dist:sample(rnd,N*SZ):rewrap(N,SZ)
      a:reset()
      a:forward(i, true)
    end
    local params  = a:ctor_params()
    local mean    = stats.amean(params.mean)
    local inv_std = stats.amean(params.inv_std)
    check.number_eq(mean, MU)
    check.number_eq(inv_std, 1/SIGMA)
end)

T("DOTPRODUCT + BATCH STANDARDIZATION TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=16,20 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.batch_standardization{
                            mean = matrix(o):linspace(0,2.0),
                            inv_std = 1/matrix(o):linspace(0.1,2),
                    } )
                              end,
                "mse", i, o, b, "DOTPRODUCT_BATCH_STANDARDIZATION")
            end
          end
        end
        return true
    end)
end)

-------------------------
-- BATCH NORMALIZATION --
-------------------------

T("DOTPRODUCT + BATCH NORMALIZATION TEST",
  function()
    check(function()
        for i=1,4 do
          for o=1,4 do
            for b=12,14 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.batchnorm{
                            mean = matrix(o):linspace(0,2.0),
                            inv_std = 1/matrix(o):linspace(0.1,2),
                    } )
                              end,
                "mse", i, o, b, "DOTPRODUCT_BATCH_NORMALIZATION")
            end
          end
        end
        return true
    end)
end)

-------------
-- SOFTMAX --
-------------

T("DOTPRODUCT + SOFTMAX TEST",
  function()
    check(function()
        for i=1,4 do
          for o=3,6 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.softmax() )
                              end,
                "mse", i, o, b, "SOFTMAX", true)
            end
          end
        end
        return true
    end)
end)

-----------------
-- LOG_SOFTMAX --
-----------------

T("DOTPRODUCT + LOG_SOFTMAX TEST",
  function()
    check(function()
        for i=1,4 do
          for o=3,6 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.log_softmax() )
                              end,
                "multi_class_cross_entropy", i, o, b, "LOG_SOFTMAX",
                true)
            end
          end
        end
        return true
    end)
end)

T("DOTPRODUCT + SOFTMAX + CE TEST",
  function()
    check(function()
        for i=1,4 do
          for o=3,6 do
            for b=1,4 do
              check_component(function()
                  return ann.components.stack():
                    push( ann.components.dot_product{ input=i, output=o } ):
                    push( ann.components.actf.softmax() )
                              end,
                "non_paired_multi_class_cross_entropy", i, o, b, "SOFTMAX_CE",
                true)
            end
          end
        end
        return true
    end)
end)

-----------------
-- DOT PRODUCT --
-----------------
T("DOTPRODUCT TEST",
  function()
    check(
      function()
        for i=2,4 do
          for o=2,4 do
            for b=1,4 do
              check_component(
                function()
                  return ann.components.dot_product{ input=i, output=o }
                end,
                "mse", i, o, b, "DOTPRODUCT"
              )
            end
          end
        end
        return true
    end
    )
end
)

-----------------------
-- PROBMAT COMPONENT --
-----------------------
T("PROBMAT TEST",
  function()
    check(
      function()
        for _,side in ipairs{ "left", "right" } do
          for i=2,3 do
            for o=2,3 do
              for h=2,4 do
                for b=1,4 do
                  check_component(
                    function()
                      return ann.components.stack():
                        push( ann.components.dot_product{ input=i, output=h },
                              ann.components.actf.softmax(),
                              ann.components.probabilistic_matrix{ side=side,
                                                                   input=h,
                                                                   output=o } )
                    end,
                    "mse", i, o, b, "PROBMAT",
                    true)
                end
              end
            end
          end
        end
        return true
    end
    )
end
)

T("PROBMAT TEST 2",
  function()
    check(
      function()
        for i=2,3 do
          for o=2,3 do
            for h=2,4 do
              for b=1,4 do
                check_component(
                  function()
                    return ann.components.stack():
                      push( ann.components.dot_product{ input=i, output=h },
                            ann.components.actf.softmax(),
                            ann.components.probabilistic_matrix{ side="left",
                                                                 input=h,
                                                                 output=o } )
                  end,
                  "non_paired_multi_class_cross_entropy", i, o, b, "PROBMAT2",
                  true)
              end
            end
          end
        end
        return true
    end
    )
end
)

T("PROBMAT TEST 3",
  function()
    check(
      function()
        for i=2,3 do
          for o=2,3 do
            for h=2,4 do
              for b=1,4 do
                check_component(
                  function()
                    return ann.components.stack():
                      push( ann.components.dot_product{ input=i, output=h },
                            ann.components.actf.softmax(),
                            ann.components.probabilistic_matrix{ side="right",
                                                                 input=h,
                                                                 output=o } )
                  end,
                  "non_paired_cross_entropy", i, o, b, "PROBMAT3",
                  true)
              end
            end
          end
        end
        return true
    end
    )
end
)
