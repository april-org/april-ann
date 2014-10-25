-- forces the use of CUDA
mathcore.set_use_cuda_default(util.is_cuda_available())
--

local check   = utest.check
local T       = utest.test
local verbose = false
local rnd     = random(1234)

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
      print(c:get_input():get_matrix())
      print("Output matrix")
      print(c:get_output():get_matrix())
      print("Error input matrix")
      print(c:get_error_input():get_matrix())
      print("Error output matrix")
      print(c:get_error_output():get_matrix())
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
          for o=1,4 do
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
