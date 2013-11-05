verbose = false
rnd     = random(1234)

function check_component(component_builder_func,loss_name,i,o,b,desc,norm)
  if verbose then
    fprintf(io.stderr, "\nGradients %s (%d,%d,%d,%s)\n",
	    desc,i,o,b,loss_name)
  end
  ann.components.reset_id_counters()
  local c = component_builder_func()
  if util.is_cuda_available() then c:set_use_cuda(true) end
  trainer = trainable.supervised_trainer(c, ann.loss[loss_name](), b)
  trainer:build()
  trainer:randomize_weights{ inf = -1, sup = 1, random = rnd }
  input  = matrix.col_major(b, i):uniformf(-1,1,rnd)
  if loss_name == "mse" then
    target = matrix.col_major(b, o):uniformf(-1,1,rnd)
  elseif not norm and (loss_name == "batch_fmeasure_micro_avg" or loss_name == "batch_fmeasure_macro_avg") then
    target = matrix.col_major(b, o):uniform(0,1,rnd)
  else
    target = matrix.col_major(b, o):uniformf(0,1,rnd)
  end
  if norm then
    apply(function(m) m:scal(1/m:sum()) end,
	  target:sliding_window():iterate())
  end
  result = trainer:grad_check_step(tokens.matrix(input),
				   tokens.matrix(target),
				   verbose)
  if not result then
    print("---- INPUT ----")
    print(input)
    print("---- TARGET ----")
    print(target)
    for name,c in trainer:iterate_components() do
      print("---- " .. name .. " ----")
      print(c:get_input():get_matrix())
      print(c:get_output():get_matrix())
      print(c:get_error_input():get_matrix())
      print(c:get_error_output():get_matrix())
    end
    error(string.format("Error at %s (%d,%d,%d,%s) !!!",desc,i,o,b,loss_name))
  end
end

------------------------
-- DOT PRODUCT + BIAS --
------------------------

for i=2,4 do
  for o=2,4 do
    for b=1,4 do
      check_component(function()
			return ann.components.stack():
			push( ann.components.dot_product{ input=i, output=o } ):
			push( ann.components.bias{ size=o } )
		      end,
		      "mse", i, o, b, "DOTPRODUCT + BIAS")
    end
  end
end

-----------------------------------
-- DOT PRODUCT + BIAS + FMEASURE --
-----------------------------------

for i=2,4 do
  for o=1,4 do
    b=32
    check_component(function()
		      return ann.components.stack():
		      push( ann.components.dot_product{ input=i, output=o } ):
		      push( ann.components.bias{ size=o } ):
		      push( ann.components.actf.logistic() )
		    end,
		    "batch_fmeasure_micro_avg", i, o, b, "DOTPRODUCT + BIAS + LOGISTIC")
  end
end

for i=2,4 do
  for o=1,4 do
    b=32
    check_component(function()
		      return ann.components.stack():
		      push( ann.components.dot_product{ input=i, output=o } ):
		      push( ann.components.bias{ size=o } ):
		      push( ann.components.actf.logistic() )
		    end,
		    "batch_fmeasure_macro_avg", i, o, b, "DOTPRODUCT + BIAS + LOGISTIC")
  end
end

for i=2,4 do
  for o=2,4 do
    b=32
    check_component(function()
		      return ann.components.stack():
		      push( ann.components.dot_product{ input=i, output=o } ):
		      push( ann.components.bias{ size=o } ):
		      push( ann.components.actf.softmax() )
		    end,
		    "batch_fmeasure_macro_avg", i, o, b, "DOTPRODUCT + BIAS + SOFTMAX",
		    true)
  end
end

--------------------------------
-- SLICE + DOT PRODUCT + BIAS --
--------------------------------

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

------------------
-- AUTO-ENCODER --
------------------

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

--------------------------------------
-- CONVOLUTION + ACTF + MAX POOLING --
--------------------------------------

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

-------------------------------
-- COPY + JOIN + DOT PRODUCT --
-------------------------------

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

--------------
-- LOGISTIC --
--------------

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

------------------
-- LOG_LOGISTIC --
------------------

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

----------
-- TANH --
----------

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

--------------
-- SOFTPLUS --
--------------

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

-------------
-- SOFTMAX --
-------------

for i=1,4 do
  for o=3,6 do
    for b=1,4 do
      check_component(function()
			return ann.components.stack():
			push( ann.components.dot_product{ input=i, output=o } ):
			push( ann.components.actf.softmax() )
		      end,
		      "mse", i, o, b, "SOFTMAX")
    end
  end
end

-----------------
-- LOG_SOFTMAX --
-----------------

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
