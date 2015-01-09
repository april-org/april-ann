-- forces the use of CUDA
mathcore.set_use_cuda_default(util.is_cuda_available())
--

local check   = utest.check
local T       = utest.test

T("ANNGraphSourceTest", ann.graph.test)

T("ANNGraphTest",
  function()
  --
  utest.check.errored(function()
      local g = ann.graph()
      g:connect("input", "output")
      g:set_bptt_truncation(4)
  end)
  utest.check.errored(function()
      local g = ann.graph()
      g:connect("input", "output")
      g:build() g:set_bptt_truncation(4)
  end)
  utest.check.success(function()
      local g = ann.graph()
      local s = ann.graph.add{ input=4, output=2 }
      g:connect("input", s, "output" )
      g:delayed(s, s)
      g:build() g:set_bptt_truncation(4)
      return true
  end)
  --
  local g = ann.graph()
  local b = ann.graph.bind{ input=2, output=4 }
  g:delayed("input", b)
  g:connect("input", b, "output")
  g:build{ input=2, output=4 }
  local m1 = matrix(2,2):linspace()
  local m2 = 2*matrix(2,2):linspace()
  g:forward(m1)
  local o = g:forward(m2)
  utest.check.eq(o, matrix(2,4,{1,2, 2,4,
                                3,4, 6,8}))
  g:backprop(matrix(2,4):ones())
  local r = g:bptt_backprop()
  utest.check.eq(#r, 2)
  utest.check.eq(r[1], matrix(2,2):ones())
  utest.check.eq(r[2], matrix(2,2):ones())
end)

T("ANNGraphComponentTest",
  function()
    -- nodes
    local c_w1 = ann.components.dot_product{ input=10, output=20, weights="w1" }
    local c_b1 = ann.components.bias{ weights="b1", size=20 }
    local c_a1 = ann.components.actf.logistic()
    local c_w2 = ann.components.dot_product{ input=20, output=4, weights="w2" }
    local c_b2 = ann.components.bias{ weights="b2", size=4 }
    local c_a2 = ann.components.actf.linear()
    -- ANN GRAPH
    local nn = ann.graph('nn')
    -- connections
    nn:connect("input", c_w1, c_b1, c_a1, c_w2, c_b2, c_a2, "output")
    -- serialization test
    local nn = nn:clone()
    local tmpname = os.tmpname()
    util.serialize(nn, tmpname)
    local nn = util.deserialize(tmpname)
    os.remove(tmpname)
    -- build
    do
      local rnd = random(1234)
      local _,weights = nn:build()
      for _,name in ipairs{ "w1", "b1", "w2", "b2" } do
        weights[name]:uniformf(-0.01, 0.01, rnd)
      end
    end
    -- input/output sizes
    check.eq(nn:get_input_size(), 10)
    check.eq(nn:get_output_size(), 4)
    -- STACK COMPONENT
    local stack = ann.components.stack()
    stack:push(c_w1:clone())
    stack:push(c_b1:clone())
    stack:push(c_a1:clone())
    stack:push(c_w2:clone())
    stack:push(c_b2:clone())
    stack:push(c_a2:clone())
    -- build
    do
      local rnd = random(1234)
      local _,weights = stack:build()
      for _,name in ipairs{ "w1", "b1", "w2", "b2" } do
        weights[name]:uniformf(-0.01, 0.01, rnd)
      end
    end
    --
    local input = matrix(20,10):uniformf(-0.1, 0.1, random(2474))
    local nn_output = nn:forward(input)
    local stack_output = stack:forward(input)
    check.eq(nn_output, stack_output)
    --
    local e_input = matrix(20,4):uniformf(-0.1, 0.1, random(94275))
    local e_nn_output = nn:backprop(e_input)
    local e_stack_output = stack:backprop(e_input)
    check.eq(e_nn_output, e_stack_output)
    --
    local nn_grads = nn:compute_gradients()
    local stack_grads = nn:compute_gradients()
    local it_zip = iterator.zip
    for nn_g,stack_g in it_zip(iterator(nn_grads), iterator(stack_grads)) do
      check.eq(nn_g, stack_g)
    end
end)

T("AddComponentTest",
  function()
    local s = ann.graph.add():clone()
    s:to_lua_string()
    --
    s:build{ input=30, output=10 }
    check.eq(s:get_input_size(), 30)
    check.eq(s:get_output_size(), 10)
    local out = s:forward(tokens.vector.bunch{ matrix(4,10):linear(),
                                               matrix(4,10):linear(),
                                               matrix(4,10):linear(), })
    check.eq(out, 3 * matrix(4,10):linear())
    local out = s:backprop(matrix(4,10):linear())
    check.TRUE(class.is_a(out, tokens.vector.bunch))
    for _,m in out:iterate() do check.eq(m, matrix(4,10):linear()) end
    check.errored(function() s:build{ input=30, output=13 } end)
end)

T("BindComponentTest",
  function()
    local s = ann.graph.bind():clone()
    s:to_lua_string()
    --
    s:build{ input=30, output=30 }
    check.eq(s:get_input_size(), 30)
    check.eq(s:get_output_size(), 30)
    local out = s:forward(tokens.vector.bunch{ matrix(4,10):linear(),
                                               matrix(4,10):linear(),
                                               matrix(4,10):linear(), })
    local j = matrix.join(2,{ matrix(4,10):linear(),
                              matrix(4,10):linear(),
                              matrix(4,10):linear(), })
    check.eq(out, j)
    local out = s:backprop(j)
    check.TRUE(class.is_a(out, tokens.vector.bunch))
    for _,m in out:iterate() do check.eq(m, matrix(4,10):linear()) end
    check.errored(function() s:build{ input=30, output=20 } end)
end)

T("IndexComponentTest",
  function()
    local s = ann.graph.index(2)
    s:build()
    local out = s:forward(tokens.vector.bunch{ matrix(4,10):fill(1),
                                               matrix(4,10):linear(),
                                               matrix(4,10):fill(3), })
    check.eq(out, matrix(4,10):linear())
    local out = s:backprop(matrix(4,10):fill(5))
    check.TRUE(class.is_a(out, tokens.vector.bunch))
    check.TRUE(class.is_a(out:at(1), tokens.null))
    check.eq(out:at(2), matrix(4,10):fill(5))
    check.TRUE(class.is_a(out:at(3), tokens.null))
end)

T("CmulComponentTest",
  function()
    local s = ann.graph.cmul():clone()
    s:to_lua_string()
    --
    s:build{ input=20, output=10 }
    check.eq(s:get_input_size(), 20)
    check.eq(s:get_output_size(), 10)
    local out = s:forward(tokens.vector.bunch{ matrix(4,10):linear(),
                                               matrix(4,10):linear(), })
    check.eq(out, matrix(4,10):linear():pow(2))
    local out = s:backprop(matrix(4,10):linear())
    check.TRUE(class.is_a(out, tokens.vector.bunch))
    for _,m in out:iterate() do check.eq(m, matrix(4,10):linear():pow(2)) end
    check.errored(function() s:build{ input=20, output=20 } end)
end)

T("ElmanTest",
  function()
    check.errored(function() ann.graph.blocks.elman() end)
    check.errored(function() ann.graph.blocks.elman{ input=10 } end)
    check.errored(function() ann.graph.blocks.elman{ output=10 } end)
    --
    local elman = ann.graph.blocks.elman{ input=10, output=20, name="a" }
    local _,weights,components = elman:build()
    check.eq(elman:get_input_size(), 10)
    check.eq(elman:get_output_size(), 20)
    check.TRUE(weights["a::b"])
    check.TRUE(weights["a::w"])
    check.TRUE(weights["a::context::b"])
    check.TRUE(weights["a::context::w"])
    check.TRUE(components["a::layer"])
    check.TRUE(components["a::b"])
    check.TRUE(components["a::w"])
    check.TRUE(components["a::actf"])
    check.TRUE(components["a::context::layer"])
    check.TRUE(components["a::context::b"])
    check.TRUE(components["a::context::w"])
    check.TRUE(components["a::memory"])
end)

T("LSTMTest",
  function()
    check.errored(function() ann.graph.blocks.lstm() end)
    check.errored(function() ann.graph.blocks.lstm{ input=10 } end)
    check.errored(function() ann.graph.blocks.lstm{ output=10 } end)
    --
    local lstm = ann.graph.blocks.lstm{ input=10, output=20, name="a" }
    local _,weights,components = lstm:build()
end)

T("StackTest",
  function()
    local net   = ann.graph()
    local s1    = ann.components.slice{ pos={1}, size={16*16} }
    local s2    = ann.components.slice{ pos={16*16+1}, size={8*8} }
    local l1    = ann.components.hyperplane{ input=16*16, output=32 }
    local l2    = ann.components.hyperplane{ input=8*8, output=32 }
    local b     = ann.graph.bind()
    local stack = ann.components.stack():
      push( ann.components.actf.relu() ):
      push( ann.components.hyperplane{ input=64, output=10 } ):
      push( ann.components.actf.log_softmax() )
    --
    net:connect('input', s1, l1)
    net:connect('input', s2, l2)
    net:connect({l1,l2}, b, stack, 'output')
    net:build{ input=16*16 + 8*8 }
    --
    check.eq(net:get_input_size(), 16*16 + 8*8)
    check.eq(net:get_output_size(), 10)
    --
    local rnd = random(1234)
    local weights = net:copy_weights()
    local order = iterator(table.keys(weights)):table() table.sort(order)
    iterator(order):apply(function(name) weights[name]:uniformf(-0.01,0.01,rnd) end)
    --
    local out = net:forward(matrix(10, net:get_input_size()):uniformf(0,1,rnd))
    check.eq(#out:dim(), 2)
    check.eq(out:dim(2), 10)
    check.eq(out:dim(1), 10)
end)
