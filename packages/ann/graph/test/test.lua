-- forces the use of CUDA
mathcore.set_use_cuda_default(util.is_cuda_available())
--

local check   = utest.check
local T       = utest.test

T("ANNGraphSourceTest", ann.graph.test)

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
    nn:connect("input", c_w1)
    nn:connect(c_w1, c_b1)
    nn:connect(c_b1, c_a1)
    nn:connect(c_a1, c_w2)
    nn:connect(c_w2, c_b2)
    nn:connect(c_b2, c_a2)
    nn:connect(c_a2, "output")
    -- build
    local rnd = random(1234)
    local _,weights = nn:build()
    for _,w in pairs(weights) do
      w:uniformf(-0.01, 0.01, rnd)
    end
    -- STACK COMPONENT
    local stack = ann.components.stack()
    stack:push(c_w1)
    stack:push(c_b1)
    stack:push(c_a1)
    stack:push(c_w2)
    stack:push(c_b2)
    stack:push(c_a2)
    stack:build()
    --
    local input = matrix(20,10):uniformf(-0.1, 0.1, rnd)
    local nn_output = nn:forward(input)
    local stack_output = stack:forward(input)
    print(nn_output)
    print(stack_output)
end)
