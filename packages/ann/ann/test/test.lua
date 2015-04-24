local check   = utest.check
local T       = utest.test

T("AllAllGenerate", function()
    local generate = ann.mlp.all_all.generate
    local net = generate("10 inputs 4 logistic 3 softmax")
    check.TRUE(net)
    local net = generate("10 inputs 4 logistic dropout{prob=0.5,random=#1} 3 softmax",
                         { random(12384) })
    check.TRUE(net)
    local net = generate("10 inputs 4 sparse_logistic{sparsity=0.1,penalty=3} 3 softmax")
    check.TRUE(net)
end)
