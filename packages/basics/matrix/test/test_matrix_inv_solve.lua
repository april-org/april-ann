mathcore.set_use_cuda_default(util.is_cuda_available())
--
local check = utest.check
local T = utest.test
T("InverseTest", function()
    -- inverse
    local M = matrix
    local m = M(2,2, {
                    -2, 1,
                    -1, 1,
    })
    local b = M(2,1, {
                  4,
                  1
    })
    local c = m:inv()*b
    check.eq(c, M(2,1,{-3,-2}), "inverse")
    -- pseudo-inverse
    local c = m:pinv()*b
    check.eq(c, M(2,1,{-3,-2}), "pseudo-inverse")
end)
