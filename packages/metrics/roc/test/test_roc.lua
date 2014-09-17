local T = utest.test
local check = utest.check
--
T("ROCTest", function()
    local roc_instance = metrics.roc()
    local output = matrix(100):uniformf(0,1,random(1234))
    local target = matrix(100):uniform(0,1,random(4567))
    roc_instance:add(output('1:50'), target('1:50'))
    roc_instance:add(output('51:100'), target('51:100'))
    check.number_eq( roc_instance:compute_area(), 0.5)
end)
