local check = utest.check
local T = utest.test

T("GenericOptionsTest", function()
    check(function() return util.options.test() end)
    local tbl = util.options.test()
    check.TRUE(tbl.clock1)
    check.TRUE(tbl.clock2)
    check.eq(tbl.str, "Hello world!")
end)
