local check = utest.check
local T = utest.test

T("GenericOptionsTest", function()
    check(function() return util.options.test() end)
    local tbl,str2 = util.options.test()
    check.TRUE(tbl.clock1)
    check.TRUE(tbl.clock2)
    check.TRUE(class.is_a(tbl.clock1, util.stopwatch))
    check.TRUE(class.is_a(tbl.clock2, util.stopwatch))
    check.eq(tbl.str, "Hello world!")
    check.eq(str2, util.to_lua_string(tbl))
end)

T("BindFunctionTest", function()
    -- bind function
    local f = bind(math.add, 5)
    local g = bind(math.div, nil, 3)
    check.eq( f(3), 8 )
    check.eq( g(6), 2 )
end)

T("MultipleUnpackTest", function()
    local t = table.pack( multiple_unpack({1,2,3},{4,5},{6,7,8}) )
    for i=1,#t do check.eq(t[i], i) end
    local t = table.pack( multiple_unpack{1,2,3,4,5,6,7,8,9} )
    for i=1,#t do check.eq(t[i], i) end
end)
