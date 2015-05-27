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
    local t = table.pack( multiple_unpack(table.pack(1, nil, 3, nil),
                                          table.pack(5, nil, nil, 8)) )
    check.eq(t.n, 8)
    for i=1,t.n do
      check.TRUE(t[i] == i or t[i] == nil)
    end
end)

T("SerializationTest", function()
    local t = {1,2,3,a={2,3},c=util.stopwatch()}
    local t2 = util.deserialize(util.serialize(t))
    check.eq(t2[1],1)
    check.eq(t2[2],2)
    check.eq(t2[3],3)
    check.eq(t2.a[1],2)
    check.eq(t2.a[2],3)
    check.eq(type(t2.c), "util.stopwatch")
end)

T("LambdaTest", function()
    local f = lambda'|x|3*x'
    local g = bind(lambda'|f,x|f(x)^2', f)
    check.eq(f(4),12)
    check.eq(g(4),144)
end)

T("CastTest", function()
    local tmp = os.tmpname()
    local f = aprilio.stream.file(tmp, "w")
    local f1 = cast.to(f, aprilio.stream)
    check.TRUE( class.is_a(f1, aprilio.stream) )
    local f2 = cast.to(f, aprilio.stream.file)
    check.TRUE( class.is_a(f2, aprilio.stream.file) )
    check.TRUE( class.is_a(f1, aprilio.stream.file) )
    check.errored(function() assert(cast.to(f, dataset)) end)
    f:close()
    os.remove(tmp)
end)