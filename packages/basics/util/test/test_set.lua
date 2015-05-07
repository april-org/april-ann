local check = utest.check
local T = utest.test

T("SetTest", function()
    local s = set()
    s:add(4)
    check.eq(#s, 1)
    s:add(1)
    check.eq(#s, 2)
    s:discard(10)
    check.eq(#s, 2)
    s:discard(1)
    check.eq(#s, 1)
    check.errored(function() s:remove(10) end)
    --
    local s1 = set{ 1,2,3,4,5 }
    local s2 = set{ 1,2,3,4,5,6,7,8 }
    check.TRUE(s1:issubset(s2))
    check.TRUE(s2:issuperset(s2))
    check.FALSE(s1 == s2)
    check.TRUE(s1 < s2)
    check.eq(s2 - s1, set{6,7,8})
    local s2 = s2 - s1
    check.eq(s2 + s1, set{1,2,3,4,5,6,7,8})
    local s2 = s2 + s1
    check.eq(s2:intersection(s1), s1)
end)
