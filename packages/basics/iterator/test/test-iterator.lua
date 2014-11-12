local check = utest.check
local T = utest.test
-- reduce
T("ReduceTest", function()
    check.eq(iterator(ipairs({4,2,1,10})):reduce(math.min, math.huge), 1)
    check.eq(iterator(string.gmatch("01101",".")):reduce(function(acc,v)return acc*2+v end, 0), 13)
end)
-- map
T("MapTest", function()
    check.eq(iterator(ipairs({1,2,3,4})):map(function(i,v) return v*2 end):concat(" "),
             "2 4 6 8")
end)
-- filter
T("FilterTest", function()
    check.eq(iterator(ipairs{1,2,3,4,5,6,7}):filter(function(i,v) return v%2==0 end):map(function(i,v)return v end):concat(" "),
             "2 4 6")
end)

T("IterableMapTest", function()
    local t = { Lemon = "sour", Cake = "nice", }
    local expected = {
      ["lemon is slightly SOUR"]=0,
      ["cake is slightly NICE"]=0,
    }
    for ingredient, modifier, taste in iterator(pairs(t)):map(function(a, b)
        return a:lower(),"slightly",b:upper()
    end) do
      local str = ingredient .." is ".. modifier .. " " .. taste
      check.eq(expected[str], 0)
      expected[str] = expected[str] + 1
    end

    local t = { Lemon = "sour", Cake = "nice", }
    local expected = {
      ["cake is very NICE"]=0,
      ["Cake is slightly nice"]=0,
      ["lemon is very SOUR"]=0,
      ["Lemon is slightly sour"]=0,
    }
    for ingredient, modifier, taste in iterator(pairs(t)):map(function(a, b)
        coroutine.yield(a:lower(),"very",b:upper())
        return a, "slightly", b
    end) do
      local str = ingredient .." is ".. modifier .. " " .. taste
      check.eq(expected[str], 0)
      expected[str] = expected[str] + 1
    end

    local idx=1
    local expected={2,4,6}
    for v in iterator(ipairs{1,2,3,4,5,6,7}):filter(function(key,value) return value%2==0 end) do
      check.eq(v, expected[idx])
      idx=idx+1
    end
    
    for k,v in iterator(ipairs{1,2,3}) do check.eq(k, v) end
end)
