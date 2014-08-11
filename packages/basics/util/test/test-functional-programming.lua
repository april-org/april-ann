value = reduce(math.min, math.huge, ipairs({4, 2, 1, 10}))
assert(value==1)
assert( iterator(ipairs({4,2,1,10})):reduce(math.min, math.huge) == 1 )

value = reduce(function(acc,v) return acc*2+v end, 0, string.gmatch("01101", "." ))
assert(value==13)
assert( iterator(string.gmatch("01101",".")):reduce(function(acc,v)return acc*2+v end, 0) == 13 )

t = { "a", "c", 3, 2 }
expected = { {1, "a", "a"}, {2, "c", "c"}, {3, 3, 3}, {4, 2, 2} }
apply(function(i,v1,v2) assert(i==expected[i][1] and v1==expected[i][2] and v2==expected[i][3]) end, multiple_ipairs(t,t))
iterator(multiple_ipairs(t,t)):apply(function(i,v1,v2) assert(i==expected[i][1] and v1==expected[i][2] and v2==expected[i][3]) end)

tmapped = map(function(v) return v*2 end, ipairs({1, 2, 3, 4}))
assert(table.concat(tmapped, " ") == "2 4 6 8")
assert(iterator(ipairs({1,2,3,4})):map(function(i,v) return v*2 end):concat(" ") == "2 4 6 8")

tmapped = map2(function(k,v) return k+v*2 end, ipairs({1, 2, 3, 4}))
assert(table.concat(tmapped, " ") == "3 6 9 12")

tmapped = mapn(function(idx, ...) return table.pack(...) end,
	       multiple_ipairs({1, 2, 3, 4},{5, 6, 7, 8}))
expected = { {1,5},{2,6},{3,7},{4,8} }
for i,v in ipairs(tmapped) do assert(v[1]==expected[i][1] and v[2]==expected[i][2]) end

t = filter(function(v) return v%2 == 0 end, ipairs{1,2,3,4,5,6,7})
assert(table.concat(t, " ") == "2 4 6")
assert( iterator(ipairs{1,2,3,4,5,6,7}):filter(function(i,v) return v%2==0 end):map(function(i,v)return v end):concat(" ") == "2 4 6")

t = { Lemon = "sour", Cake = "nice", }
expected = {
  ["lemon is slightly SOUR"]=0,
  ["cake is slightly NICE"]=0,
}
for ingredient, modifier, taste in iterable_map(function(a, b)
						  return a:lower(),"slightly",b:upper()
						end, pairs(t)) do
  local str = ingredient .." is ".. modifier .. " " .. taste
  assert(expected[str] == 0)
  expected[str] = expected[str] + 1
end

t = { Lemon = "sour", Cake = "nice", }
expected = {
  ["cake is very NICE"]=0,
  ["Cake is slightly nice"]=0,
  ["lemon is very SOUR"]=0,
  ["Lemon is slightly sour"]=0,
}
for ingredient, modifier, taste in iterable_map(function(a, b)
                                         coroutine.yield(a:lower(),"very",b:upper())
                                         return a, "slightly", b
                                       end, pairs(t)) do
  local str = ingredient .." is ".. modifier .. " " .. taste
  assert(expected[str]==0)
  expected[str] = expected[str] + 1
end


idx=1
expected={2,4,6}
for v in iterable_filter(function(key,value) return value%2==0 end,
                         ipairs{1,2,3,4,5,6,7}) do
  assert(v == expected[idx])
  idx=idx+1
end
