local check = utest.check
local T = utest.test
--
local lines = {
  "first line",
  "second line with more data",
  "third line with a lot more of data",
}
local lines_concat = table.concat(lines, "\n") .. "\n"
local REP = 10000
--
T("CStringTest", function()
    local s = aprilio.stream.c_string()
    check.TRUE( s:eof() )
    for i=1,REP do s:write(lines_concat) end
    check.FALSE( s:eof() )
    local j = 0
    for line in s:lines() do
      check.eq( line, lines[j+1] )
      j = (j+1) % #lines
    end
    check.TRUE( s:eof() )
end)
--
T("InputLuaStringTest", function()
    local s = aprilio.stream.input_lua_string(lines_concat)
    check.FALSE( s:eof() )
    local j = 0
    for line in s:lines() do
      check.eq( line, lines[j+1] )
      j = (j+1) % #lines
    end
    check.TRUE( s:eof() )
end)
--
T("ReadAllTest", function()
    local s = aprilio.stream.input_lua_string(lines_concat)
    local out = s:read("*a")
    check.eq( out, lines_concat )
end)
