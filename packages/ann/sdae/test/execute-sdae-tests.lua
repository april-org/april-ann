epsilon = 1e-03

function assert_differences(test, results)
  local f = io.open(test)
  local r = io.open(results)
  for f_line in f:lines() do
    local r_line = r:read("*l")
    local f_t = string.tokenize(f_line)
    local r_t = string.tokenize(r_line)
    for i=1,#f_t do
      if tonumber(f_t[i]) then
	assert(math.abs(tonumber(f_t[i]) - tonumber(r_t[i])) < epsilon,
	       "Incorrect value found during " .. results)
      end
    end
  end
end
--
assert(os.execute("april-ann test/test.lua > /tmp/test.log") == 0,
       "Error executing script test.lua")
assert_differences("/tmp/test.log", "test/results.log")
--
assert(os.execute("april-ann test/test_on_the_fly.lua > /tmp/test.log") == 0,
       "Error executing script test_on_the_fly.lua")
assert_differences("/tmp/test.log", "test/results_on_the_fly.log")
--
os.execute("rm -f /tmp/test.log")
