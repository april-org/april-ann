epsilon = 1e-03

tmpname = os.tmpname()

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
assert(os.execute("april-ann.debug "..string.get_path(arg[0]).."/test.lua > "..tmpname) == 0,
       "Error executing script test.lua")
assert_differences(tmpname, string.get_path(arg[0]).."test/results.log")
--
assert(os.execute("april-ann.debug "..string.get_path(arg[0]).."/test_on_the_fly.lua > "..tmpname) == 0,
       "Error executing script test_on_the_fly.lua")
assert_differences(tmpname, string.get_path(arg[0]).."test/results_on_the_fly.log")
--
os.execute("rm -f " .. tmpname)
