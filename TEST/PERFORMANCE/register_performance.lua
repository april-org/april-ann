local script = table.remove(arg,1)
assert(script, "Needs a performance script as argument")
local outfilename = string.format("%sresults_%s_v%d.%d_%d.log",
				  string.get_path(script),
				  (io.popen("hostname", "r"):read("*l")),
				  util.version())
-- 4 CORES
util.omp_set_num_threads(4)
--
local f = io.open(outfilename, "w")
-- change global environment
local old_printf = printf
local old_print = print
printf = function(fmt,...) old_printf(fmt,...) f:write(string.format(fmt,...)) f:flush() end
print = function(...) old_print(...) f:write(...) f:write("\n") f:flush() end
april_print_script_header(nil,f)
assert(loadfile(script))(arg)
f:close()
