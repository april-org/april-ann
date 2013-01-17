profiler={}

profiler.stopwatch = util.stopwatch()
profiler.last_t = 0
profiler.total_time = 0
profiler.profile = {}

function profiler.hook(event)
  profiler.stopwatch:stop()
  local t = profiler.stopwatch:read()
  local funcname = debug.getinfo(2, "n").name
  local last_func = debug.getinfo(3, "n").func
  local elapsed_time = t - profiler.last_t
  local info = debug.getinfo(2, "fnSlu")
  if event == "call" then
    --[[
    if info.what == "Lua" then
      print ("call ->", info.name, info.what, info.namewhat, info.source, info.short_src, info.lastlinedefined)
    end
    --]]
    if funcname then
      if not profiler.profile[info.func] then
        profiler.profile[info.func] = info
        profiler.profile[info.func].call_count = 0
        profiler.profile[info.func].time = 0
      end
      profiler.profile[info.func].call_count = profiler.profile[info.func].call_count + 1
    end
    if last_funcname then
      profiler.profile[last_func].time = profiler.profile[last_func].time + elapsed_time
      profiler.total_time = profiler.total_time + elapsed_time
    end
  elseif event == "return" or event == "tail return" then
    --print ("return ->", debug.getinfo(2,"n").name)
    if funcname and profiler.profile[info.func] then
      profiler.profile[info.func].time = profiler.profile[info.func].time + elapsed_time
      profiler.total_time = profiler.total_time + elapsed_time
    end
  end
  profiler.last_t = t
  profiler.stopwatch:go()
end

function profiler.start()
  profiler.stopwatch:go()
  debug.sethook(profiler.hook, "cr")
end

function profiler.stop()
  debug.sethook()
  profiler.stopwatch:stop()
end

function profiler.save(outfile)
  profile_list = {}
  for func, info in pairs(profiler.profile) do
    table.insert(profile_list, info)
  end
  table.sort(profile_list, function(a,b) return a.time > b.time end)
  fprintf(outfile, "Total time: %f\n", profiler.total_time)
  fprintf(outfile, "%58s\t%8s\t%s\n", "NAME", "TIME", "CALL_COUNT")
  for i, info in ipairs(profile_list) do
    fprintf(outfile,"[%6s]%50s\t%8.2f\t%d\n", info.what, string.format("%s@%s",info.name, info.source), info.time, info.call_count)
  end
end
