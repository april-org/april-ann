package.path = string.format("%s?.lua;%s", string.get_path(arg[0]), package.path)
require "common"
local task_script = table.remove(arg,1)
task              = common.load(task_script) or error("Error loading the script")
data              = task.data   or error("Needs a data table")
mmap              = task.map    or error("Needs a map function")
mreduce           = task.reduce or error("Needs a reduce function")
decode            = task.decode or function(...) return ... end
sequential        = task.sequential or function(...) print(...) end
shared            = task.shared or function(...) return ... end
loop              = task.loop or function() return false end
split             = task.split or
  function(data,data_size,first,last)
    return data,data_size
  end

repeat
  local reduction = {}
  for i=1,#data do
    collectgarbage("collect")
    -- data decoding
    local encoded_data = data[i][1]
    local data_size    = data[i][2]
    local decoded_data,decoded_data_size = decode(encoded_data,data_size)
    deocoded_data = decoded_data      or encoded_data
    data_size     = decoded_data_size or data_size
    --
    -- data split
    local N            = 5 -- splits in portions of size 5 (for testing is enough)
    local first,last   = 1,math.min(5,data_size)
    repeat
      local splitted_data,size = split(decoded_data,data_size,first,last)
      last = first + size - 1
      -- MAP
      local key = string.format("#%d#%d#%d#",i,first,last)
      local map_result = mmap(key,splitted_data)
      -- store map result in reduction table, accumulating all the values with
      -- the same key
      reduction = iterator(ipairs(map_result)):select(2):
      reduce(function(acc,t)
	       acc[t[1]] = table.insert(acc[t[1]] or {}, t[2])
	       return acc
	     end,
	     reduction)
      --
      first = last + 1
      last  = last + N
    until first > data_size
  end
  local result = {}
  for key,values in pairs(reduction) do
    collectgarbage("collect")
    -- REDUCE
    local k,v = mreduce(key,values)
    result[k] = v
  end
  local value = sequential(result)
  shared(value)
until not loop()
