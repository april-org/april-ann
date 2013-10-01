package.path = string.format("%s?.lua;%s", string.get_path(arg[0]), package.path)
require "common"
local task_script = table.remove(arg,1)
task              = common.load(task_script,nil,table.unpack(arg)) or error("Error loading the script")
data              = task.data   or error("Needs a data table")
mmap              = task.map    or error("Needs a map function")
mreduce           = task.reduce or error("Needs a reduce function")
decode            = task.decode or function(...) return ... end
sequential        = task.sequential or function(...) print(...) end
share             = task.share or function(...) return ... end
loop              = task.loop or function() return false end
split             = task.split or
  function(data,data_size,first,last)
    return data,data_size
  end

local total_size = iterator(ipairs(data)):select(2):field(2):reduce(math.add(),0)

repeat
  local reduction = {}
  for i=1,#data do
    collectgarbage("collect")
    -- data decoding
    local encoded_data = data[i][1]
    local data_size = data[i][2]
    local decoded_data = decode(encoded_data,data_size)
    local N = total_size / 4
    -- data split
    local first,last = 1,math.min(N,data_size)
    repeat
      local splitted_data,size = split(decoded_data,data_size,first,last)
      last = first + size - 1
      -- MAP
      local key = string.format("#%d#%d#%d#",i,first,last)
      local map_result = mmap(key,splitted_data)
      if #map_result == 0 then
	print("WARNING!!! check that the map function is returning an array "..
		" and not a dictionary")
      end
      -- store map result in reduction table, accumulating all the values with
      -- the same key
      for i=1,#map_result do
	local k,v = table.unpack(map_result[i])
	reduction[k] = table.insert(reduction[k] or {}, v)
      end
      --
      first = last + 1
      last  = math.min(last + N,data_size)
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
  share(value)
until not loop()
