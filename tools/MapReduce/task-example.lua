-- arg is available as a vararg argument when the script is loaded by loadfile
-- or load functions. Please, use arg to parse arguments with cmdOpt.
local arg = { ... }

require "common"
-- NOTE that common, worker, and master are tables used by MapReduce paradigm,
-- also note that map and reduce functions are defined by April-ANN, so use
-- other names

-- the data size is mandatory. Please, use absolute paths, instead of relative,
-- and remember that in any case the data is loaded in the workers.
dir = os.getenv("APRIL_TOOLS_DIR") .. "/MapReduce/data/"
local data = {
  { dir.."text1.txt", 20 },
  { dir.."text2.txt", 20 },
  { dir.."text3.txt", 20 },
  { dir.."text4.txt", 20 },
  { dir.."text5.txt", 20 },
  { dir.."text6.txt",  9 },
}

if #arg > 0 then
  data = iterator(ipairs(arg)):select(2):
  map(function(v)
	local f = io.popen("wc -l " .. v)
	local n = tonumber(f:read("*l"):match("^([%d]+)"))
	return {v,n}
      end):
  table()
end

-- Loads the data if it was necessary, so the master executes decoding, and
-- workers receives the decoded data. The decoded_data could be a Lua string
-- which has the ability to load the data, or directly a data value. The
-- returned value will be passed to the split function in order to split it.
local function decode(encoded_data,data_size)
  return encoded_data
end

-- This function receives a decoded data, its size, and splits it by the given
-- first,last pair of values, and returns the data value split and the size of
-- the split (it is possible to be different from the indicated last-first+1
-- size). The returned values are automatically converted using tostring()
-- function. So, it is possible to return a string, a table with strings, or a
-- value convertible to string, or a table with values convertible to string.
local function split(decoded_data,data_size,first,last)
  return string.format("return %q,%d,%d",decoded_data,first,last), last-first+1
end

-- Receives a key,value pair, and produces an array of key,value string (or able
-- to be string-converted by Lua) pairs. In Machine Learning problems, decoded
-- values could be cached by the given key, avoiding to load it every time,
-- improving the performance of the application. The common.cache function is
-- useful for this purpose. Please, be careful because all cached values will be
-- keep at memory of the machine where the task was executed.
local function mmap(key,value)
  -- local value = common.cache(key, value)
  local data,first,last = load(value)()
  local f = io.open(data)
  for i=1,first-1 do f:read("*l") end
  local out = iterator(range(first,last)):
  map(function(i) return f:read("*l") or "" end):
  map(string.tokenize):
  iterate(ipairs):select(2):
  reduce(function(acc,w)
	   acc[w] = (acc[w] or 0) + 1
	   return acc
	 end, {})
  -- returns an array of key,value pairs
  return iterator(pairs(out)):enumerate():table()
end

-- receive a key and an array of values, and produces a pair of strings
-- key,value (or able to be string-converted by Lua) pairs
local function mreduce(key,values)
  local sum = 0 for i=1,#values do sum=sum+values[i] end
  return key,sum
end

-- receives a dictionary of [key]=>value, produces a value which is shared
-- between all workers, and shows the result on user screen
local function sequential(list)
  iterator(pairs(list)):apply(function(k,v)print(v,k)end)
  return {1,2,3,4}
end

-- this function receives the shared value returned by sequential function
local function share(value)
  return value
end

-- Check for running, return true for continue, false for stop
local function loop()
  return false
end

return {
  name="EXAMPLE",
  data=data,
  decode=decode,
  split=split,
  map=mmap,
  reduce=mreduce,
  sequential=sequential,
  share=share,
  loop=loop,
}
