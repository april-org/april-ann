-- arg is available as a vararg argument when the script is loaded by loadfile
-- or load functions. Please, use arg to parse arguments with cmdOpt.
local arg = { ... }

require "common"
-- NOTE that common, worker, and master are tables used by MapReduce paradigm,
-- also note that map and reduce functions are defined by April-ANN, so use
-- other names

-- the data size could be nil, but then it is mandatory to produce a data size
-- in decode function. Please, use absolute paths, instead of relative, and
-- remember that in any case the data is loaded in the workers.
dir = "/home/pako/programas/april-ann/tools/MapReduce/"
local data = {
  { dir.."data/text1.txt", 20 },
  { dir.."data/text2.txt", 20 },
  { dir.."data/text3.txt", 20 },
  { dir.."data/text4.txt", 20 },
  { dir.."data/text5.txt", 20 },
  { dir.."data/text6.txt",  9 },
}

if #arg > 0 then
  data = iterator(ipairs(arg)):select(2):map(function(v) return {v,nil} end):table()
end

-- Loads the data if it was necessary, so the master executes decoding, and
-- workers receives the decoded data. The decoded_data could be a Lua string
-- which has the ability to load the data, or directly a data value. The size of
-- the data could be different after decoding.
local function decode(encoded_data,data_size)
  local f = io.open(encoded_data)
  local all = string.tokenize(f:read("*a"),"\n")
  f:close()
  local data = string.format("return %s", table.tostring(all))
  return data,#all
end

-- This function receives a decoded data, its size, and splits it by the given
-- first,last pair of values, and returns the data value split and the size of
-- the split (it is possible to be different from the indicated last-first+1
-- size).
local function split(decoded_data,data_size,first,last)
  local data = loadstring(decoded_data)()
  local data = table.slice(data, first, last)
  local data_str = string.format("return %s", table.tostring(data))
  return data_str,#data
end

-- Receives a key,value pair, and produces an array of key,value string (or able
-- to be string-converted by Lua) pairs. In Machine Learning problems, decoded
-- values could be cached by the given key, avoiding to load it every time,
-- improving the performance of the application. The common.cache function is
-- useful for this purpose. Please, be careful because all cached values will be
-- keep at memory of the machine where the task was executed.
local function mmap(key,value)
  local value = common.cache(key, function() return loadstring(value)() end)
  -- iterate over data lines, and line tokenization
  local out = iterator(ipairs(value)):select(2):map(string.tokenize):
  -- word iterator and reduction
  iterate(ipairs):select(2):reduce(function(acc,w)
				     acc[w] = (acc[w] or 0) + 1
				     return acc
				   end, {})
  -- returns an array of key,value pairs
  return iterator(pairs(out)):table()
end

-- receive a key and an array of values, and produces a pair of strings
-- key,value (or able to be string-converted by Lua) pairs
local function mreduce(key,values)
  return key,iterator(ipairs(values)):select(2):reduce(math.add(),0)
end

-- Check for running, return true for continue, false for stop
local function loop()
  return false
end

-- receives a dictionary of [key]=>value, produces a value which is shared
-- between all workers, and shows the result on user screen
local function sequential(list)
  iterator(pairs(list)):apply(function(k,v)print(v,k)end)
  return
end

-- this function receives the shared value returned by sequential function
local function shared(value)
  return
end

return {
  data=data,
  decode=decode,
  split=split,
  map=mmap,
  reduce=mreduce,
  sequential=sequential,
  shared=shared,
  loop=loop,
}
