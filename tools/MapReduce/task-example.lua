data = {
  { "text1.txt", 20 },
  { "text2.txt", 20 },
  { "text3.txt", 20 },
  { "text4.txt", 20 },
  { "text5.txt", 20 },
  { "text6.txt", 20 },
  { "text7.txt", 16 },
}

-- loads the data if it was necessary, so the master executes decoding, and
-- workers receives the decoded data
decode_function = nil

-- this functions receives a decoded data and splits it by the given first,last
-- pair of values, and returns the data value split and the size of the split
-- (it is possible to be different from the indicated last-first+1 size)
split_function = nil

-- receives an array of key,value pairs, and produces an array of key,value
-- string (or able to be string-converted by Lua) pairs
function map(list)
  -- iterator over all key,value pairs
  local out = iterator(iparis(list)):select(2):
  -- iterate over lines, and line tokenization
  iterate(io.lines):map(string.tokenize):
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
function reduce(key,values)
  return key,iterator(ipairs(values)):select(2):reduce(math.add(),0)
end

-- Check for running, return true for continue, false for stop
-- function loop()
--   return false
-- end

-- receives an array of key,value pairs, produces a value which is shared
-- between all workers, and shows the result on user screen
function sequential(list)
  iterator(ipairs(list)):apply(print)
  return
end
