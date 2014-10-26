--[[
  
  Copyright (c) 2014 Francisco Zamora-Martinez (pakozm@gmail.com)
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.

]]

-- Detect if APRIL-ANN is available.
local type = type
local aprilann_available = (aprilann ~= nil)
if aprilann_available then type = luatype or type end
--
local class = class or require "oop-iter.class"

--------------------------------
-- iterator module definition --
--------------------------------
local iterator,iterator_methods = class("iterator")
iterator._NAME = "iterator"
iterator._VERSION = "0.1"

local concat = table.concat
local insert = table.insert
local pack = table.pack
local remove = table.remove
local unpack = table.unpack
local wrap = coroutine.wrap
local yield = coroutine.yield

-- Filters a Lua iterator function output using a given predicate function. The
-- predicate returns true when a value must be taken, false when it must be
-- removed. The predicate function is called as f(...) where ... are the values
-- returned by the iterator function.
local function filter(predicate_func, f, s, v)
  return function(s, v)
    local tmp = pack(f(s, v))
    while tmp[1] ~= nil and not predicate_func(unpack(tmp)) do
      v = tmp[1]
      tmp = pack(f(s, v))
    end
    return unpack(tmp)
  end, s, v
end

-- Iterable map function, receives a map function and a Lua iterator and returns
-- another Lua iterator function. The map function is called as f(...) where
-- ... are the values returned by the iterator function. Multiple results are
-- possible from one map call using coroutine.yield function.
--
-- @note FROM: http://www.corsix.org/content/mapping-and-lua-iterators
local function map(map_func, f, s, v)
  local done
  local function maybeyield(...)
    if ... ~= nil then
      yield(...)
    end
  end
  local function domap(...)
    v = ...
    if v ~= nil then
      return maybeyield(map_func(...))
    else
      done = true
    end
  end
  return wrap(function()
      repeat
        local tmp = pack(f(s,v))
        v = tmp[1]
        domap(unpack(tmp))
      until done
  end), s, v
end

-- Reduce function, receives a reduce function, a reduce initial value, and a
-- Lua iterator, and returns the computation result of the reduction. The
-- reduce function is call as f(acc,...) where acc is the reduced (accumulated)
-- computation, and ... are the values returned by the iterator function.
local function reduce(reduce_func, initial_value, f, s, v)
  assert(initial_value ~= nil,
	 "reduce: needs an initial_value as second argument")
  local accum = initial_value
  local tmp = pack(f(s, v))
  while tmp[1] ~= nil do
    accum = reduce_func(accum, unpack(tmp))
    tmp = pack(f(s, tmp[1]))
  end
  return accum
end

-- Apply function, receives a function and a Lua iterator, and calls the apply
-- function with every element of the iterator. The apply function is called as
-- f(...) where ... are the values returned by the iterator function.
local function apply(apply_func, f, s, v)
  if not apply_func then apply_func = function() end end
  local tmp = pack(f(s,v))
  while tmp[1] ~= nil do
    apply_func(unpack(tmp))
    tmp = pack(f(s,tmp[1]))
  end
end

--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

-- Constructor of class iterator. It is a wrapper around a Lua iterator
-- function, which allow to keep the iterator state, allowing to write easy
-- functional operations (map, reduce, filter, ...).
function iterator:constructor(f, s, v)
  self.f,self.s,self.v = f,s,v
end

-- Returns the underlying Lua iterator.
function iterator_methods:get() return self.f,self.s,self.v end

-- Performs one iterator step, and returns its result.
function iterator_methods:step()
  local tmp = pack( self.f(self.s, self.v) )
  self.v = tmp[1]
  return unpack(tmp)
end

-- Equivalent to step() method, allowing to use iterator objects as Lua
-- iterators in generic for loops.
function iterator.meta_instance:__call() return self:step() end

-- Map method, a wrapper around map function. Multiple results are possible
-- from one map call using coroutine.yield function.
function iterator_methods:map(func)
  return iterator(map(func, self:get()))
end

-- Filter method, a wrapper around filter function.
function iterator_methods:filter(func)
  return iterator(filter(func, self:get()))
end

-- Apply method, a wrapper around apply function.
function iterator_methods:apply(func)
  apply(func, self:get())
end

-- Reduce method, a wrapper around reduce function.
function iterator_methods:reduce(func, initial_value)
  return reduce(func, initial_value, self:get())
end

-- Enumerate method, returns another iterator object which appends as a first
-- element of every iteration a enumeration number.
function iterator_methods:enumerate()
  local id = 0
  return self:map(function(...)
      id = id + 1
      return id, ...
  end)
end

-- Calls a function for every iteration value. The function is a name, and this
-- name must be declared in every iteration value (as table keys, or in its
-- metatable).
function iterator_methods:call(funcname, ...)
  local func_args = pack(...)
  return self:map(function(...)
      local arg    = pack(...)
      local result = {}
      for i=1,#arg do
        local t = pack(arg[i][funcname](arg[i],unpack(func_args)))
        for j=1,#t do insert(result, t[j]) end
      end
      return unpack(result)
  end)
end

-- Performs a nested iteration over every result. It receives a Lua function
-- which returns an iterator (as ipairs, pairs, ...)
function iterator_methods:iterate(iterator_func)
  return self:map(function(...)
      local f,s,v = iterator_func(...)
      local tmp   = pack(f(s,v))
      while tmp[1] ~= nil do
        yield(unpack(tmp))
        tmp = pack(f(s,tmp[1]))
      end
  end)
end

-- Concats the iterator results using sep1 for inter-iteration elements, and
-- sep2 for intra-iteration calls.
function iterator_methods:concat(sep1,sep2)
  local sep1,sep2 = sep1 or "",sep2 or sep1 or ""
  local t = {}
  self:apply(function(...)
      local arg = pack(...)
      insert(t, string.format("%s", concat(arg, sep1)))
  end)
  return concat(t, sep2)
end

-- Indexes iteration result by a given set of indices. It assumes that all
-- the elements in the iterator result are tables.
function iterator_methods:field(...)
  local f,s,v = self:get()
  local arg   = pack(...)
  return iterator(function(s)
      local tmp = pack(f(s,v))
      if tmp[1] == nil then return nil end
      v = tmp[1]
      local ret = { }
      for i=1,#tmp do
        for j=1,#arg do
          insert(ret, tmp[i][arg[j]])
        end
      end
      return unpack(ret)
		  end,
    s,v)
end

-- Selects an iteration result by a given set of number indices.
function iterator_methods:select(...)
  local f,s,v = self:get()
  local arg   = pack(...)
  for i=1,#arg do arg[i]=tonumber(arg[i]) assert(arg[i],"select: expected a number") end
  return iterator(function(s)
      local tmp = pack(f(s,v))
      if tmp[1] == nil then return nil end
      v = tmp[1]
      local selected = {}
      for i=1,#arg do selected[i] = tmp[arg[i]] end
      return unpack(selected)
		  end,
    s,v)
end

-- Stores the iteration into a table. If every iteration has more than one
-- result, the first element will be used as key, otherwise, an enumerated
-- key will be used.
function iterator_methods:table()
  local t = {}
  local idx = 1
  self:apply(function(...)
      local v = pack(...)
      local k = remove(v, 1)
      if #v == 0 then
        k,v = idx,k
      elseif #v == 1 then
        v = v[1]
      end
      t[k] = v
      idx = idx + 1
  end)
  return t
end

-- Returns an iterator over multiple iterators at the same time. The iteration
-- ends if any of the given iterators end.
function iterator.zip(...)
  local arg = { ... }
  for i=1,#arg do assert(class.is_a(arg[i], iterator),
                         "Needs instances of iterator class") end
  local finished = false
  return iterator(function()
      if finished then return nil end
      local result = {}
      for i=1,#arg do
        local partial = table.pack( arg[i]() )
        if not partial[1] then finished=true return nil end
        for k,v in ipairs(partial) do table.insert(result,v) end
      end
      return table.unpack(result)
  end)
end

-- In APRIL-ANN this module is defined at global environment
if aprilann_available then
  _G.apply = apply
  _G.iterator = iterator
  _G.iterable_filter = filter
  _G.iterable_map = map
  _G.reduce = reduce
end

-- UNIT TEST
function iterator.test()
  for k,v in filter(function(k,v) return v % 2 == 0 end, ipairs{1,2,3,4}) do
    assert(v % 2 == 0)
  end
  for k,v in map(function(k,v) return k,k+v end, ipairs{1,2,3,4}) do
    assert(v == 2*k)
  end
  local r = reduce(function(acc,a,b,c) return acc+a+b+c end, 0, map(function(k,v) return k,v,v end, ipairs{1,2,3,4}))
  assert(r == 3 + 6 + 9 + 12)
  apply(function(a,b,c) assert(a==b and b==c) end,
    map(function(k,v) return k,v,v end, ipairs{1,2,3,4}))
end

return iterator
