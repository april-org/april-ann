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

-- Clones the function and its upvalues
local function clone_function(func)
  -- clone by using a string dump
  local ok,func_dump = pcall(string.dump, func)
  local func_clone = (ok and loadstring(func_dump)) or func
  if func_clone ~= func then
    -- copy upvalues
    local i = 1
    while true do
      local name,value = debug.getupvalue(func,i)
      if not name then break end
      -- TODO: implement cone (deep copy) of tables
      if type(value) == "function" then value = clone_function(value) end
      debug.setupvalue(func_clone, i, value)
      i = i + 1
    end
  end
  return func_clone
end

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

local function iscallable(obj)
  local t = type(obj)
  if t == "function" then return true end
  if t == "table" or t == "userdata" then return (getmetatable(obj) or {}).__call end
  return false
end

--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

-- Function generators

-- Range iterator, receives start,stop,step
function iterator.range(...)
  local arg = pack(...)
  local start,stop,step = arg[1],arg[2],arg[3] or 1
  if not stop then start,stop=1,start end
  local i = start - step
  if step > 0 then
    return iterator(function(step, i)
        if step and i then
          i = i + step
          if i <= stop then return i end
        end
                    end, step, i)
  else
    return iterator(function(step, i)
        if step and i then
          i = i + step
          if i >= stop then return i end
        end
                    end, step, i)
  end
end

-- Duplicates its arguments in an infinite iterator.
function iterator.duplicate(...)
  return iterator(function(arg) return unpack(arg) end, pack(...))
end

-- Returns an inifite iterator which calls the given function with: f(0), f(1),
-- ..., f(i), ...
function iterator.tabulate(func)
  local x=-1
  return iterator(function() x=x+1 return func(x) end)
end

-- Returns a zero indefinitely.
function iterator.zeros()
  return iterator(function() return 0 end)
end

-- Returns a one indefinitely.
function iterator.ones()
  return iterator(function() return 1 end)
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

--------------------------------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

-- Constructor of class iterator. It is a wrapper around a Lua iterator
-- function, which allow to keep the iterator state, allowing to write easy
-- functional operations (map, reduce, filter, ...).
function iterator:constructor(f, s, v)
  if class.is_a(f, iterator) then
    assert(not s and not v, "Given s and v parameters with an iterator object")
    f,s,v = f:get()
  elseif not iscallable(f) and type(f) == "table" then
    if #f == 0 then f,s,v = iterator(pairs(f)):select(2):get()
    else f,s,v = iterator(ipairs(f)):select(2):get()
    end
  end
  assert(iscallable(f), "Needs a Lua iterator tripplete, a table, a function or a callable table")
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

-- Converts to table and then unpack the resulting table.
function iterator_methods:unpack(...)
  return table.unpack(self:table(), ...)
end

-- Returns the iterator result at nth position.
function iterator_methods:nth(nth)
  for i=1,nth-1 do
    if not self() then break end
  end
  return self()
end

-- Returns the head of the iterator.
function iterator_methods:head()
  return self:nth(1)
end

-- Returns the tail of the iterator.
function iterator_methods:tail()
  self() -- skip first value
  return self
end

-- Returns the first n elements of the iterator, or the first which satisfy a
-- given predicate
function iterator_methods:take(n)
  if type(n) == "number" then
    local i=0
    return iterator(function() if i < n then i=i+1 return self() end end)
  else
    local satisfied = true
    return iterator(function()
        if satisfied then
          local result = pack( self() )
          if not n( unpack(result) ) then
            satisfied = false
          else
            return unpack( result )
          end
        end
    end)
  end
end

-- Skips the first n elements of the iterator, or the first which satisfy a
-- given predicate
function iterator_methods:drop(n)
  if type(n) == "number" then
    for i=1,n do
      if not self() then break end
    end
    return self
  else
    local result
    repeat result = pack(self()) until not n( unpack(result) )
    return iterator(function()
        local aux
        aux,result = result,pack( self() )
        return unpack(aux)
    end)
  end
end

-- Splits an iterator into two iterators, only works with pure functional
-- iterators.
function iterator_methods:split(n)
  return self:clone():take(n),self:clone():drop(n)
end

-- Only works properly with pure functional iterators.
function iterator_methods:clone()
  local f,s,v = self:get()
  local f = clone_function(f)
  return iterator(f,s,v)
end

-- Returns the position of the first iterator index which is equals to the given
-- arguments.
function iterator_methods:index(...)
  local arg = pack(...)
  local idx=0
  while true do
    idx=idx+1
    local current = pack(self()) if not current[1] then break end
    assert(#current == #arg, "Incorrect number of arguments")
    local eq = true
    for j=1,#current do
      if current[j] ~= arg[j] then eq = false break end
    end
    if eq then return idx end
  end
  return nil
end

-- Returns an iterator to positions which are equal to the given arguments.
function iterator_methods:indices(...)
  local arg = pack(...)
  return iterator(function(self, idx)
      while true do
        idx = idx+1
        local current = pack(self()) if not current[1] then break end
        local eq = true
        for j=1,#current do
          if current[j] ~= arg[j] then eq = false break end
        end
        if eq then return idx end
      end
                  end, self, 0)
end

-- Filters by using a regular expression.
function iterator_methods:grep(regexp_string)
  assert(type(regexp_string) == "string", "Only valid with string values")
  return self:filter(function(str,...)
      assert(not ..., "Only valid for unary iterators")
      assert(type(str) == "string", "Only valid with string values")
      return str:find(regexp_string) ~= nil
  end)
end

-- Returns two iterators where elements do and not do satisfy the given
-- predicate.
function iterator_methods:partition(pred)
  return self:clone():filter(pred),
  self:clone():filter(function(...) return not pred(...) end)
end

-- Returns two iterators where elements do and not do satisfy the given
-- predicate.
function iterator_methods:size()
  return self:reduce(function(acc,x) return acc + 1 end, 0)
end

-- Returns true if all return values satisfy the predicate.
function iterator_methods:all(pred)
  return self:reduce(function(acc,...) return acc and pred(...) end, true)
end

-- Returns true if at least one return value satisfies the predicate.
function iterator_methods:any(pred)
  return self:reduce(function(acc,...) return acc or pred(...) end, false)
end

-- Returns the sum.
function iterator_methods:sum()
  return self:reduce(function(a,b) return a+b end, 0)
end

-- Returns the product.
function iterator_methods:prod()
  return self:reduce(function(a,b) return a*b end, 1)
end

-- Returns the max.
function iterator_methods:max()
  return self:reduce(math.max, -math.huge)
end

-- Returns the min.
function iterator_methods:min()
  return self:reduce(math.min, math.huge)
end

-- In APRIL-ANN this module is defined at global environment
if aprilann_available then
  _G.apply = apply
  _G.iscallable = iscallable
  _G.iterator = iterator
  _G.iterable_filter = filter
  _G.iterable_map = map
  _G.range = iterator.range
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
