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

-- DOCUMENTATION AT README.md
-- https://github.com/pakozm/lua-oop-iter/blob/master/README.md class module
-- section.

-----------------------------
-- class module definition --
-----------------------------
local class = {
  _NAME = "class",
  _VERSION = "0.4",
}

-- a list of class tables declared using class function
local class_tables_list = setmetatable({}, { __mode = "k" })
local function register_class_table(class_table, class_name)
  assert(not class_tables_list[class_name],
         string.format("%s class name exists", class_name))
  class_tables_list[class_name] = class_table
end

-----------------------
-- Private functions --
-----------------------

-- Detect if APRIL-ANN is available.
local type = type
local aprilann_available = (aprilann ~= nil)
if aprilann_available then type = luatype or type end

-- Given two object meta_instance, sets the second as parent of the first.
local function set_parent(child, parent)
  setmetatable(child.index_table, parent)
end

-- Checks if the type of the given object is "table"
local function is_table(t) return type(t) == "table" end

-- Checks if the given Lua object is a class table and returns its meta_instance
-- table.
local function has_metainstance(t)
  return is_table(t) and t.meta_instance
end

-- Checks if the given object is a class table and returns its
-- meta_instance.index_table.
local function has_class_instance_index_metamethod(t)
  return has_metainstance(t) and t.meta_instance.index_table
end

-- Checks if the given object is a class instance and returns its index_table
-- metamethod.
local function has_instance_index_metamethod(t)
  return t and getmetatable(t) and getmetatable(t).index_table
end

-- Converts a Lua object in an instance of the given class.
local function class_instance(obj, class)
  setmetatable(obj, assert(has_metainstance(class),
                           "2nd argument needs to be a class table"))
  obj.__instance__ = true
  return obj
end

----------------------
-- Public functions --
----------------------

-- Returns the class table associated with the given class_name.
--
-- @param class_name - A Lua string with the class_name.
-- @return class_table,methods_table - A Lua class table, and its methods table.
function class.find(class_name)
  local cls = class_tables_list[class_name]
  if cls then
    return class_tables_list[class_name],cls.meta_instance.index_table
  end
end

-- Removes the given class name from the auxiliary class table.
--
-- @param class_name - A Lua string with the class_name.
function class.forget(class_name)
  class_tables_list[class_name] = nil
end

-- Predicate which returns true if a given object instance is a subclass of a
-- given Lua class table.
--
-- @param object_instance - A class instance object.
-- @param base_class_table - A class table.
-- @return boolean
function class.is_a(object_instance, base_class_table)
  if not class.of(object_instance) then return false end
  assert(has_metainstance(base_class_table),
         "Needs a class table as 2nd parameter")
  local id = (base_class_table.meta_instance or {}).id
  local ok,result = pcall(function() return object_instance["is_" .. id] ~= nil end)
  return ok and result
end

-- Returns the super class table of a given derived class table. Throws an error
-- if the given class has not a super class.
--
-- @param class_table - A class table.
-- @return super_class_table - The parent (super) class table.
function class.super(class_table)
  assert(has_metainstance(class_table),
         "Needs a class table as 1st parameter")
  return assert( (getmetatable(class_table) or {}).parent,
    "The given class hasn't a super-class" )
end

-- Returns the class table of the given object instance.
--
-- @param obj - A Lua object.
-- @return class_table - The class table of the given object.
function class.of(obj)
  return (getmetatable(obj) or {}).cls
end

-- Returns the value associated with the given key at the given
-- class_table. Throws an error if the 1st parameter is not a class table.
--
-- @param class_table - A Lua class table.
-- @param key - A Lua string with the name you want to consult.
-- @return value - The Lua value associated to the given key name.
function class.consult(class_table, key)
  local index = has_class_instance_index_metamethod(class_table)
  assert(index, "The given object is not a class")
  if type(index) == "function" then
    return index(nil,key)
  else
    return index[key]
  end
end

-- Returns the value associated with the given key at the given class_table
-- meta_instance. Throws an error if the 1st parameter is not a class table.
--
-- @param class_table - A Lua class table.
-- @param key - A Lua string with the name you want to consult at the meta_instance.
-- @return value - The Lua value associated to the given key name.
function class.consult_metamethod(class_table, key)
  return assert(has_metainstance(class_table),
                "The given object is not a class")[key]
end

-- Calls a method in a given class_table using the given vararg arguments. It
-- throws an error if the 1st parameter is not a class table or if the given
-- method doesn't exist.
--
-- @param class_table - A Lua class_table.
-- @param method - A Lua value with the method key.
-- @param ... - A vararg list which will be passed to the method call.
-- @return value - The value returned by the method call.
function class.call(class_table, method, ...)
  local method = assert(class.consult(class_table, method),
                        "Method " .. method .. " not implemented")
  return method(...)
end

-- Extends the given class table with the addition of a new key = value pair
-- into the object instance table. It throws an error if the 1st parameter is
-- not a class table.
--
-- @param class_table - A Lua class table.
-- @param key - A Lua key used as index.
-- @param value - A Lua value which will be stored at the given key.
function class.extend(class_table, key, value)
  local index = has_class_instance_index_metamethod(class_table)
  assert(index, "The given 1st parameter is not a class")
  assert(type(index) == "table")
  index[key] = value
end

-- Extends the given class table with the addition of a new key = value pair
-- into the object meta_instance table, where metamethods are stored. It throws
-- an error if the 1st parameter is not a class table. Be careful, several
-- metamethods (__index, __gc) are defined by default in order to implement OOP,
-- overwritten them will produce unexpected errors. However, __tostring
-- metamethod is also defined but it is totally safe to overwrite it.
--
-- @param class_table - A Lua class table.
-- @param key - A Lua key used as index.
-- @param value - A Lua value which will be stored at the given key.
function class.extend_metamethod(class_table, key, value)
  assert(key ~= "__index" and key ~= "__gc",
         "__index and __gc metamethods are forbidden")
  assert(key ~= "id" and key ~= "cls", "id and cls keys are forbidden")
  assert(has_metainstance(class_table),
         "The given 1st parameter is not a class")[key] = value
end

-- Returns true/false if the given instance object is an instance of a derived
-- class.
--
-- @param obj - A Lua class instance.
-- @return boolean
function class.is_derived(obj)
  return getmetatable((getmetatable(obj) or {}).cls or {}).parent ~= nil
end

-- Returns true/false if the given Lua value is a class table.
--
-- @param t - A Lua value.
-- @return boolean
function class.is_class(t)
  -- not not allows to transform the returned value into boolean
  return not not has_class_instance_index_metamethod(t)
end

-- Changes the __index table by a given function, in case the function
-- returns "nil", the key would be searched at the old index field
function class.declare_functional_index(cls, func)
  assert(class.is_class(cls), "Needs a class as first argument")
  assert(type(func) == "function", "Needs a function as second argument")
  local old_index = cls.meta_instance.__index
  if type(old_index) ~= "function" then
    cls.meta_instance.__index = function(self,key)
      local v = func(self,key)
      return v~=nil and v or old_index[key]
    end
  else
    cls.meta_instance.__index = function(self,key)
      local v = func(self,key)
      return v~=nil and v or old_index(self,key)
    end
  end
end

-- TODO: reimplement this function
--
-- makes a wrapper around an object, delegating the function calls to the given
-- object if they are not implemented in the given wrapper table
local function wrapper(obj,wrapper)
  local wrapper = wrapper or {}
  local current = obj
  while class.of(current) do
    -- and not rawequal(getmetatable(current).__index,current) do
    current = instance_index_metamethod(current)
    for i,v in pairs(current) do
      if wrapper[i] == nil then
	if type(v) == "function" then
	  wrapper[i] =
	    function(first, ...)
	      if rawequal(first,wrapper) then
		return obj[i](obj, ...)
	      else
		return obj[i](...)
	      end
	    end -- function
        elseif getmetatable(v) and getmetatable(v).__call then
          error("Not implemented wrapper for callable tables")
	else -- if type(v) == "function"
	  wrapper[i] = v
	end -- if type(v) == "function" ... else
      end -- if wrapper[i] == nil
    end -- for
  end -- while
  if class.of(wrapper) then
    if class.is_derived(wrapper) then
      error("class_wrapper not works with derived or nil_safe objects")
    else
      set_parent(getmetatable(wrapper),getmetatable(obj))
    end
  else
    wrapper = class_instance(wrapper, class.of(obj))
  end
  return wrapper
end

-- Creates a class table with a given class_name. It receives an optional parent
-- class to implement simple heritance. It returns the class table; another
-- table which will contain the methods of the object. Constructor and
-- destructor methods will be declared into the class table as
-- class_name:constructor(...) and class_name:destructor(). Additionally, a
-- third optional argument is given, which allows to give a predefined
-- class_table, useful is you want to make global instead of local variables.
local class_call_metamethod = function(self, class_name, parentclass, class_table)
  local class_table = class_table or {}
  assert(not class_table.constructor and not class_table.destructor and not class_table.meta_instance,
         "3rd argument has a constructor, destructor or meta_instance field")
  class_table.constructor = function() end
  class_table.destructor  = function() end
  --
  register_class_table(class_table, class_name)
  -- local class_table = get_table_from_dotted_string(class_name, true)
  -- if type(parentclass) == "string" then
  -- parentclass = get_table_from_dotted_string(parentclass)
  -- end
  assert(parentclass==nil or has_metainstance(parentclass),
	 "The parentclass must be defined by 'class' function")
  -- local t = string.tokenize(class_name,".")
  --
  local meta_instance = {
    id         = class_name,
    cls        = class_table,
    __tostring = function(self) return "instance of " .. class_name end,
    __index    = { ["is_"..class_name] = true },
  }
  meta_instance.index_table = meta_instance.__index
  meta_instance.__gc = function(self)
    if self.__instance__ then
      class_table.destructor(self)
      if parentclass then parentclass.meta_instance.__gc(self) end
    end
  end
  local class_metatable = {
    id         = class_name .. " class",
    parent     = parentclass,
    __tostring = function() return "class ".. class_name .. " class" end,
    __concat   = function(a,b)
      assert(type(b) == "string", "Needs a string as second argument")
      return class.consult(a,b)
    end,
    __call = function(self, ...)
      local obj = class_instance({}, self)
      class_table.constructor(obj, ...)
      return obj
    end,
    --    __index    = function(t,k)
    --      local aux = rawget(t,k)
    --      if aux then return aux else return t.meta_instance.__index[k] end
    --    end,
  }
  if parentclass then
    set_parent(meta_instance, parentclass.meta_instance)
  end
  -- 
  class_table.meta_instance = meta_instance
  setmetatable(class_table, class_metatable)
  return class_table, class_table.meta_instance.__index
end
setmetatable(class, { __call = class_call_metamethod })

-- In APRIL-ANN this module is defined at global environment
if aprilann_available then
  _G.class = class
  _G.class_instance = class_instance
end

return class
