matlab = matlab or {}

local function generic_tostring(self)
  local out = {}
  for name,data in pairs(self) do
    local str = data.str and data:str() or ""
    table.insert(out,"%s : %s %s"%{name,type(data),str})
  end
  return table.concat(out, "\n")
end

-- this table keeps a dictionary of functions which receives an element of a
-- given class and returns an APRIL-ANN Lua object
local class_tolua_table = {
  [0] = function(e) return e end,
}

-- auxiliary function to add new class builder functions
local function add_wrapper(class_number, f)
  if not class_number then error("Incorrect class number") end
  class_tolua_table[class_number] = f
end

-- this function receives an element and returns its corresponding APRIL-ANN Lua
-- object
function matlab.tolua(element)
  if element:get_type() == matlab.types.matrix then
    local func = class_tolua_table[element:get_class()]
    if func then return func(element)
    else error("Not recognized class: " .. matlab.classes[element:get_class()])
    end
  else
    error(string.format("Not implemented yet for type '%s'",
			matlab.types[element:get_type()] or tostring(element:get_type()) or "nil"))
  end
  collectgarbage("collect")
end

-----------------------------
-- CLASS BUILDER FUNCTIONS --
-----------------------------
local function tomatrix(e)
  local elem,name = e:get_matrix()
  if elem == nil then
    elem,name = e:get_matrix_complex()
  end
  return elem,name
end
local function tomatrixdouble(e)
  local elem,name = e:get_matrix_double()
  if elem == nil then
    elem,name = e:get_matrix_complex()
  end
  return elem,name
end
local function tomatrixint32(e)
  local elem,name = e:get_matrix_int32()
  if elem == nil then
    error("Not supported complex numbers for integer data types")
  end
  return elem,name
end
local function tomatrixchar(e)
  local elem,name = e:get_matrix_char()
  return elem,name
end
local function tocellarray(e)
  local cell_array,name = e:get_cell_array()
  local wrapper = class_wrapper(cell_array)
  wrapper = class_instance(wrapper, class.of(cell_array))
  wrapper.get = function(obj, ...)
    return matlab.tolua(cell_array:get(...))
  end
  wrapper.raw_get = function(obj, ...)
    return matlab.tolua(cell_array:raw_get(...))
  end
  wrapper.str = function(obj)
    return "dims [%s]"%{table.concat(obj:dim(),",")}
  end
  return wrapper,name
end
local function tostructure(e)
  local dictionary,name = e:get_structure()
  for ename,elem in pairs(dictionary) do
    dictionary[ename] = matlab.tolua(elem)
  end
  setmetatable(dictionary, { __tostring = generic_tostring })
  return dictionary,name
end

-- addition of all the functions to class_tolua_table
-- FLOAT matrix
add_wrapper(matlab.classes.single, tomatrix)
-- DOUBLE matrix
add_wrapper(matlab.classes.double, tomatrixdouble)
-- INT matrix
add_wrapper(matlab.classes.int8,   tomatrixint32)
add_wrapper(matlab.classes.uint8,  tomatrixint32)
add_wrapper(matlab.classes.int16,  tomatrixint32)
add_wrapper(matlab.classes.uint16, tomatrixint32)
add_wrapper(matlab.classes.int32,  tomatrixint32)
add_wrapper(matlab.classes.uint32, tomatrixint32)
add_wrapper(matlab.classes.int64,  tomatrixint32)
add_wrapper(matlab.classes.uint64, tomatrixint32)
-- CHAR matrix
add_wrapper(matlab.classes.char, tomatrixchar)
-- CELL array
add_wrapper(matlab.classes.cell_array, tocellarray)
-- STRUCTURE array
add_wrapper(matlab.classes.structure, tostructure)
-----------------------------------------------------------------------------

-------------------
-- MAIN FUNCTION --
-------------------

-- reads a MAT file and returns a Lua table with all the elements, indexed by
-- its names, and recursively containing matrix, cell_array or struct objects
-- with all the data
function matlab.read(path)
  assert(path and type(path)=="string", "First argument must be a path string")
  local reader    = matlab.reader(path)
  local elements  = {}
  for e in reader:elements() do
    local lua_element,name = matlab.tolua(e)
    elements[name] = lua_element
  end
  setmetatable(elements, { __tostring = generic_tostring })
  return elements
end
