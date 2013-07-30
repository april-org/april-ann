matlab = matlab or {}

-- this table keeps a dictionary of functions which receives an element of a
-- given class and returns an April-ANN Lua object
local class_tolua_table = {
  [0] = function(e) return e end,
}

-- auxiliary function to add new class builder functions
local function add_wrapper(class_number, f)
  class_tolua_table[class_number] = f
end

-- this function receives an element and returns its corresponding April-ANN Lua
-- object
function matlab.tolua(element)
  if element:get_type() == matlab.types.matrix then
    return class_tolua_table[element:get_class()](element)
  else
    error(string.format("Not implemented yet for type '%s'",
			matlab.types[element:get_type()]))
  end
end

-----------------------------
-- CLASS BUILDER FUNCTIONS --
-----------------------------
local function tomatrix(e) return e:get_matrix() end
local function tocellarray(e)
  local cell_array,name = e:get_cell_array()
  local wrapper = class_wrapper(cell_array)
  wrapper.get = function(obj, ...)
    return matlab.tolua(cell_array:get(...))
  end
  return wrapper,name
end

-- addition of all the functions to class_tolua_table
add_wrapper(matlab.classes.double, tomatrix)
add_wrapper(matlab.classes.single, tomatrix)
add_wrapper(matlab.classes.int8,   tomatrix)
add_wrapper(matlab.classes.uint8,  tomatrix)
add_wrapper(matlab.classes.int16,  tomatrix)
add_wrapper(matlab.classes.uint16, tomatrix)
add_wrapper(matlab.classes.int32,  tomatrix)
add_wrapper(matlab.classes.uint32, tomatrix)

add_wrapper(matlab.classes.cell_array, tocellarray)

-----------------------------------------------------------------------------

-------------------
-- MAIN FUNCTION --
-------------------

-- reads a MAT file and returns a Lua table with all the elements, indexed by
-- its names, and recursively containing matrix, cell_array or struct objects
-- with all the data
function matlab.read(path)
  local reader   = matlab.reader(path)
  local elements = {}
  for e in reader:elements() do
    local lua_element,name = matlab.tolua(e)
    elements[name] = lua_element
  end
  return elements
end
