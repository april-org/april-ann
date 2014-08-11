matlab = matlab or {}

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
function matlab.tolua(element,col_major)
  if element:get_type() == matlab.types.matrix then
    local func = class_tolua_table[element:get_class()]
    if func then return func(element,col_major)
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
local function tomatrix(e,col_major)
  local elem,name = e:get_matrix(col_major)
  if elem ~= nil then
    print("# Loading matrix float element: ", name)
  else
    elem,name = e:get_matrix_complex(col_major)
    print("# Loading matrix complex element: ", name)
  end
  return elem,name
end
local function tomatrixdouble(e,col_major)
  local elem,name = e:get_matrix_double()
  if elem ~= nil then
    print("# Loading matrix double element: ", name)
  else
    elem,name = e:get_matrix_complex(col_major)
    print("# Loading matrix complex (casted from double) element: ", name)
  end
  return elem,name
end
local function tomatrixint32(e)
  local elem,name = e:get_matrix_int32()
  if elem ~= nil then
    print("# Loading matrix int32 element: ", name)
  else
    error("Not supported complex numbers for integer data types")
  end
  return elem,name
end
local function tomatrixchar(e)
  local elem,name = e:get_matrix_char()
  print("# Loading matrix char element:  ", name)
  return elem,name
end
local function tocellarray(e,col_major)
  local cell_array,name = e:get_cell_array()
  local wrapper = class_wrapper(cell_array)
  wrapper = class_instance(wrapper, class.of(cell_array))
  wrapper.get = function(obj, ...)
    return matlab.tolua(cell_array:get(...),col_major)
  end
  wrapper.raw_get = function(obj, ...)
    return matlab.tolua(cell_array:raw_get(...),col_major)
  end
  print("# Loading cell array element:   ", name, cell_array)
  return wrapper,name
end
local function tostructure(e,col_major)
  local dictionary,name = e:get_structure()
  for ename,elem in pairs(dictionary) do
    print("# Loading structure element:  ", name, ename, elem)
    dictionary[ename] = matlab.tolua(elem,col_major)
  end
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

local function process_element(data,name)
  local out = {}
  local prefix=""
  if name and #name > 0 then
    table.insert(out,
		 string.format("# name= '%s', type= %s\n", name, type(data)))
    prefix=name.." "
  else
    table.insert(out, string.format("# name= nil, type= %s\n", type(data)))
  end
  if type(data) == "table" then
    if data.size then
      for i=0,data:size()-1 do
	table.insert(out,
		     string.format("# %s[%s]={\n",
				   prefix,
				   table.concat(data:compute_coords(i), ",")))
	table.insert(out,process_element(data:raw_get(i)).."\n")
	table.insert(out, "# }\n")
      end
    else
      for name,elem in pairs(data) do
	table.insert(out, string.format("# %s.%s=\n", prefix, name))
	table.insert(out, process_element(elem).."\n")
      end
    end
  else
    if type(data) == "matrixChar" then
      local t = data:to_string_table()
      table.insert(out, table.concat(t, " "))
    else
      table.insert(out, tostring(data))
    end
  end
  return table.concat(out, "")
end

-------------------
-- MAIN FUNCTION --
-------------------

-- reads a MAT file and returns a Lua table with all the elements, indexed by
-- its names, and recursively containing matrix, cell_array or struct objects
-- with all the data
function matlab.read(path,col_major)
  assert(path and type(path)=="string", "First argument must be a path string")
  local col_major = col_major or false
  local reader    = matlab.reader(path)
  local elements  = {}
  for e in reader:elements() do
    local lua_element,name = matlab.tolua(e,col_major)
    elements[name] = lua_element
  end
  setmetatable(elements,
	       {
		 __tostring = function(self)
		   local out = {}
		   for name,data in pairs(self) do
		     table.insert(out,process_element(data,name))
		   end
		   return table.concat(out, "")
		 end
	       })
  return elements
end
