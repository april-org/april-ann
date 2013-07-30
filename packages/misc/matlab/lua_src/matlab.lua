matlab = matlab or {}

local tolua_table = {
  [0] = function(e) return e end,
}

local function add_wrapper(class_number, f)
  tolua_table[class_number] = f
end

function matlab.tolua(element)
  return tolua_table[element:get_class()](element)
end

local function tomatrix(e) return e:get_matrix() end
local function tocellarray(e) return e:get_cell_array() end

add_wrapper(matlab.classes.double, tomatrix)
add_wrapper(matlab.classes.single, tomatrix)
add_wrapper(matlab.classes.int8,   tomatrix)
add_wrapper(matlab.classes.uint8,  tomatrix)
add_wrapper(matlab.classes.int16,  tomatrix)
add_wrapper(matlab.classes.uint16, tomatrix)
add_wrapper(matlab.classes.int32,  tomatrix)
add_wrapper(matlab.classes.uint32, tomatrix)

add_wrapper(matlab.classes.cell_array, tocellarray)
