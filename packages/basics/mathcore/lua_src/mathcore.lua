-- global NaN and inf definition
nan = mathcore.limits.double.quiet_NaN()
inf = mathcore.limits.double.infinity()

local MAX = 4
local function make_block_tostring(name, str)
  return function(b)
    local result = {}
    local data = {}
    for i=1,math.min(MAX, b:size()) do
      data[i] = str(b:raw_get(i-1))
    end
    if b:size() > MAX then
      data[#data+1] = "..."
      data[#data+1] = str(b:raw_get(b:size()-1))
    end
    result[1] = table.concat(data, " ")
    result[2] = "# %s size %d [%s]"%{name, b:size(), b:get_reference_string()}
    return table.concat(result, "\n")
  end
end

local function make_serialize(cls)
  cls.meta_instance.serialize = function(self,stream)
    if not stream then
      local result = self:write()
      return table.concat{ self:ctor_name(), "[[", result, "]]" }
    else
      stream:write(self:ctor_name())
      stream:write("[[")
      self:write(stream)
      stream:write("]]")
    end
  end
end

local function make_index_function(cls)
  class.declare_functional_index(cls,
                                 function(obj,key)
                                   if type(key) == "number" then
                                     return obj(key)
                                   end
  end)
end

local function new_index_function(obj, key, value)
  obj:raw_set(key, value)
end

local function call_function(obj, ...)
  return obj:raw_get(...)
end

mathcore.block.float.meta_instance.__tostring =
  make_block_tostring("Float block",
                      function(value)
                        return string.format("% -13.6g", value)
  end)

mathcore.block.double.meta_instance.__tostring =
  make_block_tostring("Double block",
                      function(value)
                        return string.format("% -15.6g", value)
  end)

mathcore.block.int32.meta_instance.__tostring =
  make_block_tostring("Int32 block",
                      function(value)
                        return string.format("% 11d", value)
  end)

mathcore.block.complex.meta_instance.__tostring =
  make_block_tostring("Complex block",
                      function(value)
                        return string.format("%26s", tostring(value))
  end)

mathcore.block.bool.meta_instance.__tostring =
  make_block_tostring("Bool block",
                      function(value)
                        return string.format("%s", value and "T" or "F")
  end)

make_serialize(mathcore.block.float)
make_serialize(mathcore.block.double)
make_serialize(mathcore.block.int32)
make_serialize(mathcore.block.complex)
make_serialize(mathcore.block.bool)

make_index_function(mathcore.block.float)
make_index_function(mathcore.block.double)
make_index_function(mathcore.block.int32)
make_index_function(mathcore.block.complex)
make_index_function(mathcore.block.bool)

mathcore.block.float.meta_instance.__newindex = new_index_function
mathcore.block.double.meta_instance.__newindex = new_index_function
mathcore.block.int32.meta_instance.__newindex = new_index_function
mathcore.block.complex.meta_instance.__newindex = new_index_function
mathcore.block.bool.meta_instance.__newindex = new_index_function

mathcore.block.float.meta_instance.__call = call_function
mathcore.block.double.meta_instance.__call = call_function
mathcore.block.int32.meta_instance.__call = call_function
mathcore.block.complex.meta_instance.__call = call_function
mathcore.block.bool.meta_instance.__call = call_function
