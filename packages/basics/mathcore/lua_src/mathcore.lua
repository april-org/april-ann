local MAX = 4
local DEFAULT_SIZE = 8

-- global NaN and inf definition
nan = mathcore.limits.double.quiet_NaN()
inf = mathcore.limits.double.infinity()

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

local function len_function(obj)
  return obj:size()
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

mathcore.block.float.meta_instance.__len = len_function
mathcore.block.double.meta_instance.__len = len_function
mathcore.block.int32.meta_instance.__len = len_function
mathcore.block.complex.meta_instance.__len = len_function
mathcore.block.bool.meta_instance.__len = len_function

------------------------------------------------------------------------------

local vector,vector_methods = class("mathcore.vector")
mathcore.vector = vector -- global definition

local block_to_dtype ={
  [mathcore.block.float]   = "float",
  [mathcore.block.double]  = "double",
  [mathcore.block.complex] = "complex",
  [mathcore.block.int32]   = "int32",
  [mathcore.block.bool]    = "bool",
  [mathcore.block.char]    = "char",
}

vector.constructor =
  april_doc{
    class="method",
    summary="Constructor of dynamic vector using underlying mathcore.block data",
    params={
      "A mathcore.block instance",
    },
    outputs={
      "A dynamic vector instance",
    },
  } ..
  april_doc{
    class="method",
    summary="Constructor of dynamic vector using underlying mathcore.block data",
    params={
      dtype="Data type: float, double, complex, int32, bool [optional], by default it is float",
      reserve="Size reserved in the underlying memory block [optional]",
      size="Initial size of the vector [optional], by default it is 0",
    },
    outputs={
      "A dynamic vector instance",
    },
  } ..
  function(self, params)
  params = params or {}
  if type(params) ~= "table" then
    local block = params
    self.dtype  = block_to_dtype[class.of(block)]
    self.ctor   = class.of(block)
    self.len    = #block
    self.block  = block
  else
    local params = get_table_fields({
        dtype   = { type_match="string", default="float" },
        reserve = { type_match="number", default=DEFAULT_SIZE },
        size    = { type_match="number", default=0 },
                                    }, params)
    local dtype   = params.dtype
    local reserve = params.reserve
    local size    = params.size
    self.dtype  = dtype
    self.ctor   = assert(mathcore.block[dtype], "Incorrect block type")
    self.len    = size
    self.block  = self.ctor(math.max(reserve, size))
  end
end

vector_methods.resize = function(self, size)
  self.block = self.ctor(size):copy(self.block)
  collectgarbage("collect")
  return self
end

vector_methods.reserve = function(self, size)
  if #self.block < size then self:resize(size) end
  return self
end

vector_methods.push_back = function(self, value)
  if self.len == #self.block then self:resize(#self.block * 2) end
  self.block[self.len] = value
  self.len = self.len + 1
end

-- not guaranteed to return the underlying block, it can be a copy
vector_methods.to_block =
  april_doc{
    class="method",
    summary="Returns a mathcore.block with the data",
    description="Be careful, this function doesn't guarantee to return a new allocated block",
    outputs={
      "A mathcore.block instance",
    },
  } ..
  function(self)
    local block = self.block
    if self.len < #block then block = self.ctor(self.len):copy(block) end
    return block
  end

class.declare_functional_index(vector,
                               function(self, key)
                                 if type(key) == "number" then
                                   return self.block[key-1]
                                 end
end)

class.extend_metamethod(vector, "__newindex",
                        function(self, key, value)
                          if type(key) == "number" then
                            self.block[key-1] = value
                          else
                            rawset(self, key, value)
                          end
end)

class.extend_metamethod(vector, "__len", function(self) return self.len end)
class.extend_metamethod(vector, "__ipairs",
                        function(self)
                          return function(self,i)
                            if i < #self then
                              i=i+1
                              return i,self[i]
                            end
                          end,self,0
end)

class.extend_metamethod(vector, "__tostring",
                        function(self) return tostring(self:to_block()) end)
