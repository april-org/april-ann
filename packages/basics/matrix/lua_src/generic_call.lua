-- GENERIC __CALL METAMETHOD
matrix.__generic__ = matrix.__generic__ or {}

matrix.__generic__.__make_generic_call__ = function()
  return function(self,...)
    if ... == nil then return self end
    local arg      = table.pack(...)
    local dims     = self:dim()
    local pos,size = {},{}
    for i=1,#arg do
      local t = arg[i]
      local tt = luatype(t)
      local a,b
      if tt == "table" then
	a,b = table.unpack(t)
        if not a and not b then
          a,b = 1,dims[i]
        else
          april_assert(tonumber(a) and tonumber(b),
                       "The table for component %d must contain two numbers or none",i)
        end
      elseif tt == "number" or tonumber(t) then
	a = t
	b = a
      elseif tt == "string" then
	a = t:match("^(%d+)%:.*$") or 1
	b = t:match("^.*%:(%d+)$") or dims[i]
      else
	error("The argument %d is not a table neither a string" % {i})
      end
      a,b = tonumber(a),tonumber(b)
      april_assert(1 <= a and a <= dims[i],
		   "Range %d out of bounds for dim %d", a, i)
      april_assert(1 <= b and b <= dims[i],
		   "Range %d out of bounds for dim %d", b, i)
      table.insert(pos,  a)
      table.insert(size, b - a + 1)
    end
    for i=#arg+1,#dims do
      table.insert(pos, 1)
      table.insert(size, dims[i])
    end
    return self:slice(pos,size)
  end
end

matrix.__generic__.__make_generic_index__ = function(matrix_class)
  class.declare_functional_index(matrix_class,
                                 function(self,key)
                                   local tt = type(key)
                                   if tt == "number" then
                                     if self:num_dim() > 1 then
                                       return self:select(1, key)
                                     else
                                       return self:get(key)
                                     end
                                   elseif tt == "table" then
                                     return self(table.unpack(key))
                                   end
  end)
end

matrix.__generic__.__make_generic_newindex__ = function(matrix_class)
  assert(matrix_class and class.is_class(matrix_class),
         "Needs a class table as argument")
  return function(self,key,value)
    local tt = type(key)
    if tt == "number" then
      self = (self:num_dim() > 1) and self:select(1, key) or self(key)
      key = {}
    else
      assert(tt == "table", "Needs a table as key")
    end
    local m  = self(table.unpack(key))
    local tv = type(value)
    if tv == "number" or tv == "complex" or tb == "boolean" then
      m:fill(value)
    else
      assert(class.is_a(m, matrix_class), "Needs a number or a matrix as value")
      m:copy(value)
    end
  end
end
