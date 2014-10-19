-- GENERIC __CALL METAMETHOD
matrix.__generic__ = matrix.__generic__ or {}

matrix.__generic__.__make_generic_call__ = function()
  return function(self,...)
    if ... == nil then return self end
    print("EO")
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
      elseif tt == "string" then
	a = t:match("^(%d+)%:.*$") or 1
	b = t:match("^.*%:(%d+)$") or dims[i]
      elseif tt == "number" then
	a = t
	b = a
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

matrix.__generic__.__make_generic_newindex__ = function(matrix_class)
  assert(matrix_class and class.is_class(matrix_class),
         "Needs a class table as argument")
  return function(self,key,value)
    assert(type(key) == "table", "Needs a table as key")
    local m  = self(table.unpack(key))
    local tv = type(value)
    if tv == "number" or tv == "complex" then
      m:fill(value)
    else
      assert(class.is_a(m, matrix_class), "Needs a number or a matrix as value")
      m:squeeze():copy(value:squeeze())
    end
  end
end
