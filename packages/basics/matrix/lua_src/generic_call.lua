-- GENERIC __CALL METAMETHOD
matrix.__make_generic_call__ = function()
  return function(self,...)
    local arg      = table.pack(...)
    local dims     = self:dim()
    local pos,size = {},{}
    for i=1,#arg do
      local t = arg[i]
      local tt = luatype(t)
      local a,b
      if tt == "table" then
	a,b = table.unpack(t)
	april_assert(tonumber(a) and tonumber(b),
		     "The table for component %d must contain two numbers",i)
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
