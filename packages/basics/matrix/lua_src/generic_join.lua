-- GENERIC JOIN FUNCTION
matrix.__generic__ = matrix.__generic__ or {}

matrix.__generic__.__make_generic_join__ = function(constructor)
  return function(dim, ...)
    local arg  = table.pack(...)
    if type(arg[1]) == "table" then
      assert(#arg == 1, "Use one table argument or multiple matrix arguments")
      arg = arg[1]
    end
    assert(#arg > 0, "At least one matrix is needed")
    local size = arg[1]:dim()
    -- ERROR CHECK
    assert(dim >= 1 and dim <= #size,
	   "Incorrect given dimension number, or incorrect first matrix size")
    size[dim] = 0
    for i=1,#arg do
      local m = arg[i]
      local d = m:dim()
      assert(#d == #size,
	     "All the matrices must have the same number of dimensions")
      size[dim] = size[dim] + d[dim]
      for j=1,dim-1 do
	assert(size[j] == d[j], "Incorrect dimension size")
      end
      for j=dim+1,#size do
	assert(size[j] == d[j], "Incorrect dimension size")
      end
    end
    -- JOIN
    local outm
    if arg[1].get_major_order and arg[1]:get_major_order() == "col_major" then
      outm = constructor.col_major(table.unpack(size))
    else
      outm = constructor(table.unpack(size))
    end
    local first = matrix(#size):ones():toTable()
    for i=1,#arg do
      local m = arg[i]
      local d = m:dim()
      size[dim] = d[dim]
      local outm_slice = outm:slice(first, size)
      outm_slice:copy(m)
      first[dim] = first[dim] + size[dim]
    end
    return outm
  end
end
