--class_extension(matrix, "to_lua_string",
--                function(self, format)
--                  return string.format("matrix.sparse.fromString[[%s]]",
--                                       self:toString(format or "binary"))
--                end)

-- the constructor
matrix.sparse.csr = function(...)
  return matrix.sparse(...)
end

matrix.sparse.meta_instance.__call =
  matrix.__make_generic_call__()

matrix.sparse.meta_instance.__tostring = function(self)
  local out      = {}
  local sparse   = (self.get_sparse_format and self:get_sparse_format()) or "csr"
  local dims     = self:dim()
  local so_large = false
  local getter   = function(value) return string.format("% -11.6g", value) end
  --
  for i=1,#dims do 
    if dims[i] > 20 then so_large = true end
  end
  if not so_large then  
    local aux    = self:to_dense()
    local coords = {1,1}
    local row    = {}
    for i=1,aux:size() do
      table.insert(row, getter(aux:get(table.unpack(coords))))
      local j=#dims+1
      repeat
	j=j-1
	coords[j] = coords[j] + 1
	if coords[j] > dims[j] then coords[j] = 1 end
      until j==1 or coords[j] ~= 1
      if coords[#coords] == 1 then
	table.insert(out, table.concat(row, " ")) row={}
      end
    end
  else
    table.insert(out, "Large matrix, not printed to display")
  end
  table.insert(out, string.format("# SparseMatrix of size [%s] in %s [%s], %d non-zeros",
				  table.concat(dims, ","), sparse,
				  self:get_reference_string(),
				  self:non_zero_size()))
  return table.concat(out, "\n")
end

matrix.sparse.meta_instance.__eq = function(op1, op2)
  if type(op1) == "number" or type(op2) == "number" then return false end
  return op1:equals(op2)
end

matrix.sparse.meta_instance.__add = function(op1, op2)
  if not isa(op1,matrix.sparse) then op1,op2=op2,op1 end
  if type(op2) == "number" then
    return op1:clone():scalar_add(op2)
  else
    return op1:add(op2)
  end
end

matrix.sparse.meta_instance.__sub = function(op1, op2)
  if isa(op1,matrix.sparse) and isa(op2,matrix.sparse) then
    return op1:sub(op2)
  elseif isa(op1,matrix.sparse) then
    return op1:clone():scalar_add(-op2)
  elseif isa(op2,matrix.sparse) then
    return op2:clone():scal(-1):scalar_add(op1)
  end
end

matrix.sparse.meta_instance.__mul = function(op1, op2)
  if not isa(op1,matrix.sparse) then op1,op2=op2,op1 end
  if type(op2) == "number" then return op1:clone():scal(op2)
  else return op1:mul(op2)
  end
end

matrix.sparse.meta_instance.__pow = function(op1, op2)
  local new_mat = op1:clone()
  return new_mat:pow(op2)
end

matrix.sparse.meta_instance.__div = function(op1, op2)
  if type(op2) == "number" then
    local new_mat = op1:clone()
    return new_mat:scal(1/op2)
  elseif type(op1) == "number" then
    local new_mat = op2:clone()
    return new_mat:div(op1)
  else
    assert(isa(op1,matrix.sparse) and isa(op2,matrix.sparse),
	   "Expected a matrix and a number or two matrices")
    local new_mat1 = op1:clone()
    local new_mat2 = op2:clone():div(1)
    return new_mat1:axpy(1.0, new_mat2)
  end
end

matrix.sparse.meta_instance.__unm = function(op)
  local new_mat = op:clone()
  return new_mat:scal(-1)
end
