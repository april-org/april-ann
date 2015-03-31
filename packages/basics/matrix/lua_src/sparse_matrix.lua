class.extend(matrix.sparse, "t", matrix.sparse .. "transpose")

-- the constructor
matrix.sparse.csr = function(...)
  return matrix.sparse(...)
end

-- serialization
matrix.__generic__.__make_generic_fromFilename__(matrix.sparse)
matrix.__generic__.__make_generic_fromString__(matrix.sparse)
matrix.__generic__.__make_generic_to_lua_string__(matrix.sparse)
matrix.__generic__.__make_generic_toFilename__(matrix.sparse)
matrix.__generic__.__make_generic_toString__(matrix.sparse)

matrix.sparse.meta_instance.__call =
  matrix.__generic__.__make_generic_call__()

-- define right side operator []
matrix.__generic__.__make_generic_index__(matrix.sparse)

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
  table.insert(out, string.format("# SparseMatrix of size [%s] in %s [%s], %d non-zeros\n",
				  table.concat(dims, ","), sparse,
				  self:get_reference_string(),
				  self:non_zero_size()))
  return table.concat(out, "\n")
end

matrix.sparse.meta_instance.__eq = function(op1, op2)
  if type(op1) == "number" or type(op2) == "number" then return false end
  return op1:equals(op2)
end

matrix.sparse.meta_instance.__mul = function(op1, op2)
  if type(op2) == "number" then return op1:clone():scal(op2)
  elseif type(op1) == "number" then return op2:clone():scal(op1)
  else
    if class.is_a(op1,matrix) then
      local res = matrix(op1:dim(1),op2:dim(2))
      res:transpose():sparse_mm{ alpha=1.0, beta=0.0, A=op2, B=op1,
                                 trans_A=true, trans_B=true }
      return res
    elseif class.is_a(op2,matrix) then
      local res = matrix(op1:dim(1),op2:dim(2))
      res:sparse_mm{ alpha=1.0, beta=0.0, A=op1, B=op2 }
      return res
    else
      error("matrix.sparse only could be multiplied by scalars or matrix")
    end
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
    error("matrix.sparse only could be divided by scalars")
  end
end

matrix.sparse.meta_instance.__unm = function(op)
  local new_mat = op:clone()
  return new_mat:scal(-1)
end
