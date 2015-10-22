class.extend(matrix.sparse, "t", matrix.sparse .. "transpose")

-- the constructor
matrix.sparse.csr = matrix.sparse

-- serialization
matrix.__generic__.__make_generic_fromFilename__(matrix.sparse)
matrix.__generic__.__make_generic_fromString__(matrix.sparse)
matrix.__generic__.__make_generic_to_lua_string__(matrix.sparse)
matrix.__generic__.__make_generic_toFilename__(matrix.sparse)
matrix.__generic__.__make_generic_toString__(matrix.sparse)

local function parse_slice(slice, max)
  if not slice then return 1,max end

  local a,b
  local tt = type(slice)
  if tt == "string" then
    a = slice:match("^(%d+)%:.*$") or 1
    b = slice:match("^.*%:(%d+)$") or max
  elseif tt == "table" then
    a = slice[1] or 1
    b = slice[2] or max
  elseif tt == "number" then
    a,b = slice,slice
  else
    error("Incorrect slice format, expecting a string, a table or a number")
  end

  a = tonumber(a)
  b = tonumber(b)
  
  if a < 0 then a = max + a end
  if b < 0 then b = max + b end
  
  assert(a >= 1 and a <= max)
  assert(b >= 1 and b <= max)
  assert(a <= b)

  return a,b
end

matrix.sparse.meta_instance.__call = function(self, x_slice, y_slice)
  local x_a,x_b = parse_slice(x_slice, self:dim(1))
  local y_a,y_b = parse_slice(y_slice, self:dim(2))
  local coords = { x_a, y_a }
  local sizes  = { x_b - x_a + 1, y_b - y_a + 1 }
  return self:slice(coords, sizes, false)
end

-- define right side operator []
class.declare_functional_index
(
  matrix.sparse,
  function(self, key)
    local tt = type(key)
    if tt == "number" then
      return self:select(1, key)
    elseif tt == "table" then
      return self(key[1], key[2])
    end
  end
)

matrix.sparse.meta_instance.__newindex = function(self, key, value)
  error("Not implemented for sparse matrix instance")
end

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
