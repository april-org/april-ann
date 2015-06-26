class.extend(matrixInt32, "t", matrixInt32.."transpose")

class.extend(matrixInt32, "flatten",
             function(self)
               return self:rewrap(self:size())
end)

matrix.__generic__.__make_index_methods__(matrixInt32)

-- serialization
matrix.__generic__.__make_all_serialization_methods__(matrixInt32)

matrixInt32.meta_instance.__call =
  matrix.__generic__.__make_generic_call__()

matrixInt32.meta_instance.__newindex =
  matrix.__generic__.__make_generic_newindex__(matrixInt32)

matrix.__generic__.__make_generic_index__(matrixInt32)

matrixInt32.meta_instance.__tostring =
  matrix.__generic__.__make_generic_print__("MatrixInt32",
                                            function(value)
                                              return string.format("% 11d", value)
  end)

matrixInt32.join =
  matrix.__generic__.__make_generic_join__(matrixInt32)

matrixInt32.meta_instance.__eq = function(op1, op2)
  if type(op1) == "number" or type(op2) == "number" then return false end
  return op1:equals(op2)
end

matrixInt32.meta_instance.__add = function(op1, op2)
  if not class.is_a(op1,matrixInt32) then op1,op2=op2,op1 end
  if type(op2) == "number" then
    return op1:clone():scalar_add(op2)
  else
    return op1:add(op2)
  end
end

matrixInt32.meta_instance.__sub = function(op1, op2)
  if class.is_a(op1,matrixInt32) and class.is_a(op2,matrixInt32) then
    return op1:sub(op2)
  elseif class.is_a(op1,matrixInt32) then
    return op1:clone():scalar_add(-op2)
  elseif class.is_a(op2,matrixInt32) then
    return op2:clone():scal(-1):scalar_add(op1)
  end
end

matrixInt32.meta_instance.__mul = function(op1, op2)
  if not class.is_a(op1,matrixInt32) then op1,op2=op2,op1 end
  if type(op2) == "number" then return op1:clone():scal(op2)
  else
    error("Not implemented matrix multiplication between matrixInt32 objects")
  end
end

matrixInt32.meta_instance.__div = function(op1, op2)
  if type(op2) == "number" then
    local new_mat = op1:clone()
    return new_mat:idiv(op2)
  elseif type(op1) == "number" then
    error("Unable to divide a number by a matrixInt32")
  else
    assert(class.is_a(op1,matrixInt32) and class.is_a(op2,matrixInt32),
	   "Expected a matrixInt32 and a number or two matrices")
    return new_mat1:clone():idiv(new_mat2)
  end
end

matrixInt32.meta_instance.__unm = function(op)
  local new_mat = op:clone()
  return new_mat:scal(-1)
end
