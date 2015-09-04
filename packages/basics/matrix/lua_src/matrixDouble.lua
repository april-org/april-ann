class.extend_metamethod(matrixDouble, "__len", function(self) return self:dim(1) end)
class.extend_metamethod(matrixDouble, "__ipairs",
                        function(self)
                          return function(self,i)
                            i = i+1
                            if i <= #self then return i,self[i] end
                          end, self, 0
end)

class.extend(matrixDouble, "t", matrixDouble.."transpose")

class.extend(matrixDouble, "flatten",
             function(self)
               return self:rewrap(self:size())
end)

matrix.__generic__.__make_index_methods__(matrixDouble)

-- serialization
matrix.__generic__.__make_all_serialization_methods__(matrixDouble)

matrixDouble.meta_instance.__call =
  matrix.__generic__.__make_generic_call__()

matrixDouble.meta_instance.__newindex =
  matrix.__generic__.__make_generic_newindex__(matrixDouble)

matrix.__generic__.__make_generic_index__(matrixDouble)

matrixDouble.meta_instance.__tostring =
  matrix.__generic__.__make_generic_print__("MatrixDouble",
                                            function(value)
                                              return string.format("% -15.6g", value)
  end)

matrixDouble.join =
  matrix.__generic__.__make_generic_join__(matrixDouble)

matrixDouble.meta_instance.__eq = function(op1, op2)
  if type(op1) == "number" or type(op2) == "number" then return false end
  return op1:equals(op2)
end

matrixDouble.meta_instance.__add = function(op1, op2)
  if not class.is_a(op1,matrixDouble) then op1,op2=op2,op1 end
  if type(op2) == "number" then
    return op1:clone():scalar_add(op2)
  else
    return op1:add(op2)
  end
end

matrixDouble.meta_instance.__sub = function(op1, op2)
  if class.is_a(op1,matrixDouble) and class.is_a(op2,matrixDouble) then
    return op1:sub(op2)
  elseif class.is_a(op1,matrixDouble) then
    return op1:clone():scalar_add(-op2)
  elseif class.is_a(op2,matrixDouble) then
    return op2:clone():scal(-1):scalar_add(op1)
  end
end

matrixDouble.meta_instance.__mul = function(op1, op2)
  if class.is_a(op1,matrixDouble.sparse) or class.is_a(op2,matrixDouble.sparse) then
    if class.is_a(op2,matrixDouble.sparse) then
      local res = matrixDouble(op1:dim(1),op2:dim(2))
      res:transpose():sparse_mm{ alpha=1.0, beta=0.0, A=op2, B=op1,
                                 trans_A=true, trans_B=true }
      return res
    else
      local res = matrixDouble(op1:dim(1),op2:dim(2))
      res:sparse_mm{ alpha=1.0, beta=0.0, A=op1, B=op2 }
      return res
    end
  else
    if not class.is_a(op1,matrixDouble) then op1,op2=op2,op1 end
    if type(op2) == "number" then return op1:clone():scal(op2)
    else return op1:mul(op2)
    end
  end
end

matrixDouble.meta_instance.__pow = function(op1, op2)
  local new_mat = op1:clone()
  return new_mat:pow(op2)
end

matrixDouble.meta_instance.__div = function(op1, op2)
  if type(op2) == "number" then
    local new_mat = op1:clone()
    return new_mat:scal(1/op2)
  elseif type(op1) == "number" then
    local new_mat = op2:clone()
    return new_mat:div(op1)
  else
    assert(class.is_a(op1,matrixDouble) and class.is_a(op2,matrixDouble),
	   "Expected a matrixDouble and a number or two matrices")
    local new_mat1 = op1:clone()
    local new_mat2 = op2:clone():div(1)
    return new_mat1:cmul(new_mat2)
  end
end

matrixDouble.meta_instance.__unm = function(op)
  local new_mat = op:clone()
  return new_mat:scal(-1)
end
