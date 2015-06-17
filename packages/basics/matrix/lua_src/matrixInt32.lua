class.extend(matrixInt32, "t", matrixInt32.."transpose")

class.extend(matrixInt32, "flatten",
             function(self)
               return self:rewrap(self:size())
end)

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
