class.extend(matrix, "t", matrixInt32.."transpose")

-- serialization
matrix.__generic__.__make_all_serialization_methods__(matrixInt32)

matrixInt32.meta_instance.__call =
  matrix.__generic__.__make_generic_call__()

matrixInt32.meta_instance.__newindex =
  matrix.__generic__.__make_generic_newindex__(matrixInt32)

matrixInt32.meta_instance.__tostring =
  matrix.__generic__.__make_generic_print__("MatrixInt32",
                                            function(value)
                                              return string.format("% 11d", value)
  end)

matrixInt32.join =
  matrix.__generic__.__make_generic_join__(matrixInt32)

matrixInt32.meta_instance.__eq = function(op1, op2)
  if type(op1) == "number" or type(op2) == "number" then return false end
  local d1,d2 = op1:dim(),op2:dim()
  if #d1 ~= #d2 then return false end
  local eq_size = iterator.zip(iterator(ipairs(d1)):select(2),
                               iterator(ipairs(d2)):select(2)):
  reduce(function(acc,a,b) return acc and (a==b) end, true)
  if not eq_size then return false end
  local eq = true
  op1:map(op2, function(x,y) eq = eq and (x==y) end)
  return eq
end
