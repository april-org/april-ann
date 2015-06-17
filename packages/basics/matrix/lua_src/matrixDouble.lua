class.extend(matrixDouble, "t", matrixDouble.."transpose")

class.extend(matrixDouble, "flatten",
             function(self)
               return self:rewrap(self:size())
end)

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
