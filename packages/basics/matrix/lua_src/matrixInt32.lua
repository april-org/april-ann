-- serialization
matrix.__generic__.__make_all_serialization_methods__(matrixInt32)

matrixInt32.meta_instance.__call =
  matrix.__generic__.__make_generic_call__()

matrix.meta_instance.__newindex =
  matrix.__generic__.__make_generic_newindex__(matrixInt32)

matrixInt32.meta_instance.__tostring =
  matrix.__generic__.__make_generic_print__("MatrixInt32",
                                            function(value)
                                              return string.format("% 11d", value)
  end)

matrixInt32.join =
  matrix.__generic__.__make_generic_join__(matrixInt32)
