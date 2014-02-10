-- OVERWRITTING TOSTRING FUNCTION
class_extension(matrixInt32, "to_lua_string",
                function(self)
                  return string.format("matrixInt32.fromString[[%s]]",
                                       self:toString())
                end)

matrixInt32.meta_instance.__call =
  matrix.__make_generic_call__()

matrixInt32.meta_instance.__tostring =
  matrix.__make_generic_print__("MatrixInt32",
				function(value)
				  return string.format("% 11d", value)
				end)

matrixInt32.join =
  matrix.__make_generic_join__(matrixInt32)
