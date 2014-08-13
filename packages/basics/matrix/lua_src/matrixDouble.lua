-- OVERWRITTING TOSTRING FUNCTION
class.extend(matrixDouble, "to_lua_string",
             function(self, format)
               return string.format("matrixDouble.fromString[[%s]]",
                                    self:toString(format or "string"))
end)

-- serialization
matrix.__make_all_serialization_methods__(matrixDouble)

matrixDouble.meta_instance.__call =
  matrix.__make_generic_call__()

matrixDouble.meta_instance.__tostring =
  matrix.__make_generic_print__("MatrixDouble",
				function(value)
				  return string.format("% -11.6g", value)
  end)

matrixDouble.join =
  matrix.__make_generic_join__(matrixDouble)
