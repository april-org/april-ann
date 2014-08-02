-- GENERIC FROM FILENAME
matrix.__make_generic_fromFilename__ = function(matrix_class)
  return function(filename)
    return matrix_class.fromFileStream(april_io.stream.c_file(filename, "r"))
  end
end

-- GENERIC FROM TAB FILENAME
matrix.__make_generic_fromTabFilename__ = function(matrix_class)
  return function(filename)
    return matrix_class.fromTabFileStream(april_io.stream.c_file(filename, "r"))
  end
end

-- GENERIC FROM STRING
matrix.__make_generic_fromString__ = function(matrix_class)
  return function(str)
    return matrix_class.fromFileStream(april_io.stream.input_lua_buffer(str))
  end
end

-- GENERIC TO FILENAME
matrix.__make_generic_toFilename__ = function(matrix_class)
  return function(filename)
    return matrix_class.toFileStream(april_io.stream.c_file(filename, "w"))
  end
end

-- GENERIC TO TAB FILENAME
matrix.__make_generic_toTabFilename__ = function(matrix_class)
  return function(filename)
    return matrix_class.toTabFileStream(april_io.stream.c_file(filename, "w"))
  end
end

-- GENERIC TO STRING
matrix.__make_generic_toString__ = function(matrix_class)
  return function(str)
    return matrix_class.toFileStream(april_io.stream.output_lua_buffer(str))
  end
end
