local function archive_wrapper(s)
  if class.is_a(s, aprilio.package) then
    assert(s:number_of_files() == 1, "Expected only 1 file in the package")
    return s:open(1)
  else
    return s
  end  
end

-- GENERIC FROM FILENAME
matrix.__make_generic_fromFilename__ = function(matrix_class)
  return function(filename,...)
    return matrix_class.read(archive_wrapper( io.open(filename) ),...)
  end
end

-- GENERIC FROM TAB FILENAME
matrix.__make_generic_fromTabFilename__ = function(matrix_class)
  return function(filename,...)
    return matrix_class.readTab(archive_wrapper( io.open(filename) ),...)
  end
end

-- GENERIC FROM STRING
matrix.__make_generic_fromString__ = function(matrix_class)
  return function(str,...)
    return matrix_class.read(aprilio.stream.input_lua_string(str),...)
  end
end

-- GENERIC TO FILENAME
matrix.__make_generic_toFilename__ = function(matrix_class)
  return function(self,filename,...)
    return class.consult(matrix_class,"write")(self,io.open(filename,"w"),...)
  end
end

-- GENERIC TO TAB FILENAME
matrix.__make_generic_toTabFilename__ = function(matrix_class)
  return function(self,filename,...)
    return class.consult(matrix_class,"writeTab")(self,io.open(filename,"w"),...)
  end
end

-- GENERIC TO STRING
matrix.__make_generic_toString__ = function(matrix_class)
  return function(self,...)
    local stream = aprilio.stream.output_lua_string(str)
    class.consult(matrix_class,"write")(self,stream,...)
    return stream:value()
  end
end

function matrix.__make_all_serialization_methods__(matrix_class)
  matrix_class.fromFilename    = matrix.__make_generic_fromFilename__(matrix_class)
  matrix_class.fromTabFilename = matrix.__make_generic_fromTabFilename__(matrix_class)
  matrix_class.fromString      = matrix.__make_generic_fromString__(matrix_class)
  class.extend(matrix_class, "toFilename",
               matrix.__make_generic_toFilename__(matrix_class))
  class.extend(matrix_class, "toTabFilename",
               matrix.__make_generic_toTabFilename__(matrix_class))
  class.extend(matrix_class, "toString",
               matrix.__make_generic_toString__(matrix_class))
end
