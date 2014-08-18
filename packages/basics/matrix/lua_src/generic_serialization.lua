local function archive_wrapper(s)
  if class.is_a(s, aprilio.package) then
    assert(s:number_of_files() == 1, "Expected only 1 file in the package")
    return s:open(1)
  else
    return s
  end  
end

matrix.__generic__ = matrix.__generic__ or {}

matrix.__generic__.__make_generic_to_lua_string__ = function(matrix_class)
  local name = matrix_class.meta_instance.id
  return function(self, format)
    return string.format("%s.fromString[[%s]]",
                         name, self:toString(format or "binary"))
  end
end

-- GENERIC FROM FILENAME
matrix.__generic__.__make_generic_fromFilename__ = function(matrix_class)
  return function(filename,order)
    return matrix_class.read(archive_wrapper( io.open(filename) ),
                             { order=order })
  end
end

-- GENERIC FROM TAB FILENAME
matrix.__generic__.__make_generic_fromTabFilename__ = function(matrix_class)
  return function(filename,order)
    return matrix_class.read(archive_wrapper( io.open(filename) ),
                             { order = order, tab = true })
  end
end

matrix.__generic__.__make_generic_fromCSVFilename__ = function(matrix_class)
  return function(filename,args)
    local args = get_table_fields({
        order  = { mandatory=false, type_match="string" },
        delim  = { mandatory=true, type_match="string", default="," },
        default= { mandatory=false } }, args)
    args.keep_delim = true
    args.tab = true
    return matrix_class.read(archive_wrapper( io.open(filename) ), args)
  end
end

-- GENERIC FROM STRING
matrix.__generic__.__make_generic_fromString__ = function(matrix_class)
  return function(str)
    return matrix_class.read(aprilio.stream.input_lua_string(str))
  end
end

-- GENERIC TO FILENAME
matrix.__generic__.__make_generic_toFilename__ = function(matrix_class)
  return function(self,filename,mode)
    return class.consult(matrix_class,"write")(self,io.open(filename,"w"),
                                               { ascii = (mode=="ascii") })
  end
end

-- GENERIC TO TAB FILENAME
matrix.__generic__.__make_generic_toTabFilename__ = function(matrix_class)
  return function(self,filename,mode)
    return class.consult(matrix_class,"write")(self,io.open(filename,"w"),
                                               { ascii = (mode=="ascii"),
                                                 tab   = true })
  end
end

-- GENERIC TO STRING
matrix.__generic__.__make_generic_toString__ = function(matrix_class)
  return function(self,mode)
    return class.consult(matrix_class,"write")(self,{ ascii = (mode=="ascii") })
  end
end

function matrix.__generic__.__make_all_serialization_methods__(matrix_class)
  matrix_class.fromFilename    = matrix.__generic__.__make_generic_fromFilename__(matrix_class)
  matrix_class.fromTabFilename = matrix.__generic__.__make_generic_fromTabFilename__(matrix_class)
  matrix_class.fromString      = matrix.__generic__.__make_generic_fromString__(matrix_class)
  matrix_class.fromCSVFilename = matrix.__generic__.__make_generic_fromCSVFilename__(matrix_class)
  class.extend(matrix_class, "toFilename",
               matrix.__generic__.__make_generic_toFilename__(matrix_class))
  class.extend(matrix_class, "toTabFilename",
               matrix.__generic__.__make_generic_toTabFilename__(matrix_class))
  class.extend(matrix_class, "toString",
               matrix.__generic__.__make_generic_toString__(matrix_class))
  class.extend(matrix_class, "to_lua_string",
               matrix.__generic__.__make_generic_to_lua_string__(matrix_class))
end
