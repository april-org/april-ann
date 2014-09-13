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
  class.extend(matrix_class, "to_lua_string",
               function(self, format)
                 return string.format("%s.fromString[[%s]]",
                                      name, self:toString(format or "binary"))
  end)
end

-- GENERIC FROM FILENAME
matrix.__generic__.__make_generic_fromFilename__ = function(matrix_class)
  matrix_class.fromFilename = function(filename,order)
    local f = april_assert(io.open(filename),
                           "Unable to open %s", filename)
    local ret = table.pack(matrix_class.read(archive_wrapper( f ),
                                             { [matrix.options.order]=order }))
    f:close()
    return table.unpack(ret)
  end
end

-- GENERIC FROM TAB FILENAME
matrix.__generic__.__make_generic_fromTabFilename__ = function(matrix_class)
  matrix_class.fromTabFilename = function(filename,order)
    local f = april_assert(io.open(filename),
                           "Unable to open %s", filename)
    local ret = table.pack(matrix_class.read(archive_wrapper( f ),
                                             { [matrix.options.order] = order,
                                               [matrix.options.tab] = true }))
    f:close()
    return table.unpack(ret)
  end
end

matrix.__generic__.__make_generic_fromCSVFilename__ = function(matrix_class)
  matrix_class.fromCSVFilename = function(filename,args)
    local args = get_table_fields({
        [matrix.options.order]   = { mandatory=false, type_match="string" },
        [matrix.options.delim]   = { mandatory=true, type_match="string", default="," },
        [matrix.options.default] = { mandatory=false } }, args)
    args[matrix.options.empty] = true
    args[matrix.options.tab] = true
    local f = april_assert(io.open(filename),
                           "Unable to open %s", filename)
    local ret = table.pack(matrix_class.read(archive_wrapper( f ), args))
    f:close()
    return table.unpack(ret)
  end
end

-- GENERIC FROM STRING
matrix.__generic__.__make_generic_fromString__ = function(matrix_class)
  matrix_class.fromString = function(str)
    return matrix_class.read(aprilio.stream.input_lua_string(str))
  end
end

-- GENERIC TO FILENAME
matrix.__generic__.__make_generic_toFilename__ = function(matrix_class)
  class.extend(matrix_class, "toFilename",
               function(self,filename,mode)
                 local f = april_assert(io.open(filename,"w"),
                                        "Unable to open %s", filename)
                 local ret = table.pack(self:write(f,
                                                   { [matrix.options.ascii] = (mode=="ascii") }))
                 f:close()
                 return table.unpack(ret)
  end)
end

-- GENERIC TO TAB FILENAME
matrix.__generic__.__make_generic_toTabFilename__ = function(matrix_class)
  class.extend(matrix_class, "toTabFilename",
               function(self,filename,mode)
                 local f = april_assert(io.open(filename,"w"),
                                        "Unable to open %s", filename)
                 local ret = table.pack(self:write(f,
                                                   { [matrix.options.ascii] = (mode=="ascii"),
                                                     [matrix.options.tab]   = true }))
                 f:close()
                 return table.unpack(ret)
  end)
end

-- GENERIC TO STRING
matrix.__generic__.__make_generic_toString__ = function(matrix_class)
  class.extend(matrix_class, "toString",
               function(self,mode)
                 return self:write({ [matrix.options.ascii] = (mode=="ascii") })
  end)
end

function matrix.__generic__.__make_all_serialization_methods__(matrix_class)
  matrix.__generic__.__make_generic_fromFilename__(matrix_class)
  matrix.__generic__.__make_generic_fromTabFilename__(matrix_class)
  matrix.__generic__.__make_generic_fromString__(matrix_class)
  matrix.__generic__.__make_generic_fromCSVFilename__(matrix_class)
  matrix.__generic__.__make_generic_toFilename__(matrix_class)
  matrix.__generic__.__make_generic_toTabFilename__(matrix_class)
  matrix.__generic__.__make_generic_toString__(matrix_class)
  matrix.__generic__.__make_generic_to_lua_string__(matrix_class)
end
