-- Inspired in pandas (Python) dataframe
-- http://pandas-docs.github.io/pandas-docs-travis/
local data_frame,methods        = class("data_frame", aprilio.lua_serializable)
local groupped,groupped_methods = class("data_frame.groupped")
_G.data_frame = data_frame -- global definition
------------------------------------------------------------------

local assert      = assert
local ipairs      = ipairs
local pairs       = pairs
local tostring    = tostring
local tonumber    = tonumber
local type        = type
local NA          = nan -- NaN values are used as "Not Available"
local defNA       = "NA"
local tostring_nan = tostring(NA)

-- utilities

local function take(data, indices)
  local result
  if type(data) == "table" then
    result = {}
  else
    local ctor = class.of(data)
    result = ctor(#indices):index(1, indices)
  end
  -- FIXME: implement in C++ if possible
  for i=1,#indices do result[i] = data[indices[i]] end
  return result
end

local function sparse_join(tbl, categories)
  local nrows   = tbl[1]:dim(1)
  local ncols   = iterator(tbl):map(bind(tbl[1].dim, nil, 2)):sum()
  local values  = {}
  local indices,acc = {},0
  for i=1,#tbl do
    values[i]  = matrix(nrows, 1, tbl[i]:values())
    indices[i] = matrixInt32(nrows, 1, tbl[i]:indices() ):scalar_add(acc)
    acc = acc + tbl[i]:dim(2)
  end
  values  = matrix.join(2, values):data()
  indices = matrix.join(2, indices):data()
  local first_index = matrixInt32( tbl[1]:first_index() ):scal(#tbl):data()
  return matrix.sparse.csr(nrows, ncols, values, indices, first_index)
end

local function is_nan(v) return tonumber(v) and tostring(v) == tostring_nan end

local function build_sorted_order(tbl, NA_symbol)
  local symbols = {}
  for i,v in ipairs(tbl) do symbols[is_nan(v) and NA_symbol or v] = true end
  local order = iterator(pairs(symbols)):select(1):table()
  table.sort(order, function(a,b)
               if type(a)~=type(b) then return tostring(a) < tostring(b) else return a<b end
  end)
  return order
end

-- returns a categorized table, table of categories and inverted dictionary
local categorical =
  function(tbl, NA_symbol, order)
    assert(tbl and NA_symbol, "Needs a table and NA symbol")
    local categories = order or build_sorted_order(tbl, NA_symbol)
    local cat2id = table.invert(categories)
    local result = {}
    for i,v in ipairs(tbl) do
      result[i] = april_assert(is_nan(v) and cat2id[NA_symbol] or cat2id[v],
                               "Unknown level value %s", v)
    end
    return result,categories,cat2id
  end

-- returns the next number available in a given array
local function next_number(columns)
  local n = 0
  for i=1,#columns do
    if type(columns[i]) == "number" then n = math.max(n, columns[i]) end
  end
  return n+1
end

-- returns a matrix from the given column data
local to_matrix
do
  local matrix_ctor = {
    float   = matrix,
    double  = matrixDouble,
    complex = matrixComplex,
    int32   = matrixInt32,
    char    = matrixChar,
    bool    = matrixBool,
  }
  function to_matrix(data, dtype, ncols)
    if dtype == "sparse" then
      assert(ncols > 2, "For binary data sparse is not allowed")
      -- assuming the underlying data is categorical
      local values      = matrix(#data):fill(1.0):data()
      local indices     = matrixInt32(data):scalar_add(-1.0):data()
      local first_index = matrixInt32(indices:size()+1):linear():data()
      return matrix.sparse.csr(indices:size(), -- num rows
                               ncols,          -- num cols
                               values,
                               indices,
                               first_index)
    else
      local ctor = april_assert(matrix_ctor[dtype],
                                "Unable to build matrix type %s", dtype)
      if type(data) == "table" then
        return ctor(#data,1,data)
      else
        data = class.of(data)==ctor and data or data:convert_to(dtype)
        return data:rewrap(data:size(),1)
      end
    end
  end
end

-- concats a matrix or a table data using sep as delimiter
local function concat(array, sep)
  if type(array) ~= "table" then
    array = array:toTable()
  end
  return table.concat(array, sep)
end

local function next_token_find(line, init, match, sep, quotechar)
  local quoted
  local i,j = line:find(match, init)
  if j and line:sub(j,j) ~= sep then
    i,j = line:find(quotechar, j+1)
    i,j = line:find(sep, j+1)
    i,j = i or #line, j or #line + 1
    quoted = true
  end
  return i,j,quoted
end

local find  = string.find
local gsub  = string.gsub
local sub   = string.sub
local yield = coroutine.yield
local tonumber = tonumber

-- parses a CSV line using sep as delimiter and adding NA when required
local parse_csv_line = util.__parse_csv_line__

-- converts an array or a matrix into a string
local function stringfy(array)
  if class.of(array) then array = array:toTable() end
  return util.to_lua_string(array, "ascii")
end

-- checks if an array is a table or a matrix
local function check_array(array, field)
  if type(array) ~= "table" then
    local c = class.of(array)
    april_assert(c, "Needs an array in parameter %s", field)
    local nd = array.num_dim
    assert(nd and nd(array) == 1, "Needs a one-dimensional matrix or a table")
  else
    for i=1,#array do array[i] = tonumber(array[i]) or array[i] end
  end
  return array
end

-- returns the inverted map of an array or a matrix
local function invert(array)
  local t = {}
  for i,v in ipairs(array) do
    april_assert(not t[v], "Repeated identifier %s", tostring(v))
    t[v] = i
  end
  return t
end

---------------------------------------------------------------------------
---------------------------------------------------------------------------
---------------------------------------------------------------------------

local function quote(x, sep, quotechar, decimal)
  if tonumber(x) then
    x = tostring(x)
    if decimal ~= "." then x = x:gsub("%.", decimal) end
  end
  if x:find(sep) then
    return "%s%s%s"%{quotechar,x,quotechar}
  else
    return x
  end
end

local function dataframe_tostring(proxy)
  local self = getmetatable(proxy)  
  if not next(rawget(self, "data")) then
    return table.concat{
      "Empty data_frame\n",
      "[data_frame of %d rows x %d columns]\n"%
        {#rawget(self,"index"),#rawget(self,"columns")}
    }
  else
    local tbl = { }
    for j,col_name in ipairs(rawget(self, "columns")) do
      table.insert(tbl, "\t")
      table.insert(tbl, quote(col_name, '%s', '"', '.'))
    end
    table.insert(tbl, "\n")
    local truncated = false
    for i,row_name in ipairs(rawget(self, "index")) do
      if i > 20 then table.insert(tbl, "...\n") truncated = true break end
      table.insert(tbl, row_name)
      for j,col_name in ipairs(rawget(self, "columns")) do
        table.insert(tbl, "\t")
        table.insert(tbl, quote(rawget(self, "data")[col_name][i],
                                '%s', '"', '.'))
      end
      table.insert(tbl, "\n")
    end
    if truncated then
      local index = rawget(self,"index")
      for i=math.max(#index - 20, 21),#index do
        local row_name=index[i]
        table.insert(tbl, row_name)
        for j,col_name in ipairs(rawget(self, "columns")) do
          table.insert(tbl, "\t")
          table.insert(tbl, quote(rawget(self, "data")[col_name][i],
                                  '%s', '"', '.'))
        end
        table.insert(tbl, "\n")
      end
    end
    table.insert(tbl, "[data_frame of %d rows x %d columns]\n"%
                   {#rawget(self,"index"),#rawget(self,"columns")})
    return table.concat(tbl)
  end
end

local function dataframe_index(proxy, key)
  local self = getmetatable(proxy)
  local tt = type(key)
  if tt == "number" then
    key, tt = { rawget(self,"columns")[key] }, "table"
  end
  if tt == "table" then
    local v = methods.column(proxy, key[1])
    return v
  else
    return methods[key]
  end
end

local function dataframe_newindex(proxy, key, value)
  local self = getmetatable(proxy)
  local tt   = type(key)
  if tt == "number" then
    key, tt = { rawget(self,"columns")[key] }, "table"
  end
  if tt == "table" then
    local data = rawget(self, "data")
    if data then
      local col_data = value
      local key = tonumber(key[1]) or key[1]
      local v = data[key]
      if v then
        methods.set(proxy, key, col_data)
      else
        methods.insert(proxy, col_data, { column_name=key })
      end
    end
  else
    error("Unable to index with string, use a number or a table with the column name")
  end
end

data_frame.constructor =
  april_doc{
    class = "method",
    summary = "Constructor for data_frame objects",
    description = "Builds an empty data_frame or a data_frame taken from given data",
    params = {
      data = { "A table indexed by column names with all the expected columns data [optional]" },
      index = { "An index table (or matrix) which allow to identify every row [optional]" },
      columns = { "Order of the columns given in data [optional]" },
    },
    outputs = {
      "An instance of data_frame class",
    },
  } ..
  function(self, params)
    -- configure as proxy table
    local proxy
    do
      local mt = getmetatable(self)
      proxy = self
      self = {}
      self.__newindex  = dataframe_newindex
      self.__index     = dataframe_index
      self.__tostring  = dataframe_tostring
      self.index_table = mt.index_table
      self.cls         = mt.cls
      self.id          = mt.id
      setmetatable(proxy, self)
      setmetatable(self, mt)
    end
    local params = get_table_fields({
        data = { }, -- data can be a matrix or a Lua table
        index = { },
        columns = { },
                                    }, params or {})
    local tdata = type(data)
    rawset(self, "columns", check_array( params.columns or {}, "columns" ))
    rawset(self, "col2id", invert(rawget(self, "columns")))
    rawset(self, "data", {})
    local t_params_index = type(params.index)
    if t_params_index == "string" or t_params_index == "number" then
      rawset(self, "index", {})
    else
      rawset(self, "index", check_array( params.index or {}, "index" ))
    end
    rawset(self, "index2id", invert(rawget(self, "index")))
    local data = params.data
    if type(data) == "table" then
      if #rawget(self, "index") == 0 then
        local n = #select(2,next(data))
        rawset(self, "index", matrixInt32(n):linspace())
      end
      local n = #rawget(self, "index")
      local cols = {}
      for col_name,col_data in pairs(data) do
        col_name = tonumber(col_name) or col_name
        table.insert(cols, col_name)
        if #rawget(self, "columns") > 0 then
          april_assert(rawget(self, "col2id")[col_name],
                       "Not valid column name %s", col_name)
        end
        assert(n == #col_data, "Length of values does not match number of rows")
        if class.of(col_data) then
          local sq = assert(col_data.squeeze, "Needs matrix or table as columns")
          col_data = col_data:squeeze()
          assert(col_data:num_dim() == 1, "Needs a rank one matrix")
        end
        rawget(self, "data")[col_name] = col_data
      end
      if #rawget(self, "columns") == 0 then
        table.sort(cols)
        rawset(self, "columns", cols)
        rawset(self, "col2id", invert(cols))
      end
    elseif data then
      assert(class.of(data), "Needs a matrix or dictionary in argument data")
      local nd = data.num_dim
      assert(nd and nd(data)==2, "Needs a bi-dimensional matrix in argument data")
      if #rawget(self, "columns") == 0 then
        rawset(self, "columns", iterator.range(data:dim(2)):table())
        rawset(self, "col2id", invert(rawget(self, "columns")))
      else
        assert(data:dim(2) == #rawget(self, "columns"),
               "Incorrect number of columns in data")
      end
      if #rawget(self, "index") == 0 then
        rawset(self, "index", matrixInt32(data:dim(1)):linspace())
        rawset(self, "index2id", invert(rawget(self, "index")))
      else
        assert(data:dim(1) == #rawget(self, "index"),
               "Incorrect number of rows in data")
      end
      for j,col_name in ipairs(rawget(self, "columns")) do
        rawget(self, "data")[col_name] = data:select(2,j)
      end
    end
    if t_params_index == "string" or t_params_index == "number" then
      proxy:set_index(params.index)
    end
  end

data_frame.from_csv =
  april_doc{
    class = "function",
    summary = "Builds a data_frame from a CSV file",
    description = "This loader allow empty fields",
    params = {
      "First parameter is the path to CSV filename, second parameter is a table",
      header = { "A boolean indicating if the CSV has a header row [optional],",
                 "by default it is true" },
      sep = { "Sep charecter for every field [optional], by default it is: ,"},
      quotechar = { "Character used to as delimiter for strings [optional],",
                    "by default it is: \"" },
      decimal = { "Decimal point character [optional], by default it is: .", },
      NA = { "Not avaliable token [optional], by default it is NA" },
      index = { "Table or matrix with an index to identify every row, it",
                "can be a string indicating which column in CSV file is the",
                "index [optional]" },
      columns = { "A table with column keys [optional]" },
    },
    outputs = {
      "An instance of data_frame class"
    },
  } ..
  function(path, params)
    local proxy = data_frame()
    local self = getmetatable(proxy)
    local data = rawget(self, "data")
    local params = get_table_fields({
        header = { default=true },
        sep = { default=',' },
        quotechar = { default='"' },
        decimal = { default='.' },
        NA = { default=defNA },
        index = { },
        columns = { },
                                    }, params or {})
    local sep = params.sep
    local quotechar = params.quotechar
    local decimal = params.decimal
    local NA_str = params.NA
    local double_quote = quotechar..quotechar
    local quote_match = "^"..quotechar
    local quote_closing = quotechar.."("..quotechar.."?)"
    local decimal_match = "%"..decimal
    local number_match = "^[+-]?%d*"..decimal.."?%d+[eE]?[+-]?%d*$"
    assert(#sep == 1, "Only one character sep is allowed")
    assert(#quotechar <= 1, "Only zero or one character quotechar is allowed")
    assert(#decimal == 1, "Only one character decimal is allowed")
    local f = type(path)~="string" and path or io.open(path)
    local aux = {}
    if params.header then
      local t = parse_csv_line(aux, f:read("*l")..sep, sep, quotechar,
                               decimal, NA_str, nan)
      rawset(self, "columns", iterator(t):table())
      for i,col_name in ipairs(rawget(self, "columns")) do
        if is_nan(col_name) then
          col_name = next_number(rawget(self, "columns"))
          rawget(self, "columns")[i] = col_name
        end
      end
      if params.columns then
        assert(#rawget(self,"columns") == #params.columns,
               "Incorrect number of columns at field 'columns'")
        rawset(self, "columns", params.columns)
      end
      rawset(self, "col2id", invert(rawget(self, "columns")))
      for _,col_name in ipairs(rawget(self, "columns")) do data[col_name] = {} end
    elseif params.columns then
      rawset(self, "columns", params.columns)
      rawset(self, "col2id", invert(rawget(self, "columns")))
      for _,col_name in ipairs(rawget(self, "columns")) do data[col_name] = {} end      
    end
    local n = 0
    if #rawget(self, "columns") == 0 then
      n = n + 1
      local first_line = iterator(parse_csv_line(aux, f:read("*l")..sep, sep, quotechar,
                                                 decimal, NA_str, nan)):table()
      rawset(self, "columns", iterator.range(#first_line):table())
      rawset(self, "col2id", invert(rawget(self, "columns")))
      for j,col_name in ipairs(rawget(self, "columns")) do
        data[col_name] = { first_line[j] }
      end
    end
    local columns = rawget(self, "columns")
    for row_line in f:lines() do
      n = n + 1
      for j,value in ipairs(parse_csv_line(aux, row_line..sep, sep, quotechar,
                                           decimal, NA_str, nan)) do
        local d = data[columns[j] or j]
        d[n] = value
      end
      if n % 4096 == 0 then collectgarbage("collect") end
    end
    rawset(self, "index", matrixInt32(n):linspace())
    rawset(self, "index2id", invert(rawget(self, "index")))
    if path ~= f then f:close() end
    if params.index then proxy:set_index(params.index) end
    return proxy
  end

methods.to_csv =
  april_doc{
    class = "method",
    summary = "Writes the caller data_frame into a CSV filename",
    params = {
      "The first one is a filename, the second one is a table",
      header = { "Boolean indicating if the header should be written [optional], by default it is true" },
      sep = { "Character used as sep [optional], by default it is: ," },
      quotechar = { "Character used to as delimiter for strings [optional],",
                    "by default it is: \"" },
      decimal = { "Decimal point character [optional], by default it is: .", },
      NA = { "Not avaliable token [optional], by default it is NA" },
    },
  } ..
  function(self, path, params)
    local self = getmetatable(self)
    local params = get_table_fields({
        header = { default=true },
        sep = { default=',' },
        quotechar = { default='"' },
        NA = { default=defNA },
        decimal = { default="." },
                                    }, params or {})
    local sep = params.sep
    local quotechar = params.quotechar
    local NA_str = params.NA
    local decimal = params.decimal
    assert(#sep == 1, "Only one character sep is allowed")
    assert(#quotechar <= 1, "Only zero or one character quotechar is allowed")
    local f = type(path)~="string" and path or io.open(path, "w")
    if params.header then
      local columns = {}
      for i,col_name in ipairs(rawget(self, "columns")) do
        columns[i] = quote(col_name, sep, quotechar, decimal)
      end
      f:write(concat(columns, sep))
      f:write("\n")
    end
    local data = rawget(self, "data")
    local tbl = {}
    for i,row_name in ipairs(rawget(self, "index")) do
      for j,col_name in ipairs(rawget(self, "columns")) do
        local v = data[col_name][i]
        if tonumber(v) and is_nan(v) then v = NA_str end
        tbl[j] = quote(v, sep, quotechar, decimal)
      end
      f:write(table.concat(tbl, sep))
      f:write("\n")
      table.clear(tbl)
    end
    if path ~= f then f:close() end
  end

methods.drop =
  april_doc{
    class = "method",
    summary = "Removes one column or row from the data_frame (it is done in-place)",
    params = {
      "A dimension number (1 for rows, 2 for columns)",
      "First column or row name to drop",
      "Second column or row name to drop",
      "...",
      "Last column or row name to drop",
    }
  } ..
  function(self, dim, ...)
    local self = getmetatable(self)
    assert(dim, "Needs a dimension number, 1 or 2")
    local labels = table.pack(...)
    if dim == 1 then
      error("Not implemented for index")
    elseif dim == 2 then
      local num_cols = #rawget(self, "columns")
      local new_columns = {}
      if type(rawget(self, "columns")) ~= "table" then
        new_columns = matrixInt32(num_cols - #labels)
      end
      local deleted = {}
      for _,col_name in ipairs(labels) do
        local col_name = tonumber(col_name) or col_name
        local col_id = april_assert(rawget(self, "col2id")[col_name],
                                    "Unknown column name %s", col_name)
        deleted[col_id] = true
        rawget(self, "data")[col_name]   = nil
        rawget(self, "col2id")[col_name] = nil
      end
      local j=1
      for i=1,num_cols do
        if not deleted[i] then
          local v = rawget(self, "columns")[i]
          new_columns[j] = v
          rawget(self, "col2id")[v]  = j
          j=j+1
        end
      end
      rawset(self, "columns", new_columns)
    else
      error("Incorrect dimension number, it should be 1 or 2")
    end
  end

methods.as_matrix =
  april_doc{
    class = "method",
    summary = "Converts the whole data_frame (or a given list of its columns) into a matrix",
    params = {
      "First column [optional]",
      "...",
      "Last column [optional]",
      "Last argument can be a table of parameters with fields: dtype, categorical_dtype, categories, NA",
    },
    outputs = {
      "A matrix instance",
    }
  } ..
  function(self, ...)
    local self = getmetatable(self)
    local args = table.pack(...)
    local params = {}
    if #args > 0 and type(args[#args]) == "table" then params = table.remove(args) end
    params = get_table_fields({
        dtype = { type_match = "string", default = "float" },
        categorical_dtype = { type_match = "string", default = "float" },
        categories = { type_match = "table", default = nil },
        NA = { default = NA },
                              }, params)
    local categories = params.categories or {}
    local inv_categories = {}
    local dtype = params.dtype
    local categorical_dtype = params.categorical_dtype
    local NA = params.NA
    assert(dtype ~= "sparse", "Sparse is only allowed in categorical_dtype field")
    local data = rawget(self, "data")
    local col2id = rawget(self, "col2id")
    local cols_slice
    if #args == 0 then
      cols_slice = rawget(self, "columns")
    else
      cols_slice = args
      --if dtype == "categorical" then
      --assert(#categories == 0 or type(categories[1]) == "table" and
      --#categories == #args, "Needs a table with category arrays in categories field")
      --end
    end
    local tbl = {}
    for i,col_name in ipairs(cols_slice) do
      local dtype = dtype
      april_assert(col2id[col_name], "Unknown column name %s", col_name)
      local col_data = data[col_name]
      if dtype == "categorical" then
        dtype = categorical_dtype
        col_data,categories[i],inv_categories[i] = categorical(col_data, NA, categories[i])
      end
      local ncols = categories[i] and #categories[i]
      local m = to_matrix(col_data, dtype, ncols)
      if not is_nan(NA) then m[m:eq(nan)] = NA end
      if ncols and ncols <= 2 then m:scalar_add(-1.0) end
      table.insert(tbl, m)
    end
    if dtype == "categorical" then
      if categorical_dtype == "sparse" then
        return sparse_join(tbl, categories),categories,inv_categories
      else
        return matrix.join(2, tbl),categories,inv_categories
      end
    else
      return matrix.join(2, tbl)
    end
  end

-- methods.loc =
--   function(proxy, row_key)
--     local self = getmetatable(proxy)
--     local row_key = tonumber(row_key) or row_key
--     local i       = assert(rawget(self, "index2id")[row_key], "Unknown label")
--     return methods.iloc(proxy, i)
--   end

-- methods.iloc =
--   function(proxy, i)
--     local self    = getmetatable(proxy)
--     local data    = rawget(self, "data")
--     local result  = {}
--     for _,col_name in ipairs(rawget(self, "columns")) do
--       result[col_name] = { data[col_name][i] }
--     end
--     return data_frame{
--       data    = result,
--       index   = { (assert(rawget(self, "index")[i], "Index out-of-bounds")) },
--       columns = rawget(self, "columns"),
--     }
--   end

methods.column =
  april_doc{
    class = "method",
    summary = "Returns a column data",
    params = { "The column name" },
    outputs = { "The column data" },
  } ..
  function(self, key)
    local self = getmetatable(self)
    local data = rawget(self, "data")
    if data then
      return data[tonumber(key) or key]
    end
  end

methods.insert =
  april_doc{
    class = "method",
    summary = "Inserts a new column",
    params = {
      "The column data",
      "A table with column_name and location [optional]",
    },
  } ..
  function(self, col_data, params)
    local self = getmetatable(self)
    local params = get_table_fields({
        column_name = { },
        location = { type_match="number" },
                                    }, params or {})
    local col_name = params.column_name or next_number(rawget(self, "columns"))
    local location = params.location or (#rawget(self, "columns")+1)
    col_name = tonumber(col_name) or col_name
    assert(location >= 1 and location <= (#rawget(self, "columns")+1),
           "Parameter location is out-of-bounds")
    april_assert(not rawget(self, "col2id")[col_name],
                 "Column name collision: %s", col_name)
    local columns = rawget(self, "columns")
    if type(columns) ~= "table" then columns = columns:toTable() end
    table.insert(columns, location, col_name)
    rawset(self, "columns", columns)
    rawset(self, "col2id", invert(rawget(self, "columns")))
    if class.of(col_data) then
      local sq = assert(col_data.squeeze, "Needs matrix or table as columns")
      col_data = col_data:squeeze()
      assert(col_data:num_dim() == 1, "Needs a rank one matrix")
    end
    if #rawget(self, "index") == 0 then
      rawset(self, "index", matrixInt32(#col_data):linspace())
    end
    assert(#col_data == #rawget(self, "index"),
           "Length of values does not match number of rows")
    rawget(self, "data")[col_name] = col_data
  end

methods.set =
  april_doc{
    class = "method",
    summary = "Changes the data in a given column name",
    params = {
      "The column name",
      "The new column data",
    },
  } ..
  function(self, col_name, col_data)
    local self = getmetatable(self)
    assert(col_name, "Needs column name as first argumnet")
    assert(col_data, "Needs column data as second argument")
    local col_name = tonumber(col_name) or col_name
    april_assert(rawget(self, "col2id")[col_name],
                 "Unknown column name: %s", col_name)
    if class.of(col_data) then
      local sq = assert(col_data.squeeze, "Needs matrix or table as columns")
      col_data = col_data:squeeze()
      assert(col_data:num_dim() == 1, "Needs a rank one matrix")
    end
    assert(#col_data == #rawget(self, "index"),
           "Length of values does not match number of rows")
    rawget(self, "data")[col_name] = col_data
  end

methods.reorder =
  april_doc{
    class="method",
    summary="Changes the order of the columns",
  } ..
  function(self, columns)
    local self = getmetatable(self)
    local columns = check_array(columns)
    for i,col_name in pairs(columns) do
      april_assert(rawget(self, "col2id")[col_name],
                   "Unknown column %s", col_name)
    end
    assert(#columns == #rawget(self, "columns"),
           "Unexpected number of columns")
    rawset(self, "columns", columns)
    rawset(self, "col2id", invert(columns))
  end

methods.set_index =
  april_doc{
    class="method",
    summary="Changes the index",
  } ..
  function(self, col_name_or_data)
    local self = getmetatable(self)
    local col_name_or_data = tonumber(col_name_or_data) or col_name_or_data
    local tt = type(col_name_or_data)
    if tt == "string" or tt == "number" then
      rawset(self, "index",
             check_array(april_assert(rawget(self, "data")[col_name_or_data],
                                      "Unable to locate column %s", col_name_or_data),
                         "1"))
    else
      rawset(self, "index", check_array( col_name_or_data or {}, "1" ))
    end
    rawset(self, "index2id", invert(rawget(self, "index")))
  end

do
  local function get_key(df, key)
    if not key then
      return df:get_index()
    else
      return (april_assert(df[{ key }], "Unable to locate column name %s", key))
    end
  end
  
  methods.merge =
    april_doc{
      class="method",
      summary="Implements join operation between this and other data_frame",
    } ..
    function(self, other, params)
      local params = get_table_fields({
          how = { default="left" },
          on = { },
          left_on = { },
          right_on = { },
                                      }, params or {})
      local how       = params.how
      local left_key  = params.left_on  or params.on
      local right_key = params.right_on or params.on
      if how == "right" then self,other = other,self end
      local result          = data_frame()
      local result_proxy    = result
      local result          = getmetatable(result)
      local self_proxy      = self
      local other_proxy     = other
      local self            = getmetatable(self)
      local other           = getmetatable(other)
      local self_index      = rawget(self,  "index")
      local other_index     = rawget(other, "index")
      local other_index2id  = rawget(other, "index2id")
      local result_index
      --
      if left_key then
        april_assert(self_proxy[{ left_key }], "Unable to locate column %s", left_key)
      end
      if right_key then
        april_assert(other_proxy[{ right_key }], "Unable to locate column %s", right_key)
      end
      -- prepare the result_index depending in the given join type (how)
      if how == "right" or how == "left" then
        result_proxy:set_index(self_index)
      elseif how == "outer" then
        error("Not implemented")
        assert(type(other_index) == type(self_index),
               "Incompatible indices in given data_frames")
        local idx = {}
        for i=1,#self_index do idx[self_index[i]] = true end
        for i=1,#other_index do idx[other_index[i]] = true end
        idx = iterator(pairs(idx)):select(1):table()
        table.sort(idx)
        if type(self_index) ~= "table" then
          idx = class.of(self_index)(idx)
        end
        result_proxy:set_index(idx)
      elseif how == "inner" then
        error("Not implemented")
        local self_key  = get_key(self_proxy, key)
        local other_key = get_key(other_proxy, key)
        assert(type(other_index) == type(self_index),
               "Incompatible indices in given data_frames")
        local idx = {}
        for i=1,#self_index do
          local j = self_index[i]
          if other_index2id[j] then table.insert(idx, j) end
        end
        if type(self_index) ~= "table" then
          idx = class.of(self_index)(idx)
        end
        result_proxy:set_index(idx)
      else
        error("Incorrect how type " .. tostring(how))
      end
      local result_index = rawget(result, "index")
      local function process_columns(df, index2id, id2index)
        local col_names = rawget(df, "columns")
        local data      = rawget(df,  "data")
        for j=1,#col_names do
          local result_col2id = rawget(result, "col2id")
          local col_name = col_names[j]
          local col_data = data[col_name]
          if type(col_name) == "number" or not result_col2id[col_name] then
            local new_col_data
            if type(data) == "table" then
              new_col_data = {}
            else
              new_col_data = class.of(col_data)(#col_data)
            end
            for i=1,#result_index do
              local k = index2id[id2index[i]]
              new_col_data[i] = k and col_data[k] or NA
            end -- for every index in self
            if type(col_name) == "number" then
              col_name = next_number(rawget(result, "columns"))
            end
            result_proxy[{col_name}] = new_col_data
          else -- if it is a new column
            local new_col_data = result_proxy[{col_name}]
            for i=1,#result_index do
              local k = index2id[id2index[i]]
              if k then
                local v = new_col_data[i]
                if not v or is_nan(v) then
                  new_col_data[i] = col_data[k]
                else -- { v and not is_nan(v) }
                  assert(v == col_data[k], "Not compatible data frames")
                end
              end -- if exists current index key in df
            end -- for every index in result
          end -- existing column
        end -- for every column in df
      end -- function process_columns
      -- process all columns in both data frames
      if how == "left" then
        process_columns(self,  rawget(self, "index2id"), result_index)
        process_columns(other, invert(get_key(other_proxy, right_key)),
                        get_key(result_proxy, left_key))
      elseif how == "right" then
        process_columns(other, rawget(other, "index2id"), result_index)
        process_columns(self,  invert(get_key(self_proxy, right_key)),
                        get_key(result_proxy, left_key))
      else
        error("Not implemented")
      end
      return result_proxy
    end
end

methods.get_index =
  function(self)
    local self = getmetatable(self)
    return util.clone(rawget(self, "index"))
  end

methods.get_columns =
  function(self)
    local self = getmetatable(self)
    return util.clone(rawget(self, "columns"))
  end

methods.ncols =
  function(self)
    local self = getmetatable(self)
    return #rawget(self, "columns")
  end

methods.nrows =
  function(self)
    local self = getmetatable(self)
    return #rawget(self, "index")
  end

methods.levels =
  function(self, key, NA_symbol)
    local self = getmetatable(self)
    local key = tonumber(key) or key
    local data = rawget(self, "data")
    return build_sorted_order(data[key], NA_symbol or defNA)
  end

methods.ctor_name =
  function(self)
    return "data_frame"
  end

methods.ctor_params =
  function(self)
    local self = getmetatable(self)
    return {
      data = rawget(self, "data"),
      index = rawget(self, "index"),
      columns = rawget(self, "columns"),
    }    
  end

methods.clone = function(self) return data_frame(util.clone(self:ctor_params())) end

methods.map =
  april_doc{
    class="method",
    summary="Maps a list of columns data into a new result table",
  } ..
  function(self, ...)
    local col_names = { ... }
    local func = table.remove(col_names)
    assert(type(func) == "function", "Needs a function as last argument")
    local self = getmetatable(self)
    local data = {}
    for i=1,#col_names do
      local col_name = col_names[i]     
      data[i] = april_assert(rawget(self, "data")[col_name],
                             "Unable to locate column %s", col_name)
    end
    local result,input = {},{}
    for i=1,#rawget(self, "index") do
      for j=1,#data do input[j] = data[j][i] end
      result[i] = func(table.unpack(input))
    end
    return result
  end

methods.iterate =
  april_doc{
    class="method",
    summary="Iterates over all rows (as ipairs)",
  } ..
  function(proxy, ...)
    local self    = getmetatable(proxy)
    local data    = rawget(self, "data")
    local columns = table.pack(...)
    if #columns == 0 then columns = rawget(self, "columns") end
    local col2id  = rawget(self, "col2id")
    for _,name in ipairs(columns) do
      april_assert(col2id[name], "Unknown column name %s", tostring(name))
    end
    return function(self, i)
      if i < #rawget(self,"index") then
        i = i + 1
        local row = {}
        for _,col_name in ipairs(columns) do
          row[col_name] = data[col_name][i]
        end
        return i,row
      end
    end,self,0
  end

methods.parse_datetime =
  april_doc{
    class="method",
    summary="Transforms the given columns into timestamp",
    outputs={
      "A Lua table",
    },
  } ..
  function(self, ...)
    local self   = getmetatable(self)
    local data   = rawget(self, "data")
    local args   = table.pack( ... )
    -- returns a table with year,month,day,hour,min,sec,isdst, as in os.time
    local parser = assert( (type(args[#args]) == "function") and table.remove(args) or nil,
      "Needs a parser function as last argument" )
    local list   = iterator(ipairs(args)):
      map(function(i,col_name)
          col_name = tonumber(col_name) or col_name
          return i,april_assert(data[col_name], "Unable to locate column %s", col_name)
      end):table()
    local result = iterator(multiple_ipairs(table.unpack(list))):
      -- FIXME: check table returned by parser function
    map(function(i,...) return i,os.time(parser(table.concat({...}, " "))) end):
      table()
    return result
  end

methods.groupby =
  april_doc{
    class="method",
    summary="Groups all data using the given columns values",
    outputs={
      "A groupped data object",
    },
  } ..
  function(self, ...)
    return groupped(self, ...)
  end

methods.index =
  april_doc{
    class="method",
    summary="Indexes the data frame by rows using the given indices table or matrix",
    outputs={
      "A new allocated data_frame",
    },
  } ..
  function(proxy, indices)
    local self   = getmetatable(proxy)
    local data   = rawget(self, "data")
    local idx    = take(rawget(self, "index"), indices)
    local result = data_frame{ index=idx }
    for _,col_name in ipairs(rawget(self, "columns")) do
      result[{col_name}] = take(data[col_name], indices)
    end
    return result
  end

------------------------------------------------------------------

function groupped.constructor(self, df, ...)
  local args   = table.pack(...)
  self.depth   = args.n
  self.groups  = {}
  self.columns = args
  self.df      = df
  for i,row in df:iterate(...) do
    local destination = self.groups
    for _,col_name in ipairs(args) do
      local v = row[col_name]
      if not is_nan(v) then
        d = destination[v] or {}
        destination[v],destination = d,d
      else
        break
      end
    end
    local v = destination[1] or mathcore.vector{ dtype="int32" }
    destination[1] = v
    v:push_back(i)
  end
  local function vectors_to_matrix(d)
    if type(d) == "table" then
      for i,v in pairs(d) do d[i] = vectors_to_matrix(v) end
      return d
    else
      return d:to_matrix()
    end
  end
  self.groups = vectors_to_matrix(self.groups)
end

groupped_methods.get_group = function(self, ...)
  local key = { ... }
  assert(#key == self.depth, "Incompatible number of column values")
  local g = self.groups
  for i,value in ipairs(key) do
    g = april_assert(g[value], "Unknown column value %s", value)
  end
  return self.df:index(g[1])
end

------------------------------------------------------------------

return data_frame
