local ipairs   = ipairs
local pairs    = pairs
local tostring = tostring
local NA       = nan -- NaN values are used as "Not Available"
local defNA    = "<NA>"

-- utilities

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

local function is_NA(v) return tostring(v) == tostring(NA) end

local function build_order(tbl, NA_symbol)
  local symbols = {}
  for i,v in ipairs(tbl) do symbols[is_NA(v) and NA_symbol or v] = true end
  local order = iterator(pairs(symbols)):select(1):table()
  table.sort(order)
  return order
end

-- returns a categorized table, table of categories and inverted dictionary
local categorical =
  function(tbl, NA_symbol, order)
    assert(tbl and NA_symbol, "Needs a table and NA symbol")
    local categories = order or build_order(tbl, NA_symbol)
    local cat2id = table.invert(categories)
    local result = {}
    for i,v in ipairs(tbl) do
      result[i] = is_NA(v) and cat2id[NA_symbol] or cat2id[v]
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

-- parses a CSV line using sep as delimiter and adding NA when required
local function parse_csv_line(line, sep)
  local parsed = {}
  local init = 1
  while init <= #line do
    local v
    local i,j = line:find(sep, init, true)
    i,j = i or #line+1,j or #line
    if i == init then
      v = NA
    else
      v = line:sub(init, i-1)
      v = tonumber(v) or v
      if type(v) == "string" then
        local v_upper = v:upper()
        if v_upper == defNA or v_upper == "NA" or v_upper == "NAN" then v = NA end
      end
    end
    assert(v, "Unexpected read error")
    table.insert(parsed, v)
    init = j+1
  end
  if line:sub(#line,#line) == sep then table.insert(parsed, NA) end
  return parsed
end

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

-- Inspired in pandas (Python) dataframe
-- http://pandas-docs.github.io/pandas-docs-travis/
local data_frame,methods = class("data_frame", aprilio.lua_serializable)
_G.data_frame = data_frame -- global definition

local function dataframe_tostring(proxy)
  local self = getmetatable(proxy)  
  if not next(rawget(self, "data")) then
    return table.concat{
      "Empty data_frame\n",
      "Columns: ", stringfy(rawget(self, "columns"), "ascii"), "\n",
      "Index: ", stringfy(rawget(self, "index"), "ascii"), "\n",
    }
  else
    local tbl = { "data_frame\n" }
    for j,col_name in ipairs(rawget(self, "columns")) do
      table.insert(tbl, "\t")
      table.insert(tbl, col_name)
    end
    table.insert(tbl, "\n")
    for i,row_name in ipairs(rawget(self, "index")) do
      table.insert(tbl, row_name)
      for j,col_name in ipairs(rawget(self, "columns")) do
        table.insert(tbl, "\t")
        table.insert(tbl, tostring(rawget(self, "data")[col_name][i]))
      end
      table.insert(tbl, "\n")
    end
    return table.concat(tbl)
   end
end

local function dataframe_index(proxy, key)
  local self = getmetatable(proxy)
  if type(key) == "table" then
    local v = methods.column(proxy, key[1])
    return v
  else
    return methods[key]
  end
end

local function dataframe_newindex(proxy, key, value)
  local self = getmetatable(proxy)
  if type(key) == "table" then
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
    rawset(self, key, value)
  end
end

data_frame.constructor =
  function(self, params)
    -- configure as proxy table
    do
      local mt = getmetatable(self)
      local proxy = self
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
    rawset(self, "index", check_array( params.index or {}, "index" ))
    rawset(self, "col2id", invert(rawget(self, "columns")))
    rawset(self, "index2id", invert(rawget(self, "index")))
    rawset(self, "data", {})
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
        rawset(self, "columns", matrixInt32(data:dim(2)):linspace())
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
  end

data_frame.from_csv =
  function(path, params)
    local proxy = data_frame()
    local self = getmetatable(proxy)
    local data = rawget(self, "data")
    local params = get_table_fields({
        header = { default=true },
        sep = { default=',' },
                                    }, params or {})
    assert(#params.sep == 1, "Only one character sep is allowed")
    local f = type(path)~="string" and path or io.open(path)
    if params.header then
      rawset(self, "columns", parse_csv_line(f:read("*l"), params.sep))
      rawset(self, "col2id", invert(rawget(self, "columns")))
      for _,col_name in ipairs(rawget(self, "columns")) do data[col_name] = {} end
    end
    local n = 0
    for row_line in f:lines() do
      n = n + 1
      local tbl = parse_csv_line(row_line, params.sep)
      if #rawget(self, "columns") == 0 then
        rawset(self, "columns", matrixInt32(#tbl):linspace())
        rawset(self, "col2id", invert(rawget(self, "columns")))
        for _,col_name in ipairs(rawget(self, "columns")) do data[col_name] = {} end
      end
      assert(#tbl == #rawget(self, "columns"), "Not matching number of columns")
      for j,col_name in ipairs(rawget(self, "columns")) do
        data[col_name][n] = tbl[j]
      end
    end
    rawset(self, "index", matrixInt32(n):linspace())
    rawset(self, "index2id", invert(rawget(self, "index")))
    if path ~= f then f:close() end
    return proxy
  end

methods.to_csv =
  function(self, path, params)
    local self = getmetatable(self)
    local params = get_table_fields({
        header = { default=true },
        sep = { default=',' },
                                    }, params or {})
    local sep = params.sep
    local f = type(path)~="string" and path or io.open(path, "w")
    if params.header then
      f:write(concat(rawget(self, "columns"), sep))
      f:write("\n")
    end
    local data = rawget(self, "data")
    local tbl = {}
    for i,row_name in ipairs(rawget(self, "index")) do
      for j,col_name in ipairs(rawget(self, "columns")) do
        tbl[j] = data[col_name][i]
      end
      f:write(table.concat(tbl, sep))
      f:write("\n")
      table.clear(tbl)
    end
    if path ~= f then f:close() end
  end

methods.drop =
  function(self, dim, ...)
    local self = getmetatable(self)
    assert(dim, "Needs a dimension number, 1 or 2")
    local labels = table.pack(...)
    if dim == 1 then
      error("Not implemented for index")
    elseif dim == 2 then
      local num_cols = #rawget(self, "columns")
      for _,col_name in ipairs(labels) do
        local col_id = april_assert(rawget(self, "col2id")[col_name],
                                    "Unknown column name %s", col_name)
        rawget(self, "data")[col_name]   = nil
        rawget(self, "columns")[col_id]  = nil
        rawget(self, "col2id")[col_name] = nil
      end
      local j=1
      for i=1,num_cols do
        local v = rawget(self, "columns")[i]
        rawget(self, "columns")[i] = nil
        rawget(self, "columns")[j] = v
        if v then
          rawget(self, "col2id")[rawget(self, "columns")[j]] = j
          j=j+1
        end
      end
    else
      error("Incorrect dimension number, it should be 1 or 2")
    end
  end

methods.as_matrix =
  function(self, ...)
    local self = getmetatable(self)
    local args = table.pack(...)
    local params = {}
    if #args > 0 and type(args[#args]) == "table" then params = table.remove(args) end
    params = get_table_fields({
        dtype = { type_match = "string", default = "float" },
        categorical_dtype = { type_match = "string", default = "float" },
        categories = { type_match = "table", default = nil },
        NA = { type_match = "string", default = defNA },
                              }, params)
    local categories = params.categories or {}
    local inv_categories = {}
    local dtype = params.dtype
    local categorical_dtype = params.categorical_dtype
    assert(dtype ~= "sparse", "Sparse is only allowed in categorical_dtype field")
    local data = rawget(self, "data")
    local col2id = rawget(self, "col2id")
    local cols_slice
    if #args == 0 then
      cols_slice = rawget(self, "columns")
    else
      cols_slice = args
      if dtype == "categorical" then
        assert(#categories == 0 or type(categories[1]) == "table" and
                 #categories == #args, "Needs a table with category arrays in categories field")
      end
    end
    local tbl = {}
    for i,col_name in ipairs(cols_slice) do
      local dtype = dtype
      april_assert(col2id[col_name], "Unknown column name %s", col_name)
      local col_data = data[col_name]
      if dtype == "categorical" then
        dtype = categorical_dtype
        col_data,categories[i],inv_categories[i] = categorical(col_data, defNA, categories[i])
      end
      table.insert(tbl, to_matrix(col_data, dtype, categories[i] and #categories[i]))
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
  function(self, key)
    local self = getmetatable(self)
    local data = rawget(self, "data")
    if data then
      return data[tonumber(key) or key]
    end
  end

methods.insert =
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
    table.insert(rawget(self, "columns"), location, col_name)
    rawget(self, "col2id")[col_name] = invert(rawget(self, "columns"))
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

methods.get_index =
  function(self)
    local self = getmetatable(self)
    return rawget(self, "index")
  end

methods.get_columns =
  function(self)
    local self = getmetatable(self)
    return rawget(self, "columns")
  end

methods.levels =
  function(self, key, NA_symbol)
    local self = getmetatable(self)
    local key = tonumber(key) or key
    local data = rawget(self, "data")
    return build_order(data, NA_symbol or defNA)
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

return data_frame
