local NA = "NA"

local function concat(array, sep)
  if type(array) ~= "table" then
    array = array:toTable()
  end
  return table.concat(array, sep)
end

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
    end
    assert(v, "Unexpected read error")
    table.insert(parsed, v)
    init = j+1
  end
  if line:sub(#line,#line) == sep then table.insert(parsed, NA) end
  return parsed
end

local function stringfy(array)
  if class.of(array) then array = array:toTable() end
  return util.to_lua_string(array, "ascii")
end

local function check_array(array, field)
  if type(array) ~= "table" then
    local c = class.of(array)
    april_assert(c, "Needs an array in parameter %s", field)
    local nd = array.num_dim
    assert(nd and nd(array) == 1, "Needs a one-dimensional matrix or a table")
  end
  return array
end

local function invert(array)
  local t = {}
  for i,v in ipairs(array) do
    april_assert(not t[v], "Repeated identifier %s", tostring(v))
    t[v] = i
  end
  return t
end

-- Inspired in pandas (Python) dataframe
-- http://pandas-docs.github.io/pandas-docs-travis/
local data_frame,methods = class("data_frame")
_G.data_frame = data_frame

data_frame.constructor =
  function(self, params)
    local params = get_table_fields({
        data = { },
        rows = { },
        columns = { },
                                    }, params or {})
    local tdata = type(data)
    self.columns = check_array( params.columns or {}, "columns" )
    self.rows    = check_array( params.rows or {}, "rows" )
    self.col2id  = invert(self.columns)
    self.rows2id = invert(self.rows)
    self.data    = {}
    local data   = params.data
    if type(data) == "table" then
      if #self.rows == 0 then
        local n = #select(2,next(data))
        self.rows = matrixInt32(n):linspace()
      end
      local n = #self.rows
      local cols = {}
      for col_name,col_data in pairs(data) do
        table.insert(cols, col_name)
        if #self.columns > 0 then
          april_assert(self.col2id[col_name],
                       "Not valid column name %s", col_name)
        end
        assert(n == #col_data, "Length of values different does not match length of rows")
        self.data[col_name] = col_data
      end
      if #self.columns == 0 then
        table.sort(cols)
        self.columns = cols
        self.col2id  = invert(cols)
      end
    elseif data then
      assert(class.of(data), "Needs a matrix or dictionary in argument data")
      local nd = data.num_dim
      assert(nd and nd(data)==2, "Needs a bi-dimensional matrix in argument data")
      if #self.columns == 0 then
        self.columns = matrixInt32(data:dim(2)):linspace()
        self.col2id  = invert(self.columns)
      else
        assert(data:dim(2) == #self.columns, "Incorrect number of columns in data")
      end
      if #self.rows == 0 then
        self.rows = matrixInt32(data:dim(1)):linspace()
        self.row2id  = invert(self.rows)
      else
        assert(data:dim(1) == #self.rows, "Incorrect number of rows in data")
      end
      for j,col_name in ipairs(self.columns) do
        self.data[col_name] = data:select(2,j)
      end
    end
  end

local function dataframe_tostring(self)
  if not next(self.data) then
    return table.concat{
      "Empty data_frame\n",
      "Columns: ", stringfy(self.columns, "ascii"), "\n",
      "Rows: ", stringfy(self.rows, "ascii"), "\n",
    }
  else
    local tbl = { "data_frame\n" }
    for j,col_name in ipairs(self.columns) do
      table.insert(tbl, "\t")
      table.insert(tbl, col_name)
    end
    table.insert(tbl, "\n")
    for i,row_name in ipairs(self.rows) do
      table.insert(tbl, row_name)
      for j,col_name in ipairs(self.columns) do
        table.insert(tbl, "\t")
        table.insert(tbl, tostring(self.data[col_name][i]))
      end
      table.insert(tbl, "\n")
    end
    return table.concat(tbl)
  end
end
class.extend_metamethod(data_frame, "__tostring", dataframe_tostring)

data_frame.from_csv =
  function(path, params)
    local self = data_frame()
    local data = self.data
    local params = get_table_fields({
        header = { default=true },
        sep = { default=',' },
                                    }, params or {})
    assert(#params.sep == 1, "Only one character sep is allowed")
    local f = type(path)~="string" and path or io.open(path)
    if params.header then
      self.columns = parse_csv_line(f:read("*l"), params.sep)
      self.col2id = invert(self.columns)
      for _,col_name in ipairs(self.columns) do data[col_name] = {} end
    end
    local n = 0
    for row_line in f:lines() do
      n = n + 1
      local tbl = parse_csv_line(row_line, params.sep)
      if not #self.columns then
        self.columns = matrixInt32(#tbl):linspace()
        for _,col_name in ipairs(self.columns) do data[col_name] = {} end
      end
      assert(#tbl == #self.columns, "Not matching number of columns")
      for j,col_name in ipairs(self.columns) do
        data[col_name][n] = tbl[j]
      end
    end
    self.rows   = matrixInt32(n):linspace()
    self.row2id = invert(self.rows)
    if path ~= f then f:close() end
    return self
  end

methods.to_csv =
  function(self, path, params)
    local params = get_table_fields({
        header = { default=true },
        sep = { default=',' },
                                    }, params or {})
    local sep = params.sep
    local f = type(path)~="string" and path or io.open(path, "w")
    if params.header then
      f:write(concat(self.columns, sep))
      f:write("\n")
    end
    local data = self.data
    local tbl = {}
    for i,row_name in ipairs(self.rows) do
      for j,col_name in ipairs(self.columns) do
        tbl[j] = data[col_name][i]
      end
      f:write(table.concat(tbl, sep))
      f:write("\n")
      table.clear(tbl)
    end
    if path ~= f then f:close() end
  end

local df1 = data_frame()
local df2 = data_frame{ data = { one = {1,2,3,4},
                                 two = {5,6,7,8} },
                        columns = { "two", "one" } }
local df3 = data_frame{ data = matrix(4,20):linear() }
print(df1)
print(df2)
print(df3)

local df4 = data_frame.from_csv(aprilio.stream.input_lua_string[[id,cost
1,4
2,1
3,10
]])
print(df4)

df4:to_csv("blah.csv")

return data_frame
