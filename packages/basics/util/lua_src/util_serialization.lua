local MAGIC = "-- LS0001"
local FIND_MASK = "^" .. MAGIC:gsub("%-","%%-")

-- utilities for function serialization
local function char(c) return ("\\%03d"):format(c:byte()) end
local function szstr(s) return ('"%s"'):format(s:gsub("[^ !#-~]", char)) end

function util.function_setupvalues(func, upvalues)
  for i,value in ipairs(upvalues) do
    debug.setupvalue(func, i, value)
  end
  return func
end

function util.function_to_lua_string(func,format)
  --
  local func_dump = string.format("load(%s)", szstr(string.dump(func)))
  local upvalues = {}
  local i = 1
  while true do
    local name,value = debug.getupvalue(func,i)
    if not name then break end
    -- avoid global environment upvalue
    if name ~= "_ENV" then
      upvalues[i] = value
    end
    i = i + 1
  end
  --
  local t = {
    "util.function_setupvalues(",
    func_dump,
    ",",
    table.tostring(upvalues,format),
    ")"
  }
  return table.concat(t, "")
end

function table.tostring(t,format)
  format = format or "binary"
  if t.to_lua_string then return t:to_lua_string(format) end
  local out  = {}
  -- first serialize the array part of the table
  local j=1
  while t[j] ~= nil do
    local v = t[j]
    local value
    local tt = luatype(v)
    if tt == "table" then value = table.tostring(v,format)
    elseif tt == "string" then value = string.format("%q",v)
    elseif tt == "userdata" then
      assert(v.to_lua_string, "Needs to_lua_string method")
      value = v:to_lua_string(format)
    elseif tt == "function" then
      value = util.function_to_lua_string(v,format)
    else value = tostring(v)
    end
    table.insert(out, value)
    j=j+1
  end
  -- extract all keys removing array part
  local keys = iterator(pairs(t)):select(1):
  filter(function(key) local k = tonumber(key) return not k or k<=0 or k>=j end):table()
  -- sort the keys in order to obtain a deterministic result
  table.sort(keys, function(a,b) return tostring(a) < tostring(b) end)
  -- serialize non array part of the table
  for _,i in ipairs(keys) do
    local v = t[i]
    local key
    local value
    if luatype(i) == "string" then
      key = string.format("[%q]=",i)
    elseif tonumber(i) then key = "["..i.."]".."="
    else key = string.format("[%q]=",tostring(i))
    end
    local tt = luatype(v)
    if tt == "table" then value = table.tostring(v,format)
    elseif tt == "string" then value = string.format("%q",v)
    elseif tt == "userdata" then
      assert(v.to_lua_string, "Needs to_lua_string method")
      value = v:to_lua_string(format)
    elseif tt == "function" then
      value = util.function_to_lua_string(v,format)
    else value = tostring(v)
    end
    table.insert(out, key .. value)
  end
  return "{"..table.concat(out,",").."}"
end

function util.to_lua_string(data,format)
  local tt = luatype(data)
  if tt == "table" then
    if data.to_lua_string then
      return data:to_lua_string(format)
    else
      return table.tostring(data,format)
    end
  elseif tt == "string" then
    return string.format("%q", data)
  elseif tt == "function" then
    return util.function_to_lua_string(data,format)
  elseif tt == "userdata" then
    local idx = getmetatable(data)
    assert(idx, "Userdata needs a to_lua_string(format) method")
    idx = get_index(idx)
    assert(idx and idx.to_lua_string,
           "Userdata needs a to_lua_string(format) method")
    return data:to_lua_string(format)
  else
    return tostring(data)
  end
end

-- forward declaration
local serialize
do
  local non_structured = {
    string=true,
    number=true,
    boolean=true,
  }
  -- mapper class, it allows to map any Lua object to integers, allowing to
  -- enumerate uniquely all serialized objects
  local mapper_mt = {
    __index = {
      -- searchs a Lua object and stores it if needed
      add = function(self, obj)
        local data = self.data
        local id = data[obj]
        if not id then local n=self.n+1 id=n data[obj]=n self.n=n end
        return id
      end,
      -- returns the integer associated with a given Lua object (or nil in case
      -- of failure)
      consult = function(self, obj) return self.data[obj] end
    }
  }
  local function mapper()
    return setmetatable({ n=0, data={} }, mapper_mt)
  end
  -- lua_string_stream class, inserts Lua strings into a Lua table,
  -- and at the end it is possible to concatenate them all together.
  local lua_string_stream_mt = {
    __index = {
      write = function(self,data)
        table.insert(self, data)
      end,
      concat = function(self)
        return table.concat(self)
      end,
      is_lua_string_stream = true,
    },
  }
  local function lua_string_stream()
    return setmetatable({}, lua_string_stream_mt)
  end
  -- normalizes values converting them into strings
  local function value2str(data, tt)
    local tt = tt or type(data)
    assert(tt ~= "thread", "Unable to serialize coroutines")
    if tt == "string" then return "%q"%{data}
    else return tostring(data)
    end
  end
  local function non_structured_split(data)
    local n = 0
    local non_st = {}
    for k,v in pairs(data) do
      local tk = type(k)
      local tv = type(v)
      if non_structured[tk] and non_structured[tv] then
        non_st[k]=v
        n=n+1
      end
    end
    return n,non_st
  end
  -- transforms a given Lua object (data), returning a string with the
  -- transformed object
  local function transform(map, varname, data, destination)
    local tt = type(data)
    -- plain types are returned as is
    if non_structured[tt] then return value2str(data, tt) end
    local id = map:consult(data)
    -- If data is not found in the map, it is necessary to process it and
    -- transform all of its dependencies. Otherwise, the id is used to retrieve
    -- the transformed data from previously id position in varname.
    if not id then
      id = map:add(data)
      if tt == "table" then
        -- split table into non-structured table data (string and numbers)
        local n,non_st = non_structured_split(data)
        -- creates a new table, and subsequently traverses all its content
        destination:write("%s[%d]="%{varname,id})
        if n==0 then destination:write("{}") -- empty table
        else destination:write(table.tostring(non_st)) end -- table with data
        destination:write("\n")
        for k,v in pairs(data) do
          if not non_st[k] then
            local kstr = transform(map, varname, k, destination)
            local vstr = transform(map, varname, v, destination)
            destination:write("%s[%d][%s]=%s\n"%{varname,id,kstr,vstr})
          end
        end
      elseif tt == "function" then
        -- serializes the function with all of its upvalues
        -- NOTE that upvalues which are plain objects (strings or integers)
        -- are not shared between functions, but tables and userdata objects
        -- will be shared when serializing together several functions.
        local upvalues = {}
        local i = 1
        while true do
          local name,value = debug.getupvalue(data,i)
          if not name then break end
          -- avoid global environment upvalue
          if name ~= "_ENV" then upvalues[i] = value end
          i = i + 1
        end
        local upv_str = transform(map, varname, upvalues, destination)
        local func_dump = "load(%s)"%{ szstr(string.dump(data)) }
        destination:write("%s[%d]=util.function_setupvalues(%s,%s)\n"%
                            {varname,id,func_dump,upv_str})
      elseif class.of(data) then
        local serialize = getmetatable(data).serialize
        if serialize then
          destination:write("%s[%d]="%{varname,id})
          if destination.is_lua_string_stream then
            destination:write("%s"%{serialize(data)})
          else
            serialize(data, destination)
          end
          destination:write("\n")
        else
          -- it is an object, so we need to use its introspection methods
          assert(data.ctor_params,
                 "Userdata needs a function called ctor_params_table to be serializable")
          assert(data.ctor_name,
                 "Userdata needs a function called ctor_name to be serializable")
          local params = table.pack( data:ctor_params() )
          local ctor_name = data:ctor_name()
          local params_str = ""
          if params.n > 0 then
            local needs_unpack = true
            if params.n == 1 then
              params = params[1] needs_unpack=false
            else
              params.n = nil
            end
            params_str = transform(map, varname, params, destination)
            if needs_unpack then params_str = "table.unpack(%s)"%{params_str} end
          end
          destination:write("%s[%d]=%s(%s)\n"%{varname,id,ctor_name,params_str})
        end
      else
        -- general case
        local value = value2str(data)
        destination:write("%s[%d]=%s\n"%{varname,id,value})
      end
    end
    return "%s[%d]"%{varname,id}
  end
  
  serialize =
    april_doc{
      class = "function",
      summary = "Serializes an object to a filename or a string",
      params = {
        "Any object, table, string, number, ...",
        "A filename [optional], if not given an string would be returned",
        "Format string: 'ascii' or 'binary' [optional]",
      },
      outputs = {
        "In case of nil second argument, this function",
        "returns a string with the serialized object.",
      },
    } ..
    function(data, destination, format)
      local version = { util.version() } table.insert(version, os.date())
      local comment = "-- version info { major, minor, commit number, commit hash, date }"
      local version_info = ",\n%s\n%s\n"%{ comment,
                                           util.to_lua_string(version, format) }
      --
      local map = mapper()
      local destination = destination or lua_string_stream()
      local do_close = false
      if type(destination)=="string" then
        destination = io.open(destination, "w")
        do_close = true
      end
      local varname = "_"
      destination:write(MAGIC)
      destination:write("\n")
      destination:write("local %s={}\n"%{varname})
      local str = transform(map, "_", data, destination, format)
      destination:write("return %s%s"%{str,version_info})
      if type(destination) == "table" then
        return destination:concat()
      elseif do_close then
        destination:close()
      end
    end
end

local deserialize =
  april_doc{
    class = "function",
    summary = "Deserializes an object from a filename or a string",
    params = {
      "A string with a filename or a serialized object",
      { "... a variadic list of arguments to be passed to",
        "the object during deserialization", },
    },
    outputs = {
      "A deserialized object",
    },
  } ..
  function(destination, ...)
    local result
    if type(destination) == "string" then
      local loader
      if destination:find(FIND_MASK) or destination:find("^return ") then
        -- it is a previously serialized string
        loader = assert( loadstring(destination) )
      else
        loader = assert( loadfile(destination) )
      end
      result = table.pack( loader(...) )
    elseif iscallable(from) then
      local f = load(from())
      result = table.pack( f(...) )
    else
      assert(destination.read, "Needs a string or an open file")
      result = deserialize(destination:read("*a"))
    end
    return table.unpack( result )
  end

------------------------------------------------------------------------------

util.serialize   = serialize
util.deserialize = deserialize