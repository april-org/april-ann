-- modificamos el io.open
aprilio = aprilio or {}
local open_by_extension = { }

local io_old_open = io.open
aprilio.open = function(name, ...)
  local open = (name and open_by_extension[string.get_extension(name)]) or io_old_open
  return open(name, ...)
end
io.open = aprilio.open

local io_old_lines = io.lines
aprilio.lines = function(name, ...)
  if not name then return io_old_lines() end
  local open = open_by_extension[string.get_extension(name)] or io_old_open
  local f = april_assert(open(name), "cannot open file '%s'", name)
  assert(f.lines, "lines method not implemented to this kind of file")
  local it = f:lines(...)
  return function()
    local result = table.pack(it())
    if #result == 0 then
      f:close()
    else
      return table.unpack(result)
    end
  end
end
io.lines = aprilio.lines

function aprilio.register_open_by_extension(ext, func)
  open_by_extension[ext] = func
end

class.extend(aprilio.stream, "lines",
             function(self, ...)
               local arg = { ... }
               return function()
                 return self:read(table.unpack(arg))
               end
end)

aprilio.lua_open  = io_old_open
aprilio.lua_lines = io_old_lines

class.extend(aprilio.package, "files",
             function(self)
               local i,n = 0,self:number_of_files()
               return function()
                 if i<n then i=i+1 return self:name_of(i) end
               end
end)

---------------------------------------------------------------------------

local function to_lua_string(self, ...)
  local params = table.pack(self:ctor_params())
  local params_str = ""
  if params.n > 0 then
    local needs_unpack = true
    if params.n == 1 then
      params = params[1] needs_unpack=false
    else
      params.n = nil
    end
    params_str = table.tostring(params, ...)
    if needs_unpack then params_str = "table.unpack(%s)"%{params_str} end
  end
  return "%s(%s)"% { self:ctor_name(), params_str }
end

class.extend(aprilio.serializable, "to_lua_string", to_lua_string)

local lua_serializable,
lua_serializable_methods = class("aprilio.lua_serializable")
-- global declaration
aprilio.lua_serializable = lua_serializable

lua_serializable_methods.to_lua_string = to_lua_string
lua_serializable_methods.ctor_name = function()
  error("Serialization not implemented")
end
lua_serializable_methods.ctor_params = function()
  error("Serialization not implemented")
end
lua_serializable_methods.save = util.serialize
lua_serializable.load         = util.deserialize

---------------------------------------------------------------------------

april_set_doc(aprilio.serializable.."write", {
		class = "method",
		summary = "It allows to store an object into a stream.",
		description ={
		  "It allows to store an object into a stream.",
		  "It uses a format expected by read method.",
		},
		params = {
		  { "A aprilio.stream instance. If not given any,",
                    "the object is serialized to a Lua string", },
		  "A Lua table with options [optional].",
		},
                outputs = {
                },
})

april_set_doc(aprilio.serializable.."write", {
		class = "method",
		summary = "It allows to store an object into a Lua string.",
		description ={
		  "It allows to store an object into a Lua string.",
		  "It uses a format expected by read method.",
		},
		params = {
		  "A Lua table with options [optional].",
		},
                outputs = {
                  "A Lua string with the serialization result.",
                },
})

return aprilio
