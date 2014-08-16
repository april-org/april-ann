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

return aprilio
