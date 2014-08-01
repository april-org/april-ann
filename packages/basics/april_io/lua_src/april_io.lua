-- modificamos el io.open
april_io = april_io or {}
local open_by_extension = { }

local io_old_open = io.open
april_io.open = function(name, ...)
  local open = (name and open_by_extension[string.get_extension(name)]) or io_old_open
  return open(name, ...)
end
io.open = april_io.open

local io_old_lines = io.lines
april_io.lines = function(name, ...)
  if not name then return io_old_lines() end
  local open = open_by_extension[string.get_extension(name)] or io_old_open
  local f = april_assert(open(name), "cannot open file '%s'", name)
  assert(f.lines, "lines method not implemented to this kind of file")
  return f:lines(...)
end
io.lines = april_io.lines

function april_io.register_open_by_extension(ext, func)
  open_by_extension[ext] = func
end

class_extension(april_io.file, "lines",
                function(self, ...)
                  local arg = { ... }
                  return function()
                    local values = { self:read(table.unpack(arg)) }
                    if #values == 0 or values[1] == nil then return nil end
                    return table.unpack(values)
                  end
end)

april_io.lua_open  = io_old_open
april_io.lua_lines = io_old_lines

return april_io
