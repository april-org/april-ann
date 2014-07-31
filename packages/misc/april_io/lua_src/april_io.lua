-- modificamos el io.open
april_io = {}
local open_by_extension = { }
local lines_by_extension = { }
local io_old_open = io.open
io.open = function(name, ...)
  local open = (name and open_by_extension[string.get_extension(name)]) or io_old_open
  return open(name, ...)
end

local io_old_lines = io.lines
io.lines = function(name, ...)
  local lines = (name and lines_by_extension[string.get_extension(name)]) or io_old_lines
  return lines(name, ...)
end

function april_io.register_open_by_extension(ext, func)
  open_by_extension[ext] = func
end
function april_io.register_lines_by_extension(ext, func)
  lines_by_extension[ext] = func
end

april_io.lua_open  = io_old_open
april_io.lua_lines = io_old_lines

return april_io
