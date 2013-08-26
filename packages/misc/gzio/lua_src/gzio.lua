-- modificamos el io.open
local io_old_open = io.open
io.open = function(name, mode)
	    local f
	    if string.get_extension(name) == "gz" then
	      f = gzio.open(name, mode)
	    else f = io_old_open(name, mode)
	    end
	    return f
	  end

local io_old_lines = io.lines
io.lines = function(name, ...)
  if name~=nil and string.get_extension(name) == "gz" then
    local arg = { ... }
    local f = gzio.open(name)
    return function()
      local values = { f:read(unpack(arg)) }
      if #values == 0 or values[1] == nil then f:close() return nil end
      return unpack(values)
    end
  else return io_old_lines(name, ...) end
end

function gzio.meta_instance.lines(self, ...)
  local arg = { ... }
  return function()
    local values = { self:read(unpack(arg)) }
    if values[1] == nil then return nil end
    return unpack(values)
  end
end
