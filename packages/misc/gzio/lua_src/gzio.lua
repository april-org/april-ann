local function gzio_lines(self, ...)
  local arg = { ... }
  return function()
    local values = { self:read(table.unpack(arg)) }
    if #values == 0 or values[1] == nil then return nil end
    return table.unpack(values)
  end
end
gzio.lines = function(name, ...)
  local f = april_assert( gzio.open(name), "cannot open file '%s'", name )
  return gzio_lines(f, ...)
end
class_extension(gzio, "lines", gzio_lines)

april_io.register_open_by_extension("gz", gzio.open)
april_io.register_lines_by_extension("gz", gzio.lines)
