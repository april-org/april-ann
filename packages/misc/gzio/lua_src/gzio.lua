gzio = gzio or {}

gzio.open  = function(...)
  return gzio.stream(...)
end

gzio.lines = function(name, ...)
  local f = april_assert( gzio.open(name), "cannot open file '%s'", name )
  return f:lines(...)
end

april_io.register_open_by_extension("gz", gzio.open)
april_io.register_open_by_extension("tgz", gzio.open)
