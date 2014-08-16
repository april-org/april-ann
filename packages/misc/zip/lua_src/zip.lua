zip = zip or {}

zip.open = function(...)
  return zip.package(...)
end

aprilio.register_open_by_extension("zip", zip.open)
