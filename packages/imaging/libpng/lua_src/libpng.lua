ImageIO.handlers["png"] = { read=libpng.read, write=libpng.write }

-- support for IPyLua
do
  local handlers = debug.getregistry().APRILANN.IPyLua_output_handlers

  handlers[ ImageRGB ] = function(obj)
    local data = {
      ["text/plain"] = tostring(obj),
      ["image/png"] = ee5_base64.encode(libpng.write(obj)),
    }
    local w,h = obj:geometry()
    local metadata = {
      ["image/png"] = { width = w, height = h }
    }
    return data,metadata
  end
  
  handlers[ Image ] = function(obj)
    local obj = obj:to_RGB()
    return handlers[ obj ]( obj )
  end
end
