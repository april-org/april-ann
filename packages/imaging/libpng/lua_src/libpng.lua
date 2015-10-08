ImageIO.handlers["png"] = { read=libpng.read, write=libpng.write }

-- support for IPyLua
local function ImageRGB_show(obj)
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

-- support for IPyLua
class.extend_metamethod(ImageRGB, "ipylua_show", ImageRGB_show)

local function Image_show(obj)
  local obj = obj:to_RGB()
  return ImageRGB_show(obj)
end

-- support for IPyLua
class.extend_metamethod(Image, "ipylua_show", Image_show)
