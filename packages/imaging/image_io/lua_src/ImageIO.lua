-- ImageIO.handlers is a table {string->{read=function, write=function} which contains
-- handlers for reading and writing image files.
ImageIO={}
ImageIO.handlers={}

-- ImageIO.read: Reads a image from a file.
--
-- params:
--   filename: name of the image file to be read
--   img_format[optional, defaults to the file extension]: image format
--
-- return value: a ImageRGB containing the read image
--
function ImageIO.read(filename, img_format)
  img_format = img_format or string.get_extension(filename)
  img_format = string.lower(img_format)

  local format_handler = ImageIO.handlers[img_format]

  if format_handler ~= nil then
    return format_handler.read(filename)
  else
    -- TODO: call convert, read ppm
    error(string.format("Image format '%s' not supported", img_format))
  end
end

-- ImageIO.write: Writes a image to a file.
--
-- params:
--   img: a ImageRGB to be written
--   filename: name of the image file to be written
--   img_format[optional, defaults to the file extension]: image format
--
-- return value: none
--   
function ImageIO.write(img, filename, img_format)
  -- TODO: Add proper grayscale image support
  if type(img) == "Image" then img = img:to_RGB() end

  img_format = img_format or string.get_extension(filename)
  img_format = string.lower(img_format)

  local format_handler = ImageIO.handlers[img_format]
  
  if format_handler ~= nil then
    return format_handler.write(img, filename)
  else
    -- TODO: write ppm, call convert
    error(string.format("Image format '%s' not supported", img_format))
  end
end


