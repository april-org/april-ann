imageName = arg[1]
top    = tonumber(arg[2])
bottom    = tonumber(arg[3])
imageOut = arg[4]

img = ImageIO.read(imageName):to_grayscale()
img = img:add_rows(top, bottom, 1.0)
ImageIO.write(img, imageOut, "png")


