-- usage: april median_filter.lua orig.png radio dest.png

orig_filename = arg[1]
radio         = tonumber(arg[2])
dest_filename = arg[3]

print"loading"
image = ImageIO.read(orig_filename, "png"):to_grayscale()
print"filtering"
print(image)
dest_image = image:median_filter(radio)
print"saving"
ImageIO.write(dest_image:to_RGB(), dest_filename, "png")
print"done!"


