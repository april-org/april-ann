imageName = arg[1]
radius    = tonumber(arg[2])

print(imageName, radius)

img = ImageIO.read(imageName):to_grayscale()

img_median = img:median_filter(radius)

img_clean = img:substract(img_median)

ImageIO.write(img_median, "median.png", "png")
ImageIO.write(img_clean, "substract.png", "png")


