img = Image.load(arg[1])

img2 = img:resize(arg[2], arg[3])

Image.save(img2, "test-result.pgm")

