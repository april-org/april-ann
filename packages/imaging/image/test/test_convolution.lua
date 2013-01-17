img = Image.load(arg[1])

img2 = img:convolution5x5(
  {0,0,0,0,0,
   0,0,0,0,0,
   0,0,-1,0,0,
   0,0,1,0,0,
   0,0,0,0,0})

Image.save(img2, "test-result.pgm")

