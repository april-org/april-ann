img = Image.load(arg[1])

sin = math.sin
cos = math.cos
pi = math.pi

a=45*pi/180
img2 = img:affine_transform(AffineTransform2D():scale(0.25,0.5):rotate(a), 0)

Image.save(img2, "test-result.pgm")

