img = Image.empty(300,300)
src = Image.load("a01-000u-s00-02.png") 
img:copy(src, -300, 100)
Image.save(img, "prueba.pgm")
