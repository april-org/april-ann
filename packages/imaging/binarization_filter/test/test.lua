filename = arg[1] or 'D09TRHW01-1.png'
win = arg[2] or 7


gray_image = ImageIO.read(filename):to_grayscale()

black_image = gray_image:binarize_niblack_simple(7)

ImageIO.write(black_image:to_RGB(),filename.."-niblack.png")

black_image = gray_image:binarize_otsus()

ImageIO.write(black_image:to_RGB(),filename.."-otsus.png")

black_image = gray_image:binarize_threshold(0.7)

ImageIO.write(black_image:to_RGB(),filename.."-threshold.png")
