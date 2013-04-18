imgFile = arg[1]

img = ImageIO.read(imgFile):to_grayscale()

interest_points.extract_points_from_image(img)
