-- Create a simple Hist object
-- Open the image
imgFile = arg[1] or "D09TEH01.png"
print("ImgFile " .. imgFile)
myImg = ImageIO.read(imgFile):to_grayscale()

myHist = image.image_histogram(myImg, 4)
printf("Histogram created with %d gray levels\n", myHist:gray_levels() )
