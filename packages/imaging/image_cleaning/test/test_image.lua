-- Create a simple Hist object
-- Open the image
imgFile = arg[1] or "D09TEH01.png"
print("ImgFile " .. imgFile)
myImg = ImageIO.read(imgFile):to_grayscale()

mHistogram = myImg:get_window_histogram(4,5)

print("Histogram ", mHistogram)
