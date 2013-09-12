
m = matrix.fromString[[
4 2
ascii
1 0 
1 1 
1 0 
1 0 
]]

myImg = Image(m)
local height = m:dim()[1]
local width  = m:dim()[2]

print("Image", height, width)
print("Integral Histogram Computation")
levels = 2
myHist = image.image_histogram(myImg, levels)
hist = myHist:get_image_histogram()

print(hist:dim()[1])
for i =1, levels do
  print(i,hist:get(i))
end

print("Greedy Histogram Computation")
levels = 2
hist = image.image_histogram.get_histogram(myImg, levels)

print(hist:dim()[1])
for i =1, levels do
  print(i,hist:get(i))
end

print("Horizontal Histogram Computation")
hist = myHist:get_horizontal_histogram()

for i = 1, height do
    for l =1, levels do
        print(i,l, hist:get(i,l))
    end
end

print("Horizontal Radio Histogram Computation")
hist = myHist:get_horizontal_histogram(1)

for i = 1, height do
    for l =1, levels do
        print(i,l, hist:get(i,l))
    end
end

print("Vertical Histogram Computation")
hist = myHist:get_vertical_histogram()

for i = 1, width do
    for l =1, levels do
        print(i,l, hist:get(i,l))
    end
end


print("Vertical window histogram Computation")
hist = myHist:get_vertical_histogram(1)

for i = 1, width do
    for l =1, levels do
        print(i,l, hist:get(i,l))
    end
end


