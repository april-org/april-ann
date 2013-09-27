imgFile = "sample2.png"
pointsFile = "sample2.txt"

img = ImageIO.read(imgFile):to_grayscale()

-- Generates the CCs of the image
cps = interest_points.ConnectedPoints(img)

-- Open the interest points list
for line in io.open(pointsFile, "r"):lines() do
    point = string.tokenize(line)
    print("Adding ", point[1], point[2], point[3])
    cps:addPoint(point)
end

print("Total added: ", cps:getNumPoints())

print("Components\n")

cps:printComponents()

cps:sortByConfidence()

print("Sorted components\n")
cps:printComponents()


t = cps:getComponentPoints()

print("I have ", #t, "components")

for i,component in ipairs(t) do
    print ("Component ", i) --component, #component)
  for j, point in ipairs(component) do
      print(point[1], point[2], point[3])
  end
end
