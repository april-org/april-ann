imgFile = "sample2.png"
pointsFile = "sample2.txt"

img = ImageIO.read(imgFile):to_grayscale()

-- Generates the CCs of the image
cps = interest_points.ConnectedPoints(img)

-- Open the interest points list
for line in io.open(pointsFile, "r"):lines() do
    point = map(tonumber, ipairs(string.tokenize(line)))
    print (point[3])
    if point[3] == 4 then 
        print("Adding ", point[1], point[2], point[3])
        cps:addPoint(point)
    end
end

print("Total added: ", cps:getNumPoints())

print("Components\n")

cps:printComponents()

cps:sortByX()

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


width, height = img:geometry()

mySVG = imageSVG.fromImageFile(imgFile, width, height)
mySVG:addPaths(t)
mySVG:write("sample2.svg")
