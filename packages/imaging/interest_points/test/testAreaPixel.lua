imgFile = "sample_iam.png"
pointsFile = "sample_iam.txt"

img = ImageIO.read(imgFile):to_grayscale()


points = {{},{},{},{},{}}

-- Open the interest points list
for line in io.open(pointsFile, "r"):lines() do
    point = map(tonumber, ipairs(string.tokenize(line)))
    if point[3] <= 5 then
     table.insert(points[point[3]], point)
    end
end

for i,v in ipairs(points) do
    print (i, #v)
end

result = interest_points.classify_pixel(img, points[1],
points[2],points[4], points[5])

ImageIO.write(result,"classified.png", "png")
-- Load The dataset image

local m_indexes = interest_points.get_indexes_from_colored(result)

local num_classes = 3;

w,h = result:geometry()
print(w, h)
local dsIndex = dataset.identity(num_classes)
local dsPositions = dataset.matrix(m_indexes,{ 
         patternSize = {1,1},
         offset = {0,0},
         numSteps = {m_indexes:dim(1),1},
})

local dsTags = dataset.matrix(m_indexes,{ 
         patternSize = {1,1},
         offset = {0,1},
         numSteps = {m_indexes:dim(1),1},
})

local dsSoftmax = dataset.indexed(dsTags, {dsIndex})
local computed = interest_points.get_image_area_from_dataset(dsSoftmax, dsPositions, w, h)

ImageIO.write(computed,"computed.png", "png")


