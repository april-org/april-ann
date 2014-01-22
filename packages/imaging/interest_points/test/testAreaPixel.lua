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

print (result)
ImageIO.write(result,"classified.png", "png")
