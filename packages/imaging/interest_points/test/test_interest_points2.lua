imgFile = arg[1]
mlpFile     = arg[2]
bunchsize   = arg[3] or 64

img = ImageIO.read(imgFile):to_grayscale()
-- Load the image
print (img_inv, img)
w, h = img:geometry()
print(mlpFile)
mlp = ann.mlp.all_all.load(mlpFile, bunchsize)

mySVG = imageSVG.fromImageFile(imgFile, w, h)
--img = img:invert_colors()
uppers, lowers = interest_points.extract_points_from_image(img)
print(#uppers)
print(#lowers)

outFile = string.remove_extension(imgFile)..".svg" 
point = uppers[1]


pc = interest_points.pointClassifier(500,250,50, 30, false)


res = pc:compute_point(img, point, mlp)
cl = pc:classify_point(img, point, mlp)


-- Classify uppers
local uppers_classified = pc:classify_points(img, uppers, mlp)
--local uppers_res  = pc:compute_points(img_inv, uppers, mlp)

--- Transform in tables
local tables = interest_points.sort_by_class(uppers_classified, 5)
mySVG:addPointsFromTables(tables)

print("Tables: ", #tables)
for i=1, 5 do
  print(i, #tables[i])
end

-- Classify lowers
local lowers_classified = pc:classify_points(img, lowers, mlp)
local lower_res  = pc:compute_points(img, uppers, mlp)

--- Transform in tables
tables = interest_points.sort_by_class(lowers_classified, 5)
mySVG:addPointsFromTables(tables)

print("Tables: ", #tables)
for i=1, 5 do
  print(i, #tables[i])
end
--]]
mySVG:write(outFile)
print(#uppers)
print(#lowers)
