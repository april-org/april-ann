imgFile = arg[1]
mlpFile     = arg[2]
bunchsize   = arg[3] or 64

img = ImageIO.read(imgFile):to_grayscale()
-- Load the image
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

pc = interest_points.pointClassifier(500,250,50, 30)


res = pc:compute_point(img, point, mlp)

print(res)
for i, v in ipairs(res) do
    print(v)
end
--[[for _, p in ipairs(uppers) do
mySVG:addSquare(p, {})

end
for _, p in ipairs(lowers) do
mySVG:addSquare(p, {color = "red"})


end
mySVG:write(outFile)
print(#uppers)
print(#lowers)
--]]
