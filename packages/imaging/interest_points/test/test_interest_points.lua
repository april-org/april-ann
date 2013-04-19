imgFile = arg[1]

img = ImageIO.read(imgFile):to_grayscale()


--mySVG = imageSVG({width=300, height = 300})
--mySVG:setHeader({})
--mySVG:setFooter({})
        
w, h = img:geometry()
print(w, h)

mySVG = imageSVG.fromImageFile(imgFile, w, h)
--img = img:invert_colors()
points = interest_points.extract_points_from_image(img)

outFile = string.remove_extension(imgFile)..".svg" 

for _, p in ipairs(points) do
    mySVG:addSquare(p, {})


end
mySVG:write(outFile)
print(#points)
