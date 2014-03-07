imgFile = arg[1]

img = ImageIO.read(imgFile):to_grayscale()


--mySVG = imageSVG({width=300, height = 300})
--mySVG:setHeader({})
--mySVG:setFooter({})
        
w, h = img:geometry()
print(w, h)

mySVG = imageSVG.fromImageFile(imgFile, w, h)
--img = img:invert_colors()
uppers, lowers = interest_points.extract_points_from_image(img)
print(#uppers)
print(#lowers)
print("uppers", uppers[1][1], uppers[1][2])
outFile = string.remove_extension(imgFile)..".svg" 

for _, p in ipairs(uppers) do
    mySVG:addSquare(p, {})


end
for _, p in ipairs(lowers) do
    mySVG:addSquare(p, {color = "red"})
end
mySVG:write(outFile)
print(#uppers)
print(#lowers)
