filename = arg[1]
neighbors = tonumber(arg[2])
mlpFile = arg[3]
bunchsize = tonumber(arg[4]) or 32


local fileout  = string.remove_extension(filename).."-clean.png"

print(filename.." -> "..fileout)

local mlpClean = ann.mlp.all_all.load(mlpFile, bunchsize)
local imgDirty = ImageIO.read(filename):to_grayscale()
local imgClean = image.image_cleaning.apply_filter_std(imgDirty, neighbors, mlpClean)

ImageIO.write(imgClean, fileout)


