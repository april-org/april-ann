filename  =  arg[1]
neighbors = tonumber(arg[2])
levels    = tonumber(arg[3])
radius    = tonumber(arg[4])
mlpFile = arg[5]
bunchsize = tonumber(arg[6]) or 32

local fileout  = string.remove_extension(filename).."-clean_hist.png"

print(filename.." -> "..fileout)
local mlpClean = ann.mlp.all_all.load(mlpFile, bunchsize)
local imgDirty = ImageIO.read(filename):to_grayscale()
local imgClean = image.image_cleaning.apply_filter_histogram(imgDirty, neighbors, levels, radius, mlpClean)

ImageIO.write(imgClean, fileout)
