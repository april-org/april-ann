img = ImageIO.read(arg[1]):to_grayscale():invert_colors()

-- Get the matrix indexes
threshold = arg[2] or 0.5
indexes = img:get_indexes(threshold)
print("Indexes", indexes)
print("Dim", #indexes:dim())
local n_indexes = indexes:dim(1)
local w, h = img:geometry()

local total = w*h
printf("%d/%d (%.5f) pixels are above the threshold (%f)\n",
n_indexes, total, n_indexes/total, threshold)


