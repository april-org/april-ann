-- Create a simple matrix
--

m = matrix(5, 6, { 0, 1, 1, 1, 1, 1, 
                   0, 0, 1, 1, 0, 0, 
                   0, 1, 1, 1, 1, 0,
                   1, 0, 1, 1, 1, 1, 
                   1, 1, 1, 0, 0, 1})

-- Create the image
local myImg = Image(m)

local n_comps = image.test_connected_components(myImg)

print (myImg:geometry())
print(n_comps)
local comps = image.connected_components(myImg)

local index_matrix = comps:get_pixel_matrix()

print (m)
print (index_matrix)
