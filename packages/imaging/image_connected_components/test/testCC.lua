-- Create a simple matrix
--

m = matrix(5, 6, { 0, 1, 1, 1, 1, 1, 
                   0, 0, 1, 1, 0, 0, 
                   0, 1, 1, 1, 1, 0,
                   1, 0, 1, 1, 1, 1, 
                   1, 1, 1, 1, 0, 1})

-- Create the image
local myImg = Image(m)

local n_comps = image.test_connected_components(myImg)

print (myImg:geometry())
print(n_comps)
local comps = image.connected_components(myImg)

local index_matrix = comps:get_pixel_matrix()

local color_matrix = comps:get_colored_image()


ImageIO.write(color_matrix, "matrix.png", "png")
print (m)
print (index_matrix)

print ("Bounding boxes")

bbs = comps:get_bounding_boxes()
for i,v in ipairs(bbs) do

  print (v[1], v[2], v[3], v[4])
end
