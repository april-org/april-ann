local rnd=random(1234)
local kx=17
local ky=17
local h=10
-- a matrix of ROWSxCOLUMNSx3
--m = ImageIO.read(string.get_path(arg[0]) .. "photo.png"):matrix()
local m = ImageIO.read("packages/ann/ann/test/photo.png"):matrix():transpose():clone()
print(m)
local m = m:padding(0,0,8,8,8,8)
local k = matrix(h,3,kx,ky):uniformf(-0.1,0.1,rnd)
local o = m:convolution{ kernel=k, D=2 }
print(o)
local x,y = o:dim(3),o:dim(4)
local o = o:rewrap(o:dim(2), o:size()/o:dim(2))
local img = ann.connections.input_filters_image(o, {x,y})
ImageIO.write(img, "wop.png")
