local img = ImageIO.read(arg[1])
local points_file = arg[2]


w, h = img:geometry()
local smoothed = matrix.fromTabFilename(points_file)

local bodyMat = matrix(w,2)

--Fill until first point
local first_col = smoothed:get(1,1)
local first_up = smoothed:get(1,2)
local first_low = smoothed:get(1,3)

local last_col  = smoothed:get(smoothed:dim(1),1)
local last_up  = smoothed:get(smoothed:dim(1),2)
local last_low  = smoothed:get(smoothed:dim(1),3)
for c=1, first_col do
  bodyMat:set(c, 1, first_up)
  bodyMat:set(c, 2, first_low)
end

for c = last_col, smoothed:dim(1) do
  bodyMat:set(c, 1, last_up)
  bodyMat:set(c, 2, last_low)
end
for i=1,smoothed:dim(1) do
    local col   = smoothed:get(i,1)
    local top   = smoothed:get(i,2)
    local lower = smoothed:get(i,3)
    bodyMat:set(col,1,top)
    bodyMat:set(col,2,lower)
end 

local finalMat = ocr.off_line_text_preprocessing.add_asc_desc(img:to_grayscale():invert_colors(), bodyMat)
local img2 = img:clone()




for i = 1, finalMat:dim(1) do
  local asc = math.max(0,math.round(finalMat:get(i,1)))
  local desc = math.min(h-1,math.round(finalMat:get(i,4)))
  local upper = finalMat:get(i,2)
  local lower = finalMat:get(i,3)


  print(i,asc, desc, w, h)
  img:putpixel(i,asc,1,0,0);
  img:putpixel(i,desc, 0,1,0);
  img:putpixel(i,upper,1,0,1);
  img:putpixel(i,lower, 0,0,1);
end

imgNorm = ocr.off_line_text_preprocessing.normalize_from_matrix(img2:to_grayscale():invert_colors(), 0.1,0.2,finalMat,40)

finalMat:toTabFilename("prueba.mat")
ImageIO.write(img, "prueba.png")
ImageIO.write(imgNorm:invert_colors(), "prueba-norm.png")
print (bodyMat)
