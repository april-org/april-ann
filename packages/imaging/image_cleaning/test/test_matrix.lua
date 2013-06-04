
m = matrix.fromString[[
4 4
ascii
1 0 1 0
1 1 0 1
1 0 1 1
0 0 1 0
]]

myImg = Image(m)

myHist = image.image_histogram(myImg,2)

hist_integral = myHist:get_integral_histogram()

local height = m:dim()[1]
local width  = m:dim()[2]

--
print(hist_integral:toString())
printf("Image Histogram!\n")

hist_window = myHist:generate_window_histogram(1)
for i = 1, height do
    for j = 1, width do
        printf("%f ", hist_window:get(i,j,2))
    end
    printf("\n");
end

print(hist_window:toString())

