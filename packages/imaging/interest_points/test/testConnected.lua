imgFile = arg[1]
netFile = arg[2]

local width = 500
local height = 250
local miniwidth = 50
local miniheight = 30

img = ImageIO.read(imgFile):to_grayscale()

-- Generates the CCs of the image
cps = interest_points.ConnectedPoints(img)

pc = interest_points.pointClassifier(
-- compute interest_points
print(cps)


