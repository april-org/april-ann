local check = utest.check
local T = utest.test
--
local path = string.get_path(arg[0]).."../../../../TEST"
local m = ImageIO.read(path.."/digitos/digits.png"):invert_colors():to_grayscale():matrix()
local ds = dataset.matrix(m,
			  {
			    patternSize = {16,16},
			    offset      = {0,0},
			    numSteps    = {100,10},
			    stepSize    = {16,16},
			    orderStep   = {1,0}
			  })
local m = ds:toMatrix()
local aux = stats.pca.center_by_pattern(m:clone())
local aU,aS,aVT = stats.pca(aux)

-- PCA THRESHOLD STATISTICS
T("PCAThresholdTest",
  function()
    local takeN,eigen_value,prob_mass=stats.pca.threshold(aS, 0.99)
    check.eq(takeN, 193)
    check.number_eq(eigen_value,0.017318)
    check.number_eq(prob_mass,0.990025)
end)

local zca_whitening,new
if ann.components.zca_whitening then
  zca_whitening = ann.components.zca_whitening{
    U=aU,
    S=aS,
    epsilon=0.017,
    takeN=193,
  }
  new = zca_whitening:forward(aux)
end

local new2 = stats.zca.whitening(aux:clone(), aU(':','1:193'),
				 aS('1:193','1:193'), 0.017)

if new then
  T("ZCAWhiteningTest",
    function()
      for i=1,new:dim(1) do
        local d = new(i,':'):clone():rewrap(16,16):adjust_range(0,1)
        local d2 = new2(i,':'):clone():rewrap(16,16):adjust_range(0,1)
        check.eq(d, d2)
        -- ImageIO.write(Image(d), "wop-" .. string.format("%03d",i) .. ".png")
      end
  end)
end

-- PCA FILTERS
-- for i=1,aU:dim(1) do
--   local d = aU:select(2,i):clone():rewrap(16,16):adjust_range(0,1)
--   ImageIO.write(Image(d), "filter-" .. string.format("%03d",i) .. ".png")
-- end
