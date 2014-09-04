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
local aux = stats.mean_centered_by_pattern(m:clone("col_major"))
local aU,aS,aVT = stats.pca(aux)
local takeN,eigen_value,prob_mass=stats.pca_threshold(aS, 0.99)

-- PCA THRESHOLD STATISTICS
T("PCAThresholdTest",
  function()
    local takeN,eigen_value,prob_mass=stats.pca_threshold(aS, 0.99)
    -- print(takeN, eigen_value, prob_mass)
    check.eq(takeN, 192)
    check.lt(math.abs(eigen_value-0.01752162), 1e-03)
    check.lt(math.abs(prob_mass-0.9897367), 1e-03)
end)

local zca_whitening,new
if ann.components.zca_whitening then
  zca_whitening = ann.components.zca_whitening{
    U=aU,
    S=aS,
    epsilon=0.017,
    takeN=192,
  }
  new = zca_whitening:forward(aux):get_matrix()
end

local new2 = stats.zca_whitening(aux:clone(), aU(':','1:192'),
				 aS('1:192','1:192'), 0.017)

if new then
  T("ZCAWhiteningTest",
    function()
      for i=1,new:dim(1) do
        local d = new(i,':'):clone("row_major"):rewrap(16,16):adjust_range(0,1)
        local d2 = new2(i,':'):clone("row_major"):rewrap(16,16):adjust_range(0,1)
        check.eq(d, d2)
        -- ImageIO.write(Image(d), "wop-" .. string.format("%03d",i) .. ".png")
      end
  end)
end

-- PCA FILTERS
-- for i=1,aU:dim(1) do
--   local d = aU:select(2,i):clone("row_major"):rewrap(16,16):adjust_range(0,1)
--   ImageIO.write(Image(d), "filter-" .. string.format("%03d",i) .. ".png")
-- end
