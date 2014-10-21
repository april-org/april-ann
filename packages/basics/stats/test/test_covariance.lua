local check = utest.check
local T = utest.test
T("CovarianceCorrelationTest", function()
    local m = matrix(3,3,{1,6,9,
			  4,5,5,
			  7,4,1})
    check.eq(stats.cov(m,m),
	     matrix(3,3,{9,-3,-12,
			   -3,1,4,
			   -12,4,16}))
    check.eq(stats.cor(m,m),
	     matrix(3,3,{1,-1,-1,
			 -1,1,1,
			   -1,1,1}))
end)
