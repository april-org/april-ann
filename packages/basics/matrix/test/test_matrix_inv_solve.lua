local check = utest.check
local T = utest.test
-- inverse
local M = matrix.col_major
local m = M(2,2, {
		-2, 1,
		-1, 1,
		 })
local b = M(2,1, {
	      4,
	      1
		 })
local c = m:inv()*b
T("InverseTest", function() check.eq(c, M(2,1,{-3,-2})) end)

-- pseudo-inverse
local c = m:pinv()*b
T("PseudoInverseTest", function() check.eq(c, M(2,1,{-3,-2})) end)
