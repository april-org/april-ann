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
assert(c:equals(M(2,1,{-3,-2})))

-- pseudo-inverse
local c = m:pinv()*b
assert(c:equals(M(2,1,{-3,-2})))
