m = matrix(2,2, {
	     -2, 1,
	     -1, 1,
		})
b = matrix(2,1, {
	     4,
	     1
		})
c = m:inv()*b

assert(c:get(1,1) == -3)
assert(c:get(2,1) == -2)
