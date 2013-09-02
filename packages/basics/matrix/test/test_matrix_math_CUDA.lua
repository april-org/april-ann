a = matrix.fromString[[
1 3
ascii col_major
1 2 3
]]

b = matrix.fromString[[
3 1
ascii col_major
1
4
7
]]

if util.is_cuda_available() then a:set_use_cuda(true) b:set_use_cuda(true) end

c = a*b

assert(c:get(1) == 1*1+2*4+3*7)

assert((b*a):equals(matrix.col_major(3,3,{
				       1,  2,  3,
				       4,  8, 12,
				       7, 14, 21,
					 })))

d = matrix.fromString[[
3 3
ascii col_major
1 2 3
4 5 6
7 8 9
]]
if util.is_cuda_available() then d:set_use_cuda(true) end

assert(d:clone():equals(matrix.col_major(3,3,{
					    1, 2, 3,
					    4, 5, 6,
					    7, 8, 9,
					 })))
assert(d:transpose():equals(matrix.col_major(3,3,{
						1, 4, 7,
						2, 5, 8,
						3, 6, 9,
					     })))
e = d * d 
assert(e:equals(matrix.col_major(3,3,{
				    30,   36,  42,
				    66,   81,  96,
				    102, 126, 150,
				 })))

d = d:clone()
e = d * d
assert(e:equals(matrix.col_major(3,3,{
				   30,   36,  42,
				   66,   81,  96,
				   102, 126, 150,
				     })))

h = d:slice({2,2},{2,2})
assert(h:equals(matrix.col_major(2,2,{
				   5, 6,
				   8, 9,
				     })))

e = h * h
assert(e:equals(matrix.col_major(2,2,{
				    73,  84,
				   112, 129,
				     })))

l = matrix.col_major(2,2):fill(4) + h
assert(l:equals(matrix.col_major(2,2,{
				    9, 10,
				   12, 13,
				     })))

g = matrix.col_major(3,2,{1,2,
			  3,4,
			  5,6})
if util.is_cuda_available() then g:set_use_cuda(true) end
assert(g:transpose():equals(matrix.col_major(2,3,{
						1, 3, 5,
						2, 4, 6,
					     })))
j = g:transpose() * g
assert(j:equals(matrix.col_major(2,2,{
				    35, 44,
				    44, 56,
				 })))

j = matrix.col_major(2,2):gemm{
  trans_A=true, trans_B=false,
  alpha=1.0, A=g, B=g,
  beta=0.0
}
assert(j:equals(matrix.col_major(2,2,{
				    35, 44,
				    44, 56,
				 })))

j = matrix.col_major(2,2):gemm{
  trans_A=true, trans_B=false,
  alpha=1.0, A=g:clone(), B=g:clone(),
  beta=0.0
}
assert(j:equals(matrix.col_major(2,2,{
				   35, 44,
				   44, 56,
				     })))

-- LAPACK

m = matrix.col_major(2,2, {
			-2, 1,
			-1, 1,
		  })
b = matrix.col_major(2,1, {
			4,
			1
		     })
c = m:inv()*b

assert(c:get(1,1) == -3)
assert(c:get(2,1) == -2)
