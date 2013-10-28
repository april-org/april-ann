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

c = a*b

assert(c:get(1) == 1*1+2*4+3*7)

assert((b*a):equals(matrix.col_major(3,3,{
				       1,  2,  3,
				       4,  8, 12,
				       7, 14, 21,
					 })))

d = matrix.fromString[[
3 3
ascii
1 2 3
4 5 6
7 8 9
]]

assert(d:clone("col_major"):equals(matrix.col_major(3,3,{
						      1, 2, 3,
						      4, 5, 6,
						      7, 8, 9,
							})))
assert(d:transpose():equals(matrix(3,3,{
				     1, 4, 7,
				     2, 5, 8,
				     3, 6, 9,
				       })))
e = d * d 
assert(e:equals(matrix(3,3,{
			 30,   36,  42,
			 66,   81,  96,
			 102, 126, 150,
			   })))

d = d:clone("col_major")
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

g = matrix(3,2,{1,2,
		3,4,
		5,6})
assert(g:transpose():clone("col_major"):equals(matrix.col_major(2,3,{
								  1, 3, 5,
								  2, 4, 6,
								    })))
assert(g:transpose():clone("col_major"):clone("row_major"):equals(matrix(2,3,{
									   1, 3, 5,
									   2, 4, 6,
									     })))
assert(g:transpose():equals(matrix(2,3,{
				     1, 3, 5,
				     2, 4, 6,
				       })))
j = g:transpose() * g
assert(j:equals(matrix(2,2,{
			 35, 44,
			 44, 56,
			   })))

j = matrix(2,2):gemm{
  trans_A=true, trans_B=false,
  alpha=1.0, A=g, B=g,
  beta=0.0
}
assert(j:equals(matrix(2,2,{
			 35, 44,
			 44, 56,
			   })))

j = matrix.col_major(2,2):gemm{
  trans_A=true, trans_B=false,
  alpha=1.0, A=g:clone("col_major"), B=g:clone("col_major"),
  beta=0.0
}
assert(j:equals(matrix.col_major(2,2,{
				   35, 44,
				   44, 56,
				     })))

---------------------------------------------------------------
---------------------------------------------------------------
---------------------------------------------------------------

local m = matrix.col_major(4,5,{1,0,0,0,2,
				0,0,3,0,0,
				0,0,0,0,0,
				0,4,0,0,0})
local U,S,V = m:svd()
assert(U:equals(matrix.col_major(4,4,
				 {
				   0,0,1, 0,
				   0,1,0, 0,
				   0,0,0,-1,
				   1,0,0, 0,
				 })))
assert(S:equals(matrix.col_major(4,{4,3,2.23607,0})))
assert(V:equals(matrix.col_major(5,5,
				 {
				   0,1,0,0,0,
				   0,0,1,0,0,
				   0.447214,0,0,0,0.894427,
				   0,0,0,1,0,
				     -0.894427,0,0,0,0.447214,
				 })))

---------------------------------------------------------------
---------------------------------------------------------------
---------------------------------------------------------------

local m = matrix(20,20):uniformf()
local subm = m:slice({5,4},{6,9})
subm:equals( m("5:10","4:12") )
subm:equals( m({5,10},{4,12}) )
local subm = m:slice({1,4},{6,m:dim(2)-3})
subm:equals( m(":6","4:") )
