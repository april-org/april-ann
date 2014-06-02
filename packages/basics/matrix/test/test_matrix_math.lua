local check = utest.check
--
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
check.eq(c:get(1,1), 1*1+2*4+3*7)

check.eq(b*a, matrix.col_major(3,3,{
                                 1,  2,  3,
                                 4,  8, 12,
                                 7, 14, 21,
                                   }))

d = matrix.fromString[[
3 3
ascii
1 2 3
4 5 6
7 8 9
]]

check.eq(d:clone("col_major"), matrix.col_major(3,3,{
                                                  1, 2, 3,
                                                  4, 5, 6,
                                                  7, 8, 9,
                                                    }))
check.eq(d:transpose(), matrix(3,3,{
                                 1, 4, 7,
                                 2, 5, 8,
                                 3, 6, 9,
                                   }))
e = d * d 
check.eq(e, matrix(3,3,{
                     30,   36,  42,
                     66,   81,  96,
                     102, 126, 150,
                       }))

d = d:clone("col_major")
e = d * d
check.eq(e, matrix.col_major(3,3,{
                               30,   36,  42,
                               66,   81,  96,
                               102, 126, 150,
                                 }))

h = d:slice({2,2},{2,2})
check.eq(h, matrix.col_major(2,2,{
                               5, 6,
                               8, 9,
                                 }))

e = h * h
check.eq(e, matrix.col_major(2,2,{
                               73,  84,
                               112, 129,
                                 }))

l = matrix.col_major(2,2):fill(4) + h
check.eq(l, matrix.col_major(2,2,{
                               9, 10,
                               12, 13,
                                 }))

g = matrix(3,2,{1,2,
		3,4,
		5,6})
check.eq(g:transpose():clone("col_major"), matrix.col_major(2,3,{
                                                              1, 3, 5,
                                                              2, 4, 6,
                                                                }))
check.eq(g:transpose():clone("col_major"):clone("row_major"), matrix(2,3,{
                                                                       1, 3, 5,
                                                                       2, 4, 6,
                                                                         }))
check.eq(g:transpose(), matrix(2,3,{
                                 1, 3, 5,
                                 2, 4, 6,
                                   }))
j = g:transpose() * g
check.eq(j, matrix(2,2,{
                     35, 44,
                     44, 56,
                       }))

j = matrix(2,2):gemm{
  trans_A=true, trans_B=false,
  alpha=1.0, A=g, B=g,
  beta=0.0
                    }
check.eq(j, matrix(2,2,{
                     35, 44,
                     44, 56,
                       }))

j = matrix.col_major(2,2):gemm{
  trans_A=true, trans_B=false,
  alpha=1.0, A=g:clone("col_major"), B=g:clone("col_major"),
  beta=0.0
                              }
check.eq(j, matrix.col_major(2,2,{
                               35, 44,
                               44, 56,
                                 }))

---------------------------------------------------------------
---------------------------------------------------------------
---------------------------------------------------------------

local m = matrix.col_major(4,5,{1,0,0,0,2,
				0,0,3,0,0,
				0,0,0,0,0,
				0,4,0,0,0})
local U,S,V = m:svd()
check.eq(U, matrix.col_major(4,4,
                             {
                               0,0,1, 0,
                               0,1,0, 0,
                               0,0,0,-1,
                               1,0,0, 0,
                             }))
check.eq(S:to_dense(), matrix.col_major(4,{4,3,2.23607,0}):diagonalize())
check.eq(V, matrix.col_major(5,5,
                             {
                               0,1,0,0,0,
                               0,0,1,0,0,
                               0.447214,0,0,0,0.894427,
                               0,0,0,1,0,
                                 -0.894427,0,0,0,0.447214,
                             }))

---------------------------------------------------------------
---------------------------------------------------------------
---------------------------------------------------------------

local m = matrix(20,20):uniformf()
local subm = m:slice({5,4},{6,9})
check.eq(subm, m("5:10","4:12"))
check.eq(subm, m({5,10},{4,12}))
local subm = m:slice({1,4},{6,m:dim(2)-3})
check.eq(subm, m(":6","4:"))

---------------------------------------------------------------
---------------------------------------------------------------
---------------------------------------------------------------

local m = matrix(3,3,{ 1, 2, 3,
                       4, 5, 6,
                       1,-1, 3, })
check.lt(math.abs(m:det()+18.0), 1e-03)

local m = matrix(3,3,{ 1, 2, 3,
                       4, 5, 6,
                       1, 5, 3, })
local logdet,sign = m:logdet()
check.lt(math.abs(logdet-math.log(m:det())), 1e-03)
check.eq(sign,1)

---------------------------------------------------------------

local m = matrix.col_major(3,3,{ 4, 12, -16,
                                 12, 37, -43,
                                   -16, -43, 98 })
local U = m:cholesky('U')
local L = m:cholesky('L')

check.eq( U:transpose()*U, m)
check.eq( L*L:transpose(), m)
