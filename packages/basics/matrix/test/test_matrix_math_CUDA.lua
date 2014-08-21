local check = utest.check
local T = utest.test
local M = matrix.col_major
mathcore.set_use_cuda_default(util.is_cuda_available())
--

T("MathOpTest",
  function()
    local a = matrix.fromString[[
1 3
ascii col_major
1 2 3
]]

    local b = matrix.fromString[[
3 1
ascii col_major
1
4
7
]]

    local c = a*b
    check.eq(c:get(1,1), 1*1+2*4+3*7, "dot product")
    
    check.eq(b*a, M(3,3,{
                      1,  2,  3,
                      4,  8, 12,
                      7, 14, 21,
                   }),
             "cross product")

    local d = matrix.fromString[[
3 3
ascii col_major
1 2 3
4 5 6
7 8 9
]]

    check.eq(d:clone("col_major"), M(3,3,{
                                       1, 2, 3,
                                       4, 5, 6,
                                       7, 8, 9,
                                    }),
             "col_major clone")
    
    check(d:transpose(), M(3,3,{
                             1, 4, 7,
                             2, 5, 8,
                             3, 6, 9,
                          }),
          "transpose")
    
    local e = d * d 
    check(e, M(3,3,{
                 30,   36,  42,
                 66,   81,  96,
                 102, 126, 150,
              }),
          "matrix mult *")

    local d = d:clone("col_major")
    local e = d * d
    check(e, M(3,3,{
                 30,   36,  42,
                 66,   81,  96,
                 102, 126, 150,
              }),
          "matrix mult * in col_major")

    local h = d:slice({2,2},{2,2})
    check(h, M(2,2,{
                 5, 6,
                 8, 9,
              }),
          "matrix slice")

    local e = h * h
    check(e, M(2,2,{
                 73,  84,
                 112, 129,
              }),
          "matrix slice mul *")

    local l = M(2,2):fill(4) + h
    check(l, M(2,2,{
                 9, 10,
                 12, 13,
              }),
          "matrix fill and slice add +")

    local g = M(3,2,{1,2,
                     3,4,
                     5,6})
    check(g:transpose():clone("col_major"), M(2,3,{
                                                1, 3, 5,
                                                2, 4, 6,
                                             }),
          "transpose + col_major clone")
    
    check(g:transpose():clone("col_major"), M(2,3,{
                                                1, 3, 5,
                                                2, 4, 6,
                                             }),
          "transpose + col_major clone + row_major clone")

    check(g:transpose(), M(2,3,{
                             1, 3, 5,
                             2, 4, 6,
                          }),
          "transpose")

    local j = g:transpose() * g
    check(j, M(2,2,{
                 35, 44,
                 44, 56,
              }),
          "transpose mul *")

    local j = M(2,2):gemm{
      trans_A=true, trans_B=false,
      alpha=1.0, A=g, B=g,
      beta=0.0
                         }
    check(j, M(2,2,{
                 35, 44,
                 44, 56,
              }),
          "gemm in row_major")
    
    local j = M(2,2):gemm{
      trans_A=true, trans_B=false,
      alpha=1.0, A=g:clone("col_major"), B=g:clone("col_major"),
      beta=0.0
                         }
    check(j, M(2,2,{
                 35, 44,
                 44, 56,
              }),
          "gemm in col_major")
end)

---------------------------------------------------------------
---------------------------------------------------------------
---------------------------------------------------------------

T("SVDTest",
  function()
    local m = M(4,5,{1,0,0,0,2,
                     0,0,3,0,0,
                     0,0,0,0,0,
                     0,4,0,0,0})
    local U,S,V = m:svd()
    check(U, M(4,4,
               {
                 0,0,1, 0,
                 0,1,0, 0,
                 0,0,0,-1,
                 1,0,0, 0,
    }))
    check(S:to_dense(), M(4,{4,3,2.23607,0}):diagonalize())
    check(V, M(5,5,
               {
                 0,1,0,0,0,
                 0,0,1,0,0,
                 0.447214,0,0,0,0.894427,
                 0,0,0,1,0,
                   -0.894427,0,0,0,0.447214,
    }))
end)

---------------------------------------------------------------
---------------------------------------------------------------
---------------------------------------------------------------

T("SliceTest",
  function()
    local m = M(20,20):uniformf()
    local subm = m:slice({5,4},{6,9})
    check.eq(subm, m("5:10","4:12"))
    check.eq(subm, m({5,10},{4,12}))
    local subm = m:slice({1,4},{6,m:dim(2)-3})
    check.eq(subm, m(":6","4:"))
end)
