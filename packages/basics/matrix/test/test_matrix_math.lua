local check = utest.check
local T = utest.test
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

    check.eq(b*a, matrix.col_major(3,3,{
                                     1,  2,  3,
                                     4,  8, 12,
                                     7, 14, 21,
    }),
    "cross product")

    local d = matrix.fromString[[
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
    }),
    "col_major clone")
    
    check(d:transpose(), matrix(3,3,{
                                  1, 4, 7,
                                  2, 5, 8,
                                  3, 6, 9,
    }),
    "transpose")
    
    local e = d * d 
    check(e, matrix(3,3,{
                      30,   36,  42,
                      66,   81,  96,
                      102, 126, 150,
    }),
    "matrix mult *")

    local d = d:clone("col_major")
    local e = d * d
    check(e, matrix.col_major(3,3,{
                                30,   36,  42,
                                66,   81,  96,
                                102, 126, 150,
    }),
    "matrix mult * in col_major")

    local h = d:slice({2,2},{2,2})
    check(h, matrix.col_major(2,2,{
                                5, 6,
                                8, 9,
    }),
    "matrix slice")

    local e = h * h
    check(e, matrix.col_major(2,2,{
                                73,  84,
                                112, 129,
    }),
    "matrix slice mul *")

    local l = matrix.col_major(2,2):fill(4) + h
    check(l, matrix.col_major(2,2,{
                                9, 10,
                                12, 13,
    }),
    "matrix fill and slice add +")

    local g = matrix(3,2,{1,2,
                          3,4,
                          5,6})
    check(g:transpose():clone("col_major"), matrix.col_major(2,3,{
                                                               1, 3, 5,
                                                               2, 4, 6,
    }),
    "transpose + col_major clone")
    
    check(g:transpose():clone("col_major"):clone("row_major"), matrix(2,3,{
                                                                        1, 3, 5,
                                                                        2, 4, 6,
    }),
    "transpose + col_major clone + row_major clone")
    
    check(g:transpose(), matrix(2,3,{
                                  1, 3, 5,
                                  2, 4, 6,
    }),
    "transpose")
    
    local j = g:transpose() * g
    check(j, matrix(2,2,{
                      35, 44,
                      44, 56,
                   }),
          "transpose mul *")

    local j = matrix(2,2):gemm{
      trans_A=true, trans_B=false,
      alpha=1.0, A=g, B=g,
      beta=0.0
                              }
    check(j, matrix(2,2,{
                      35, 44,
                      44, 56,
                   }),
          "gemm in row_major")
    
    local j = matrix.col_major(2,2):gemm{
      trans_A=true, trans_B=false,
      alpha=1.0, A=g:clone("col_major"), B=g:clone("col_major"),
      beta=0.0
                                        }
    check(j, matrix.col_major(2,2,{
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
    local m = matrix.col_major(4,5,{1,0,0,0,2,
                                    0,0,3,0,0,
                                    0,0,0,0,0,
                                    0,4,0,0,0})
    local U,S,V = m:svd()
    check(U, matrix.col_major(4,4,
                              {
                                0,0,1, 0,
                                0,1,0, 0,
                                0,0,0,-1,
                                1,0,0, 0,
    }))
    check(S:to_dense(), matrix.col_major(4,{4,3,2.23607,0}):diagonalize())
    check(V, matrix.col_major(5,5,
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
    local m = matrix(20,20):uniformf()
    local subm = m:slice({5,4},{6,9})
    check.eq(subm, m("5:10","4:12"))
    check.eq(subm, m({5,10},{4,12}))
    local subm = m:slice({1,4},{6,m:dim(2)-3})
    check.eq(subm, m(":6","4:"))
end)

local function load_csv()
  local def = 0.0/0.0
  local m = matrix.readTab( aprilio.stream.input_lua_string[[1,2,3,,4\n5,6,7,8,\n4,,,,5]],
                            nil,     -- order
                            ",",     -- delim
                            true,    -- keep_delim
                            def)     -- default value
end

T("CSVTest", function()
    local m = load_csv()
    check.eq(m, matrix(3,5,{1,2,3,def,4,5,6,7,8,def,4,def,def,def,5}))
end)

T("EQandNEQTest", function()
    local m   = load_csv()
    local def = 0.0/0.0
    check.eq(m:clone():eq(def),
             matrix(3,5,{0,0,0,1,0,0,0,0,0,1,0,1,1,1,0}))
    check.eq(m:clone():neq(def),
             matrix(3,5,{1,1,1,0,1,1,1,1,1,0,1,0,0,0,1}))
    
    check.eq(m:clone():eq(4),
             matrix(3,5,{0,0,0,0,1,0,0,0,0,0,1,0,0,0,0}))
    check.eq(m:clone():neq(4),
             matrix(3,5,{1,1,1,1,0,1,1,1,1,1,0,1,1,1,1}))

    check.eq(m:clone():eq(m), matrix(3,5):ones())
    check.eq(m:clone():neq(m), matrix(3,5):zeros())
end)
