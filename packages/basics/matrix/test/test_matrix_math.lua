-- forces the use of CUDA
mathcore.set_use_cuda_default(util.is_cuda_available())
--
local check = utest.check
local T = utest.test
--

if not util.is_cuda_available() then
  T("MathOpTest",
    function()
      local a = matrix(2,4,3,{
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2,
                         0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4,
      })
      ca = a:select(2,2):complement()
      check.eq(ca,
               matrix(2,4,3,{
                        0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.9, 0.8,
                        0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
               }):select(2,2))
      
      
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

      local h = d:slice({2,2},{2,2},true)
      check(h, matrix.col_major(2,2,{
                                  5, 6,
                                  8, 9,
                               }),
            "matrix slice clone")

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
end

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
    --
    local m = matrix.col_major(1,4):linear()
    check.eq(m:slice({1,1},{1,2},true), matrix.col_major(1,2,{0,1}))
    check.eq(m:slice({1,3},{1,2},true), matrix.col_major(1,2,{2,3}))
    check.eq(m:slice({1,1},{1,2}), matrix.col_major(1,2,{0,1}))
    check.eq(m:slice({1,3},{1,2}), matrix.col_major(1,2,{2,3}))
    --
    local m = matrix(1,4):linear()
    check.eq(m:slice({1,1},{1,2},true), matrix(1,2,{0,1}))
    check.eq(m:slice({1,3},{1,2},true), matrix(1,2,{2,3}))
    check.eq(m:slice({1,1},{1,2}), matrix(1,2,{0,1}))
    check.eq(m:slice({1,3},{1,2}), matrix(1,2,{2,3}))
end)

local function load_csv()
  local def = 0.0/0.0
  local m = matrix.read( aprilio.stream.input_lua_string"1,2,3,,4\n5,6,7,8,\n4,,,,5",
                         { [matrix.options.delim]=",",
                           [matrix.options.empty]=true,
                           [matrix.options.default]=def,
                           [matrix.options.tab]=true } )
  return m
end

T("CSVTest", function()
    local m = load_csv()
    -- TODO: nan comparison returns always FALSE
    -- check.eq(m, matrix(3,5,{1,2,3,"-nan",4,5,6,7,8,"-nan",4,"-nan","-nan","-nan",5}))
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

T("SumTest", function()
    local m = matrix(2, 3, {1, 2, 3,
                            4, 5, 6})
    check.eq(m:sum(), 1+2+3+4+5+6)
    check.eq(m:sum(1), matrix(1, 3, {1+4, 2+5, 3+6}))
    check.eq(m:sum(2), matrix(2, 1, {1+2+3, 4+5+6}))

    local m = matrix(2, 4, 3, { 1,  2,  3,    4,  5,  6,    7,  8,  9,   10, 11, 12,
                               13, 14, 15,   16, 17, 18,   19, 20, 21,   22, 23, 24})
    check.eq(m:sum(), 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24)
    check.eq(m:sum(1), matrix(1, 4, 3, { 1+13,  2+14,  3+15,
                                         4+16,  5+17,  6+18,
                                         7+19,  8+20,  9+21,
                                        10+22, 11+23, 12+24}))
    check.eq(m:sum(2), matrix(2, 1, 3, {  1+4+7+10,   2+5+8+11,    3+6+9+12,
                                        13+16+19+22, 14+17+20+23, 15+18+21+24}))
    check.eq(m:sum(3), matrix(2, 4, 1, { 1+2+3,    4+5+6,    7+8+9,   10+11+12,
                                        13+14+15, 16+17+18, 19+20+21, 22+23+24}))
end)

T("MaxTest", function()
    local m = matrix(2, 3, {1, 4, 2,
                            6, 3, 5})
    check.eq(m:max(), 6)
    check.eq(m:max(1), matrix(1, 3, {6, 4, 5}))
    check.eq(m:max(2), matrix(2, 1, {4, 6}))

    local m = matrix(2, 4, 3, { 12, 14, 18,   8,  5, 6,    7, 16,  9,   10, 24, 1,
                                13,  2, 15,   4, 17, 3,   19, 20, 21,   22, 23, 11})
    check.eq(m:max(), 24)
    local a,b = m:max(1)
    check.eq(a, matrix(1, 4, 3, { 13,  14, 18,
                                  8,   17,  6,
                                  19,  20, 21,
                                  22,  24, 11 }))
    check.eq(b:to_float(), matrix(1, 4, 3, { 2, 1, 1,
                                             1, 2, 1,
                                             2, 2, 2,
                                             2, 1, 2 }))
    local a,b = m:max(2)
    check.eq(a, matrix(2, 1, 3, { 12, 24, 18,
                                  22, 23, 21 }))
    check.eq(b:to_float(), matrix(2, 1, 3, { 1, 4, 1,
                                             4, 4, 3 }))
    local a,b = m:max(3)
    check.eq(a, matrix(2, 4, 1, { 18, 8, 16, 24,
                                  15, 17, 21, 23 }))
    check.eq(b:to_float(), matrix(2, 4, 1, { 3, 1, 2, 2,
                                             3, 2, 3, 2 }))
end)

T("MinTest", function()
    local m = matrix(2, 3, {1, 4, 2,
                            6, 3, 5})
    check.eq(m:min(), 1)
    check.eq(m:min(1), matrix(1, 3, {1, 3, 2}))
    check.eq(m:min(2), matrix(2, 1, {1, 3}))

    local m = matrix(2, 4, 3, { 12, 14, 18,   8,  5, 6,    7, 16,  9,   10, 24, 1,
                                13,  2, 15,   4, 17, 3,   19, 20, 21,   22, 23, 11})
    check.eq(m:min(), 1)
    local a,b = m:min(1)
    check.eq(a, matrix(1, 4, 3, { 12,   2, 15,
                                  4,    5,  3,
                                  7,   16,  9,
                                  10,  23,  1 }))
    check.eq(b:to_float(), matrix(1, 4, 3, { 1, 2, 2,
                                             2, 1, 2,
                                             1, 1, 1,
                                             1, 2, 1 }))
    local a,b = m:min(2)
    check.eq(a, matrix(2, 1, 3, {  7,  5,  1,
                                   4,  2,  3 }))
    check.eq(b:to_float(), matrix(2, 1, 3, { 3, 2, 4,
                                             2, 1, 2 }))
    local a,b = m:min(3)
    check.eq(a, matrix(2, 4, 1, { 12, 5,  7,  1,
                                   2,  3, 19, 11 }))
    check.eq(b:to_float(), matrix(2, 4, 1, { 1, 2, 1, 3,
                                             2, 3, 1, 3 }))
end)

T("LargeMatrices", function()
    local m1 = matrix(300,200,100)
    local m2 = matrix(300,200,100)
    for i=1,m1:dim(3) do m1:select(3,i):fill(i) end
    for i=1,m2:dim(3) do m2(':',':',i):fill(i) end
    check.eq(m1, m2, "select and slice")
end)
