-- TODO: implement unit tests for index, indexCopy, indexFill, operator [{}]

local check = utest.check
local T = utest.test
--

function do_test()

  T("MathOpTest",
    function()
      local m = matrix(3,4,5):linspace()
      local m2 = m:t(1,2)
      local a = m2:select(1,1)
      local b = matrix(5,2):linspace()
      local c = a*b
      check.eq(c, matrix(3,2,{ 95,   110,
                               595,  710,
                               1095, 1310, }))

      local a = matrix(2,4,3,{
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2,
                         0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4,
      })
      ca = a:select(2,2):complement()
      check.eq(ca,
               matrix(2,4,3,{
                        0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.9, 0.8,
                        0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
               }):select(2,2),
               "select")
      
      
      local a = matrix.fromString[[
1 3
ascii
1 2 3
]]

      local b = matrix.fromString[[
3 1
ascii
1
4
7
]]

      local c = a*b
      check.eq(c:get(1,1), 1*1+2*4+3*7, "dot product")
      
      check.eq(b*a, matrix(3,3,{
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

      check.eq(d:clone(), matrix(3,3,{
                                   1, 2, 3,
                                   4, 5, 6,
                                   7, 8, 9,
                                }),
               "clone")
      
      check(d:transpose(), matrix(3,3,{
                                    1, 4, 7,
                                    2, 5, 8,
                                    3, 6, 9,
                                 }),
            "transpose")

      check(d:transpose():clone(), matrix(3,3,{
                                            1, 4, 7,
                                            2, 5, 8,
                                            3, 6, 9,
                                         }),
            "transpose clone")
      
      local e = d * d 
      check(e, matrix(3,3,{
                        30,   36,  42,
                        66,   81,  96,
                        102, 126, 150,
                     }),
            "matrix mult *")

      local h = d:slice({2,2},{2,2})
      check(h, matrix(2,2,{
                        5, 6,
                        8, 9,
                     }),
            "matrix slice")

      local h = d:slice({2,2},{2,2},true)
      check(h, matrix(2,2,{
                        5, 6,
                        8, 9,
                     }),
            "matrix slice clone")

      local e = h * h
      check(e, matrix(2,2,{
                        73,  84,
                        112, 129,
                     }),
            "matrix slice mul *")

      local l = matrix(2,2):fill(4) + h
      check(l, matrix(2,2,{
                        9, 10,
                        12, 13,
                     }),
            "matrix fill and slice add +")

      local g = matrix(3,2,{1,2,
                            3,4,
                            5,6})
      check(g:transpose():clone(), matrix(2,3,{
                                            1, 3, 5,
                                            2, 4, 6,
                                         }),
            "transpose + clone")
      
      check(g:transpose():clone():transpose():clone(), matrix(2,3,{
                                                                1, 3, 5,
                                                                2, 4, 6,
                                                             }),
            "transpose + clone + transpose + clone")
      
      check(g:transpose(), matrix(2,3,{
                                    1, 3, 5,
                                    2, 4, 6,
                                 }),
            "transpose")

      check(g:transpose():clone(), matrix(2,3,{
                                            1, 3, 5,
                                            2, 4, 6,
                                         }),
            "transpose clone")

      local h = matrix(table.unpack(g:transpose():dim()))
      
      check(h:copy(g:transpose()), matrix(2,3,{
                                            1, 3, 5,
                                            2, 4, 6,
                                         }),
            "copy transposed")
      
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
            "gemm")
      
      local j = matrix(2,2):transpose():gemm{
        trans_A=true, trans_B=false,
        alpha=1.0, A=g, B=g,
        beta=0.0
                                            }
      check(j, matrix(2,2,{
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
      local m = matrix(4,5,{1,0,0,0,2,
                            0,0,3,0,0,
                            0,0,0,0,0,
                            0,4,0,0,0})
      local U,S,V = m:svd()
      check(U, matrix(4,4,
                      {
                        0,0,1, 0,
                        0,1,0, 0,
                        0,0,0,-1,
                        1,0,0, 0,
                     }),
            "SVD U matrix")
      check(S:to_dense(), matrix(4,{4,3,2.23607,0}):diagonalize(),
            "SVD S matrix")
      check(V, matrix(5,5,
                      {
                        0,1,0,0,0,
                        0,0,1,0,0,
                        0.447214,0,0,0,0.894427,
                        0,0,0,1,0,
                          -0.894427,0,0,0,0.447214,
                     }),
            "SVD V matrix")
  end)

  ---------------------------------------------------------------
  ---------------------------------------------------------------
  ---------------------------------------------------------------

  T("SliceTest",
    function()
      local m = matrix(20,20):uniformf()
      local subm = m:slice({5,4},{6,9})
      check.eq(subm, m("5:10","4:12"), "slice() 1")
      check.eq(subm, m({5,10},{4,12}), "slice() 2")
      local subm = m:slice({1,4},{6,m:dim(2)-3})
      check.eq(subm, m(":6","4:"), "slice() 3")
      --
      local m = matrix(1,4):transpose():linear()
      check.eq(m:slice({1,1},{2,1},true), matrix(2,1,{0,1}), "slice 4")
      check.eq(m:slice({3,1},{2,1},true), matrix(2,1,{2,3}), "slice 5")
      check.eq(m:slice({1,1},{2,1}), matrix(2,1,{0,1}), "slice 6")
      check.eq(m:slice({3,1},{2,1}), matrix(2,1,{2,3}), "slice 7")
      --
      local m = matrix(1,4):linear()
      check.eq(m:slice({1,1},{1,2},true), matrix(1,2,{0,1}), "slice 8")
      check.eq(m:slice({1,3},{1,2},true), matrix(1,2,{2,3}), "slice 9")
      check.eq(m:slice({1,1},{1,2}), matrix(1,2,{0,1}), "slice 10")
      check.eq(m:slice({1,3},{1,2}), matrix(1,2,{2,3}), "slice 11")
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

  local tmpname = os.tmpname()
  T("SerializationTest", function()
      local m = matrix(10,20):uniformf()
      m:toFilename(tmpname)
      check.eq(matrix.fromFilename(tmpname), m, "toFilename/fromFilename")
      m:toFilename(tmpname, "ascii")
      check.eq(matrix.fromFilename(tmpname), m, "toFilename/fromFilename ascii")
      m:toFilename(tmpname, "binary")
      check.eq(matrix.fromFilename(tmpname), m, "toFilename/fromFilename binary")
      m:toTabFilename(tmpname)
      check.eq(matrix.fromTabFilename(tmpname), m, "toTabFilename/fromTabFilename")
      check.eq(matrix(10,20,m:toTable()), m)
  end)
  os.remove(tmpname)

  T("EQandNEQTest", function()
      local m   = load_csv()
      local def = 0.0/0.0
      check.eq(m:eq(def),
               matrixBool(matrix(3,5,{0,0,0,1,0,0,0,0,0,1,0,1,1,1,0})), "NAN eq")
      check.eq(m:neq(def),
               matrixBool(matrix(3,5,{1,1,1,0,1,1,1,1,1,0,1,0,0,0,1})), "NAN neq")
      
      check.eq(m:eq(4),
               matrixBool(matrix(3,5,{0,0,0,0,1,0,0,0,0,0,1,0,0,0,0})), "4 eq")
      check.eq(m:neq(4),
               matrixBool(matrix(3,5,{1,1,1,1,0,1,1,1,1,1,0,1,1,1,1})), "4 neq")

      check.eq(m:eq(m), matrixBool(3,5):ones(), "eq m")
      check.eq(m:neq(m), matrixBool(3,5):zeros(), "neq m")
  end)

  T("SumTest", function()
      local m = matrix(2, 3, {1, 2, 3,
                              4, 5, 6})
      check.eq(m:sum(), 1+2+3+4+5+6, "sum()")
      check.eq(m:sum(1), matrix(1, 3, {1+4, 2+5, 3+6}), "sum(1)")
      check.eq(m:sum(2), matrix(2, 1, {1+2+3, 4+5+6}), "sum(2)")

      local m = matrix(2, 4, 3, { 1,  2,  3,    4,  5,  6,    7,  8,  9,   10, 11, 12,
                                  13, 14, 15,   16, 17, 18,   19, 20, 21,   22, 23, 24})
      check.eq(m:sum(), 1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24,
               "sum() 2")
      check.eq(m:sum(1), matrix(1, 4, 3, { 1+13,  2+14,  3+15,
                                           4+16,  5+17,  6+18,
                                           7+19,  8+20,  9+21,
                                           10+22, 11+23, 12+24}),
               "sum(1) 2")
      check.eq(m:sum(2), matrix(2, 1, 3, {  1+4+7+10,   2+5+8+11,    3+6+9+12,
                                            13+16+19+22, 14+17+20+23, 15+18+21+24}),
               "sum(2) 2")
      check.eq(m:sum(3), matrix(2, 4, 1, { 1+2+3,    4+5+6,    7+8+9,   10+11+12,
                                           13+14+15, 16+17+18, 19+20+21, 22+23+24}),
               "sum(3) 2")
  end)

  T("MaxTest", function()
      local m = matrix(2, 3, {1, 4, 2,
                              6, 3, 5})
      check.eq(m:max(), 6, "max()")
      check.eq(m:max(1), matrix(1, 3, {6, 4, 5}), "max(1)")
      check.eq(m:max(2), matrix(2, 1, {4, 6}), "max(2)")

      local m = m - 10
      check.eq(m:max(), -4, "max()")
      check.eq(m:max(1), matrix(1, 3, {-4, -6, -5}), "max(1)")
      check.eq(m:max(2), matrix(2, 1, {-6, -4}), "max(2)")

      local m = matrix(2, 4, 3, { 12, 14, 18,   8,  5, 6,    7, 16,  9,   10, 24, 1,
                                  13,  2, 15,   4, 17, 3,   19, 20, 21,   22, 23, 11})
      check.eq(m:max(), 24, "max() 2")
      local a,b = m:max(1)
      check.eq(a, matrix(1, 4, 3, { 13,  14, 18,
                                    8,   17,  6,
                                    19,  20, 21,
                                    22,  24, 11 }), "max(1) a")
      check.eq(b:to_float(), matrix(1, 4, 3, { 2, 1, 1,
                                               1, 2, 1,
                                               2, 2, 2,
                                               2, 1, 2 }), "max(1) b")
      local a,b = m:max(2)
      check.eq(a, matrix(2, 1, 3, { 12, 24, 18,
                                    22, 23, 21 }), "max(2) a")
      check.eq(b:to_float(), matrix(2, 1, 3, { 1, 4, 1,
                                               4, 4, 3 }), "max(2) b")
      local a,b = m:max(3)
      check.eq(a, matrix(2, 4, 1, { 18, 8, 16, 24,
                                    15, 17, 21, 23 }), "max(3) a")
      check.eq(b:to_float(), matrix(2, 4, 1, { 3, 1, 2, 2,
                                               3, 2, 3, 2 }), "max(3) b")
  end)

  T("MinTest", function()
      local m = matrix(2, 3, {1, 4, 2,
                              6, 3, 5})
      check.eq(m:min(), 1, "min()")
      check.eq(m:min(1), matrix(1, 3, {1, 3, 2}), "min(1)")
      check.eq(m:min(2), matrix(2, 1, {1, 3}), "min(2)")

      local m = m - 10
      check.eq(m:min(), -9, "min()")
      check.eq(m:min(1), matrix(1, 3, {-9, -7, -8}), "min(1)")
      check.eq(m:min(2), matrix(2, 1, {-9, -7}), "min(2)")

      local m = matrix(2, 4, 3, { 12, 14, 18,   8,  5, 6,    7, 16,  9,   10, 24, 1,
                                  13,  2, 15,   4, 17, 3,   19, 20, 21,   22, 23, 11})
      check.eq(m:min(), 1, "min() 2")
      local a,b = m:min(1)
      check.eq(a, matrix(1, 4, 3, { 12,   2, 15,
                                    4,    5,  3,
                                    7,   16,  9,
                                    10,  23,  1 }), "min(1) a")
      check.eq(b:to_float(), matrix(1, 4, 3, { 1, 2, 2,
                                               2, 1, 2,
                                               1, 1, 1,
                                               1, 2, 1 }), "min(1) b")
      local a,b = m:min(2)
      check.eq(a, matrix(2, 1, 3, {  7,  5,  1,
                                     4,  2,  3 }), "min(2) a")
      check.eq(b:to_float(), matrix(2, 1, 3, { 3, 2, 4,
                                               2, 1, 2 }), "min(2) b")
      local a,b = m:min(3)
      check.eq(a, matrix(2, 4, 1, { 12, 5,  7,  1,
                                    2,  3, 19, 11 }), "min(3) a")
      check.eq(b:to_float(), matrix(2, 4, 1, { 1, 2, 1, 3,
                                               2, 3, 1, 3 }), "min(3) b")
  end)

  T("BrodacastTest", function()
      local a = matrix(4,1,{0,10,20,30})
      local b = matrix(3,{0,1,2})
      local c = matrix.ext.broadcast(a,b,function(a,b) return a+b end)
      local d = matrix.ext.broadcast(a,b,bind(a.axpy, nil, 1))
      local e = matrix(4,3,{0,1,2,
                            10,11,12,
                            20,21,22,
                            30,31,32})
      check.eq(c,e)
      check.eq(d,e)
      --
      local x   = matrix(4):linear()
      local xx  = x:rewrap(4,1)
      local y   = matrix(5):ones()
      local z   = matrix(3,4):ones()
      local add = bind(a.axpy, nil, 1.0)
      check.errored(function()
          matrix.ext.broadcast(x, y, add)
      end)
      check.eq(matrix.ext.broadcast(xx, y, add),
               matrix(4,5,{1,1,1,1,1,
                           2,2,2,2,2,
                           3,3,3,3,3,
                           4,4,4,4,4,}))
      check.eq(matrix.ext.broadcast(x, z, add),
               matrix(3,4,{1,2,3,4,
                           1,2,3,4,
                           1,2,3,4}))
  end)

  T("LargeMatrices", function()
      local m1 = matrix(300,200,100)
      local m2 = matrix(300,200,100)
      for i=1,m1:dim(3) do m1:select(3,i):fill(i) end
      for i=1,m2:dim(3) do m2(':',':',i):fill(i) end
      check.eq(m1, m2, "select and slice")
      m1,m2=nil,nil
      collectgarbage("collect")
      --
      local m1 = matrix(300,200,100):fill(200)
      local eq = true
      for i=1,m1:size() do eq = eq and m1:raw_get(i-1) == 200 end
      check.TRUE(eq, "fill")
      --
      m1:scalar_add(10)
      local eq = true
      for i=1,m1:size() do eq = eq and m1:raw_get(i-1) == 210 end
      check.TRUE(eq, "scalar_add")
      --
      m1:scal(0.5)
      local eq = true
      for i=1,m1:size() do eq = eq and m1:raw_get(i-1) == 105 end
      check.TRUE(eq, "scalar")
      --
      local aux = m1:sum(1):scal(1/300):sum(1):scal(1/200):sum()/100
      check.number_eq(aux, 105)
      local aux = m1:select(3,1):sum(1):scal(1/300):sum()/200
      check.number_eq(aux, 105)
      local m1 = nil
      collectgarbage("collect")
      --
      local m1 = matrix(300,200,100):uniformf()
      m1:set(290,190,1, -20)
      check.eq(-20, (m1:min()))
      check.eq(-20, (m1:min(1):min(1):min()))
      check.eq(-20, (m1:select(3,1):min()))
      check.eq(-20, (m1:select(3,1):min(1):min()))
  end)
end

--

do_test()
if util.is_cuda_available() then
  -- forces the use of CUDA
  mathcore.set_use_cuda_default(util.is_cuda_available())
  do_test()
end
