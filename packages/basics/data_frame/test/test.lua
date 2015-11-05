local check    = utest.check
local T        = utest.test
local warning  = utest.warning

T("DataFrameTest", function()
    local df1 = data_frame.from_csv(aprilio.stream.c_string("a,b,c\n1,2,3\n4,5,6"),
                                    { columns={"d","e","f"} })
    check.eq(table.concat(df1:get_columns(),","), "d,e,f" )
    check.errored(function()
        local df1 = data_frame.from_csv(aprilio.stream.c_string("a,c\n1,3\n4,6"),
                                        { columns={"d","e","f"} })
    end)
    local df1 = data_frame.from_csv(aprilio.stream.c_string("a,b,c\n1,2,3\n4,5,6"))
    check.eq(table.concat(df1:get_columns(),","), "a,b,c")
    local df1 = data_frame()
    check.eq(#df1:get_index(), 0)
    check.eq(#df1:get_columns(), 0)
    check.NIL(df1[{"a"}])
    check.TRUE(tostring(df1))
    --
    local df2 = data_frame{ data = { one = {1,2,3,4},
                                     two = {5,6,7,8},
                                     three = { "A", "B", "B", "C" }, },
                            columns = { "two", "one", "three" },
                            index = { "a", "b", "c", "d" } }
    print(df2:groupby("three"):get_group("B"))
    print(df2:groupby("three", "one"):get_group("B",2))
    
    check.eq(#df2:get_index(), 4)
    check.eq(#df2:get_columns(), 3)
    check.eq(df2[{"three"}][1], "A")
    check.eq(df2[{"three"}][2], "B")
    check.eq(df2[{"three"}][3], "B")
    check.eq(df2[{"three"}][4], "C")
    local m_index   = df2:as_matrix("three", { dtype="categorical" })
    local m_sparse  = df2:as_matrix("three", { dtype="categorical",
                                               categorical_dtype="sparse" })
    local m_sparse2 = df2:as_matrix("three", "three",
                                    { dtype="categorical",
                                      categorical_dtype="sparse" })
    check.eq(m_index, matrix(4,1,{1,2,2,3}))
    check.eq(m_sparse, matrix.sparse(matrix(4,3,{1,0,0,0,1,0,0,1,0,0,0,1})))
    check.eq(m_sparse2,
             matrix.sparse(matrix(4,6,{1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1})))
    local m = df2:as_matrix("one", "two")
    check.eq(m, matrix(4,2,{1,5,2,6,3,7,4,8}))
    local df22 = data_frame{ data = { three = { "A", "C", "D" },
                                      four = { 3, 4, 5 } },
                             columns = { "three", "four" },
                             index = { "a", "e", "f" } }
    print(df2:merge(df22, { how="left"  }))
    print(df2:merge(df22, { how="right" }))
    -- print(df2:merge(df22, { how="inner" }))
    -- print(df2:merge(df22, { how="outer" }))
    --
    local df3 = data_frame{ data = matrix(4,20):linear() }
    local _   = df3[{3}]
    local _   = df3:as_matrix(2, { dtype="complex", })
    --
    local df4 = data_frame.from_csv(aprilio.stream.c_string[[id,cost
1,4
2,1
3,10
]])
    df4[{"cost"}] = {2,3,4}
    df4:drop(2, "id")
    local tmp = os.tmpname()
    df4:to_csv(tmp)
    os.remove(tmp)
    --
    local m = matrix(20,1):linspace()
    local df5 = data_frame()
    df5:insert(m)
    --
    local df6 = data_frame{ data = { one = { "A", "B", "B", "A" }, } }
    local m_index = df6:as_matrix({ dtype="categorical" })
    check.eq(m_index, matrix(4,1,{0,1,1,0}))
    check.errored(function()
        local m_index = df6:as_matrix({ dtype="categorical",
                                        categorical_dtype = "sparse" })
    end)
end)

T("TimeSeriesTest", function()
    local x = matrix(100):logspace(1,1000):toTable()
    local y = matrix(100, 1):logspace(1,10000)
    local df = data_frame{ data={ x=x, y=y } }
    local ts = data_frame.series(df, "x", "y")
    local tsp = ts:resampleU(12)
    local result = tsp:to_data_frame():as_matrix()
    local target = matrix.fromString[[81 2
ascii
24 69.562454223633
36 119.13578796387
48 174.67581176758
60 235.10885620117
72 299.74032592773
84 368.08569335938
96 439.77239990234
108 514.53790283203
120 592.09564208984
132 672.33374023438
144 755.01239013672
156 840.02197265625
168 927.25207519531
180 1016.5922851562
192 1107.9356689453
204 1201.1889648438
216 1296.2810058594
228 1393.1770019531
240 1491.8707275391
252 1592.0684814453
264 1693.8811035156
276 1797.4688720703
288 1902.2569580078
300 2008.7111816406
312 2116.6069335938
324 2225.6887207031
336 2336.4819335938
348 2448.1472167969
360 2561.6040039062
372 2675.83984375
384 2791.7150878906
396 2908.6181640625
408 3026.5341796875
420 3146.1591796875
432 3266.0690917969
444 3388.0759277344
456 3510.5451660156
468 3633.9838867188
480 3759.2885742188
492 3884.640625
504 4011.5393066406
516 4139.83984375
528 4268.1416015625
540 4398.0336914062
552 4529.3530273438
564 4660.671875
576 4793.0141601562
588 4927.3706054688
600 5061.7802734375
612 5196.404296875
624 5333.3442382812
636 5470.9174804688
648 5608.490234375
660 5746.794921875
672 5887.4311523438
684 6028.2412109375
696 6169.05078125
708 6310.6962890625
720 6454.6796875
732 6598.8017578125
744 6742.9248046875
756 6887.408203125
768 7034.4306640625
780 7181.9453125
792 7329.4599609375
804 7476.9755859375
816 7625.9130859375
828 7776.8823242188
840 7927.8676757812
852 8078.8520507812
864 8229.8388671875
876 8382.673828125
888 8537.212890625
900 8691.751953125
912 8846.291015625
924 9000.8310546875
936 9156.484375
948 9314.5732421875
960 9472.7490234375
972 9630.923828125
984 9789.099609375
]]
    check.eq(result, target)

    local tsp = ts:resampleU(12, { method="rectangle" })
    -- TODO: check method=rectangle output
end)

T("ParseCSVLineTest",
  function()
    local ref = 'Company2 Success 2011 3 125 nan Market Research|Marketing|Crowdfunding Marketing, sales nan nan nan No nan nan nan United States North America 5 0 2 0 4 20 No 0 medium Yes Large Yes Yes No No Product No Public Yes Both No Platform Local Non-Linear No Few Yes No Yes Yes No No Yes Yes Yes No Online B2C Low High Yes High Masters 21 Supply Chain Management & Entrepreneurship Yes Yes Tier_1 500 High 0 0 0 Medium 13 None 34 High Medium Yes No Low No Info No Yes Yes No No Yes Medium 1067034 Yes Yes No 3 Medium 0 6.666666667 5 Not Applicable 10 9 Trough 2 to 5 15.88235294 11.76470588 15 12.94117647 0 8.823529412 21.76470588 10.88235294 2.941176471 0 0 0 0 0 8'
    
    local x = 'Company2,Success,2011,3,125,,Market Research|Marketing|Crowdfunding,"Marketing, sales",,,,No,,,,United States,North America,5,0,2,0,4,20,No,0,medium,Yes,Large,Yes,Yes,No,No,Product,No,Public,Yes,Both,No,Platform,Local,Non-Linear,No,Few,Yes,No,Yes,Yes,No,No,Yes,Yes,Yes,No,Online,B2C,Low,High,Yes,High,Masters,21,Supply Chain Management & Entrepreneurship,Yes,Yes,Tier_1,500,High,0,0,0,Medium,13,None,34,High,Medium,Yes,No,Low,No Info,No,Yes,Yes,No,No,Yes,Medium,1067034,Yes,Yes,No,3,Medium,0,6.666666667,5,Not Applicable,10,9,Trough,2 to 5,15.88235294,11.76470588,15,12.94117647,0,8.823529412,21.76470588,10.88235294,2.941176471,0,0,0,0,0,8,'
    local y = 'Company2,Success,2011,3,125,,Market Research|Marketing|Crowdfunding,"Marketing, sales",,,,No,,,,United States,North America,5,0,2,0,4,20,No,0,medium,Yes,Large,Yes,Yes,No,No,Product,No,Public,Yes,Both,No,Platform,Local,Non-Linear,No,Few,Yes,No,Yes,Yes,No,No,Yes,Yes,Yes,No,Online,B2C,Low,High,Yes,High,Masters,21,Supply Chain Management & Entrepreneurship,Yes,Yes,Tier_1,500,High,0,0,0,Medium,13,None,34,High,Medium,Yes,No,Low,No Info,No,Yes,Yes,No,No,Yes,Medium,1067034,Yes,Yes,No,3,Medium,0,6.666666667,5,Not Applicable,10,9,Trough,2 to 5,15.88235294,11.76470588,15,12.94117647,0,8.823529412,21.76470588,10.88235294,2.941176471,0,0,0,0,0,"8",'
    local z = 'Company2 Success 2011 3 125  Market Research|Marketing|Crowdfunding "Marketing, sales"    No    United States North America 5 0 2 0 4 20 No 0 medium Yes Large Yes Yes No No Product No Public Yes Both No Platform Local Non-Linear No Few Yes No Yes Yes No No Yes Yes Yes No Online B2C Low High Yes High Masters 21 Supply Chain Management & Entrepreneurship Yes Yes Tier_1 500 High 0 0 0 Medium 13 None 34 High Medium Yes No Low No Info No Yes Yes No No Yes Medium 1067034 Yes Yes No 3 Medium 0 "6,666666667" 5 Not Applicable 10 9 Trough 2 to 5 "15,88235294" "11,76470588" 15 "12,94117647" 0 "8,823529412" "21,76470588" "10,88235294" "2,941176471" 0 0 0 0 0 "8" '
    
    local tx = util.__parse_csv_line__({}, x, ',', '"', '.', 'NA', nan)
    local ty = util.__parse_csv_line__({}, y, ',', '"', '.', 'NA', nan)
    local tz = util.__parse_csv_line__({}, z, ' ', '"', ',', 'NA', nan)
    
    local sx = table.concat(tx," ")
    local sy = table.concat(ty," ")
    local sz = table.concat(tz," ")

    check.eq(sx, sy)
    check.eq(sx, sz)
    check.eq(sx, ref)    
end)
