local check    = utest.check
local T        = utest.test
local warning  = utest.warning

T("DataFrameTest", function()
    local df1 = data_frame()
    check.eq(#df1:get_index(), 0)
    check.eq(#df1:get_columns(), 0)
    check.NIL(df1[{"a"}])
    check.TRUE(tostring(df1))
    --
    local df2 = data_frame{ data = { one = {1,2,3,4},
                                     two = {5,6,7,8},
                                     three = { "A", "B", "B", "A" }, },
                            columns = { "two", "one", "three" },
                            index = { "a", "b", "c", "d" } }
    check.eq(#df2:get_index(), 4)
    check.eq(#df2:get_columns(), 3)
    check.eq(df2[{"three"}][1], "A")
    check.eq(df2[{"three"}][2], "B")
    check.eq(df2[{"three"}][3], "B")
    check.eq(df2[{"three"}][4], "A")
    local m_index   = df2:as_matrix("three", { dtype="categorical" })
    local m_sparse  = df2:as_matrix("three", { dtype="categorical",
                                               categorical_dtype="sparse" })
    local m_sparse2 = df2:as_matrix("three", "three",
                                    { dtype="categorical",
                                      categorical_dtype="sparse" })
    check.eq(m_index, matrix(4,1,{1,2,2,1}))
    check.eq(m_sparse, matrix.sparse(matrix(4,2,{1,0,0,1,0,1,1,0})))
    check.eq(m_sparse2,
             matrix.sparse(matrix(4,4,{1,0,1,0,0,1,0,1,0,1,0,1,1,0,1,0})))
    local m = df2:as_matrix("one", "two")
    check.eq(m, matrix(4,2,{1,5,2,6,3,7,4,8}))
    --
    local df3 = data_frame{ data = matrix(4,20):linear() }
    local _   = df3[{3}]
    local _   = df3:as_matrix(2, { dtype="complex", })
    --
    local df4 = data_frame.from_csv(aprilio.stream.input_lua_string[[id,cost
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
end)