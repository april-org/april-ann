local df1 = data_frame()
local df2 = data_frame{ data = { one = {1,2,3,4},
                                 two = {5,6,7,8},
                                 three = { "A", "B", "B", "A" }, },
                        columns = { "two", "one", "three" },
                        index = { "a", "b", "c", "d" } }
local df3 = data_frame{ data = matrix(4,20):linear() }
print(df1)
print(df2)
print(df3)
pprint(df2:as_matrix("three", { dtype="categorical" }))
pprint(df2:as_matrix("three", { dtype="categorical",
                                categorical_dtype="sparse" }))
pprint(df2:as_matrix("three", "three", { dtype="categorical",
                                         categorical_dtype="sparse" }))

-- print(df2:iloc(2))
-- print(df2:loc("b"))

local df4 = data_frame.from_csv(aprilio.stream.input_lua_string[[id,cost
1,4
2,1
3,10
]])
print(df4)
df4[{"cost"}] = {2,3,4}
print(df4)

-- pprint(df4:loc(2))

df4:drop(2, "id")

df4:to_csv("blah.csv")
os.remove("blah.csv")

print(df3[{3}])

print(df2:as_matrix("one", "two"))
print(df3:as_matrix(2, { dtype="complex", }))

local m = matrix(20,1):linspace()
local df5 = data_frame()
df5:insert(m)
