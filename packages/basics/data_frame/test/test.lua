local df1 = data_frame()
local df2 = data_frame{ data = { one = {1,2,3,4},
                                 two = {5,6,7,8} },
                        columns = { "two", "one" } }
local df3 = data_frame{ data = matrix(4,20):linear() }
print(df1)
print(df2)
print(df3)

local df4 = data_frame.from_csv(aprilio.stream.input_lua_string[[id,cost
1,4
2,1
3,10
]])
print(df4)

df4:drop(2, "id")

df4:to_csv("blah.csv")

print(df3[3])

print(df2:as_matrix())
print(df3:as_matrix("complex", 2))
