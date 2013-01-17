
a = matrix.fromString[[
1 3
ascii
1 2 3
]]

b = matrix.fromString[[
3 1
ascii
1
4
7
]]

c = a:multiply(b)

print(1*1+2*4+3*7)
print(c:toString())

d = matrix.fromString[[
3 3
ascii
1 2 3
4 5 6
7 8 9
]]

e = d:multiply(d)
print(e:toString())
