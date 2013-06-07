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

print("= a")
print(a)
print("= b")
print(b)
print("= a*b = " .. 1*1+2*4+3*7)
print(c)

print("= b*a")
print(b*a)

d = matrix.fromString[[
3 3
ascii
1 2 3
4 5 6
7 8 9
]]

print("= d")
print(d)
print("= d:clone('col_major')")
print(d:clone("col_major"))
print("= d:transpose()")
print(d:transpose())

e = d * d 
print("= d:mul(d)")
print(e)

d = d:clone("col_major")
e = d * d
print("= d:mul(d) in col_major")
print(e)

h = d:slice({2,2},{2,2})
print("= d:slice({2,2},{2,2},true) in col_major")
print(h)

e = h * h
print("= h:mul(h)")
print(e)

l = matrix.col_major(2,2):fill(4) + h
print(l)

print("= g")
g = matrix(3,2,{1,2,
		3,4,
		5,6})
print(g)
print("= g:transpose():clone('col_major'))")
print(g:transpose():clone("col_major"))
print("= g:transpose():clone('col_major'):clone('row_major')")
print(g:transpose():clone("col_major"):clone("row_major"))

print(g)

print(g:transpose())
j = g:transpose() * g
print("= g:transpose():mul(g)")
print(j)

j = matrix(2,2):gemm{
  trans_A=true, trans_B=false,
  alpha=1.0, A=g, B=g,
  beta=0.0
}
print("= gemm{ ... }")
print(j)

print("= col_major gemm{ ... }")
j = matrix.col_major(2,2):gemm{
  trans_A=true, trans_B=false,
  alpha=1.0, A=g:clone("col_major"), B=g:clone("col_major"),
  beta=0.0
}
print(j)
