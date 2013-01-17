
a = matrix.fromString[[
# primero vienen las dimensiones de la matriz
4 3
# y luego el tipo
ascii
# y los datos:
1 2 3,,,,, 
4 5 6
7 8 9
10 11 12
]]

dims = a:dim()
print(string.format("{%s}",table.concat(dims,',')))

printf('En ascii: "%s"\n',a:toString())

printf('En binario: "%s"\n',a:toString("binary"))


txt = a:toString("binary")
c = matrix.fromString(txt)

print('Leemos binario y lo volvemos a mostrar en ascii:\n"'..c:toString()..'"')

x = matrix.fromString[[
2 5
ascii
0 2 3 4 5
6 7 8 9 10
]]

print(x:toString())
print("ahora normalizamos entre 0 y 1")
x:adjust_range(0,1)
print(x:toString())


x = matrix.fromString[[
2 5
ascii
1 1 1 1 1
1 1 1 1 1
]]

print(x:toString())
print("ahora normalizamos entre 0 y 1")
x:adjust_range(0,1)
print(x:toString())
