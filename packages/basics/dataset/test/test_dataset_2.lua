print("empezamos")
a = matrix(2,2,{1,2,3,4})
b = dataset.matrix(a)
print("UNO",a,b)
for i,j in b:patterns() do print(table.concat(j,",")) end
a = matrix(2,2,2,{1,2,3,4,5,6,7,8})
b = dataset.matrix(a)
print("DOS",a,b)
for i,j in b:patterns() do print(table.concat(j,",")) end
a = matrix(4,{1,2,3,4})
b = dataset.matrix(a)
print("TRES",a,b)
for i,j in b:patterns() do print(table.concat(j,",")) end
print("finalizamos")
