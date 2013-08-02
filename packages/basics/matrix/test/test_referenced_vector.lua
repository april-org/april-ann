a = util.vector_float()
for i = 1,50 do
  a:push_back(i)
end
m = a:toMatrix(false) -- do not reuse vector
print(m)

print("La longitud del vector es",a:size())

print("Tomamos el vector para la matriz")
m = a:toMatrix() -- reuse vector
print(m)
print("La longitud del vector es",a:size())

a = util.vector_uint()
for i = 1,10 do
  a:push_back(i)
end
print(a)
print(a:toMatrix())



