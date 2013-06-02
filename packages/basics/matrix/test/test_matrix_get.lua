m = matrix(3,4,5)

local aux = {}
for i=1,3*4*5 do
  table.insert(aux,i)
end
m:set(aux)

for i=1,3 do
  for j=1,4 do
    for k=1,5 do
      print(i,j,k,m:getElement(i,j,k))
    end
  end
end
