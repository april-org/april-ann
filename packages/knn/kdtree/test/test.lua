rnd = random(1234)
m1  = matrix(10,4):uniformf(0,1,rnd)
m2  = matrix(10,4):uniformf(0,1,rnd)
print(m1)
print(m2)
kdt = knn.kdtree(m1:dim(2),rnd)
kdt:push(m1)
kdt:push(m2)
kdt:build()
kdt:print()
p = matrix(1,4):uniformf(0,1,rnd)
bestid,bestdist,best = kdt:searchNN(p)
print(p)
print(bestid)
print(bestdist)

for i=1,m1:dim(1) do
  local aux = m1(i,':')
  print(i, (aux-p):pow(2):sum())
end
for i=1,m2:dim(1) do
  local aux = m2(i,':')
  print(m1:dim(1)+i, (aux-p):pow(2):sum())
end
