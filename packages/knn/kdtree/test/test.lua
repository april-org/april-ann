local N   = 1000000
local D   = 10
local rnd = random(1234)
local m1  = matrix(N,D):uniformf(-1,1,rnd)
local m2  = matrix(N,D):uniformf(-1,1,rnd)
--
local kdt = knn.kdtree(m1:dim(2),random(8424))
kdt:push(m1)
kdt:push(m2)
kdt:build()
-- KDTREE SEARCH
local p = matrix(1,D):uniformf(-1,1,rnd)
local bestid,bestdist,best = kdt:searchNN(p)
print(p)
print(bestid)
print(bestdist)
print("stats:",kdt:stats())
-- KNN SEARCH
local result = kdt:searchKNN(4,p)
for i=1,#result do
  print(i, result[i][1], result[i][2])
end
print("stats:",kdt:stats())
-- NAIVE LINEAR SEARCH
local besti
local bestd = math.huge
for i=1,m1:dim(1) do
  local aux = m1(i,':')
  local d = (aux-p):pow(2):sum()
  if d < bestd then bestd,besti = d,i end
end
for i=1,m2:dim(1) do
  local aux = m2(i,':')
  local d = (aux-p):pow(2):sum()
  local j = m1:dim(1)+i
  if d < bestd then bestd,besti = d,j end
end
print(besti)
print(bestd)
