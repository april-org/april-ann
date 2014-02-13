
id5  = dataset.identity(5)
id10 = dataset.identity(10)
dsBase = dataset.matrix(matrix(3,2,{1,4,  2,2,   5,10}))

print("Base dataset")
for ipat,pat in dsBase:patterns() do
  print(ipat,table.concat(pat,","))
end

-- patternSize of dsBase is 2, so there are 2 datasets used as
-- dictionaries in the second argument:
dsIndexed = dataset.indexed(dsBase, {id5,id10})

print("Indexed dataset")
for ipat,pat in dsIndexed:patterns() do
  print(ipat,table.concat(pat,","))
end

-- The output is:

-- Base dataset
-- 1	1,4
-- 2	2,2
-- 3	5,10
-- Indexed dataset
-- 1	1,0,0,0,0,0,0,0,1,0,0,0,0,0,0
-- 2	0,1,0,0,0,0,1,0,0,0,0,0,0,0,0
-- 3	0,0,0,0,1,0,0,0,0,0,0,0,0,0,1

-- Observe the indexed dataset if we artificially separate its
-- patterns in two blocks of sizes 5 and 10:

-- Indexed dataset
-- 1	1,0,0,0,0,   0,0,0,1,0,0,0,0,0,0
-- 2	0,1,0,0,0,   0,1,0,0,0,0,0,0,0,0
-- 3	0,0,0,0,1,   0,0,0,0,0,0,0,0,0,1

