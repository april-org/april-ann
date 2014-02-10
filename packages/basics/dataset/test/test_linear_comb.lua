
-- This test illustrates how to use a dataset.linearcomb

-- first, we have to create a dataset.linear_comb_conf
-- which receives a vector of tables
-- (one table per output of the final dataset).
-- Each table contains a list of tuples (index,weight).

-- Let us create a dataset with 2 patterns of size 3:
ds = dataset.matrix(matrix(2,3,
			   {1,2,3,
			    4,5,6}))

-- now, we create a dataset.linear_comb_conf to linearly combine
-- patterns of size 3 into patterns of size 2 so that:
-- the first one is 0.5 + 8*pattern(1) + 10*pattern(3)
-- the second one is -1 + pattern(2) + 5*pattern(3)

dlc = dataset.linear_comb_conf{
  -- first output pattern:
  { {0,0.5},{1,8},{3,10} },
  -- second output pattern:
  { {0,-1},{2,1},{3,5} },
}

-- finally, we apply "dlc" to obtain a dataset.linearcomb from ds:

resul = dataset.linearcomb(ds,dlc)

print("Original dataset")
for ipat,pat in ds:patterns() do
  print(ipat,table.concat(pat,","))
end

print("Linear combination dataset")
for ipat,pat in resul:patterns() do
  print(ipat,table.concat(pat,","))
end

-- The output is as follows:

-- Original dataset
-- 1	1,2,3
-- 2	4,5,6
-- Linear combination dataset
-- 1	38.5,16
-- 2	92.5,34

-- Observe that
-- 38.5 = 0.5+8*1+10*3
-- 16   = -1 + 2 + 5*3

-- 92.5 = 0.5+8*4+10*6
-- 34   = -1 + 5 + 5*6
 
