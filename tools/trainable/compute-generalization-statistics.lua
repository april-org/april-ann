datafilename     = arg[1]
val_mean_column  = tonumber(arg[2])
val_var_column   = tonumber(arg[3])
test_mean_column = tonumber(arg[4])
test_var_column  = tonumber(arg[5])
repetitions      = tonumber(arg[6])
expsize_reps     = tonumber(arg[7] or 1)
expsizes         = {}
for i=8,#arg do table.insert(expsizes, tonumber(arg[i])) end

if datafilename == "-" then datafilename = nil end

function sample_random_from_bash()
  return tonumber(io.popen("od -N2 -An -i /dev/urandom"):read("*l"))
end

-- read data
data = {}
for line in io.uncommented_lines(datafilename) do
  local t    = string.tokenize(line)
  local val_mean  = tonumber(t[val_mean_column])
  local val_var   = tonumber(t[val_var_column])
  local test_mean = tonumber(t[test_mean_column])
  local test_var  = tonumber(t[test_var_column])
  table.insert(data, { val_mean  = val_mean,
		       val_var   = val_var,
		       test_mean = test_mean,
		       test_var  = test_var,
		       sum_mean2_var2 = (test_mean*test_mean +
					 test_var*test_var),
		       distribution = random(sample_random_from_bash()) })
end
if #expsizes == 0 then table.insert(expsizes, #data) end
-----------------------------------------
rnd = random(sample_random_from_bash())

-- for each random experiment size
for _,expsize in ipairs(expsizes) do
  -- table to keep computed test means
  local test_means_tbl = {}
  for rep=1,expsize_reps do
    -- populate expsize-sized random experiment
    local wins       = {}
    local population = {}
    for i=1,expsize do
      table.insert(population, data[rnd:randInt(1,#data)])
      table.insert(wins, 0)
    end
    -- sampling loop for estimation of random experiment weights
    for i=1,repetitions do
      local min    = nil
      local argmin = 0
      -- sample from each hyperparameter set distribution
      for j,one_exp in ipairs(population) do
	local sampled_error = one_exp.distribution:randNorm(one_exp.val_mean,
							    one_exp.val_var)
	if argmin == 0 or sampled_error < min then
	  argmin,min = j,sampled_error
	end
      end
      -- wins measures how many times a given set of hyperparameters was the
      -- best, so at the end is possible to compute weights normalizing this
      -- table
      wins[argmin] = wins[argmin] + 1
    end
    -----------------------------------------
    -- compute test mean and test variance
    local test_mean = 0
    local test_var  = 0
    for i,one_exp in ipairs(population) do
      local weight = wins[i]/repetitions
      test_mean = test_mean + weight * one_exp.test_mean
      test_var  = test_var  + weight * one_exp.sum_mean2_var2
    end
    test_var = test_var - test_mean*test_mean
    -----------------------------------------
    -- print(expsize, test_mean, test_var)
    table.insert(test_means_tbl, test_mean)
  end
  -- sort test_means_tbl
  table.sort(test_means_tbl)
  -- compute box-whisker data
  local q0 = 1
  local q1 = math.round(#test_means_tbl/4)
  local q2 = math.round(#test_means_tbl/2)
  local q3 = math.round(#test_means_tbl*3/4)
  local q4 = #test_means_tbl
  print(expsize,
	test_means_tbl[q0],
	test_means_tbl[q1],test_means_tbl[q2],test_means_tbl[q3],
	test_means_tbl[q4])
end
