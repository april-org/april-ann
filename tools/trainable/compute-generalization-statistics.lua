if #arg < 5 then
  print("Syntax error!")
  printf("\t%s FILENAME VAL_MEAN_COL VAL_VAR_COL TEST_MEAN_COL TEST_VAR_COL [W_REPS [EXP_REPS [EXP1_SIZE [EXP2_SIZE ...] ] ] ]\n", arg[0])
  printf("\nThis script computes the generalization ability given a set of experiments\n")
  printf("using different hyperparameters.\n\n")
  printf("\tuse - as FILENAME for stdin\n")
  printf("\tVAL_MEAN_COL is the column with the MEAN for your loss-function\n")
  printf("\t             over all validation samples\n")
  printf("\tVAL_VAR_COL is the column with the VARIANCE of the MEAN for your\n")
  printf("\t            loss-function over all validation samples\n")
  printf("\tTEST_MEAN_COL idem but with TEST\n")
  printf("\tTEST_VAR_COL idem but with TEST\n")
  printf("\tW_REPS is the number of simulation repetitions to compute\n")
  printf("\t       generalization weights (by default 1000)\n")
  printf("\tEXP_REPS is the number of simulation repetitions to compute\n")
  printf("\t         box-whiskers plots for each size (by default 1)\n")
  printf("\tThe following arguments are as many experiment sizes as you need\n")
  printf("\t(taken as subset of all your data), by default num_lines_of(FILENAME)\n")
  printf("\n\tSee 2012, Bergstra and Bengio, 'Random Search for Hyper-Parameter Optimization'\n")
  printf("\tfor more details: http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf\n")
  os.exit(1)
end
datafilename     = arg[1]
val_mean_column  = tonumber(arg[2])
val_var_column   = tonumber(arg[3])
test_mean_column = tonumber(arg[4])
test_var_column  = tonumber(arg[5])
repetitions      = tonumber(arg[6] or 1000)
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
    if #expsizes == 1 and expsize_reps == 1 then
      print(expsize, test_mean, test_var)
    end
    table.insert(test_means_tbl, test_mean)
  end
  if #test_means_tbl > 1 then
    -- sort test_means_tbl
    table.sort(test_means_tbl)
    -- compute box-whisker data
    local q0 = 1
    local q1 = math.max(1,math.round(#test_means_tbl/4))
    local q2 = math.max(1,math.round(#test_means_tbl/2))
    local q3 = math.max(1,math.round(#test_means_tbl*3/4))
    local q4 = #test_means_tbl
    print(expsize,
	  test_means_tbl[q0],
	  test_means_tbl[q1],test_means_tbl[q2],test_means_tbl[q3],
	  test_means_tbl[q4])
  end
end
