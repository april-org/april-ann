help_string =
  [[
     Command line args must be:

     april-ann  random-search.....lua  conf.lua  ["all_hyperparams.TAG.option='blah'" "global_vars.working_dir='blah'" "global_vars.n=blah" "global_vars.seed=blah"... ]

     This script executes a random search optimization of hyperparameters of any
     executable (not only april scripts). It receives one mandatory argument, a
     configuration file which must fullfil following template:

return {
  hyperparams = {
    { option="--o1=",  value=10,  tag="o1", sampling="fixed", hidden=true },
    { option="--o2=",  value=20,  tag="o2", sampling="fixed" },
    { option="--r1",   tag="r1", sampling = "log-uniform",
      type="integer",
      values= { { min=1, max=10 }, { min=20, max=100, step=10} },
      filter = function(hyperparams) hyperparams["r1"] = "200" return true end },
    { option="--r2=", tag="r2", sampling = "uniform", values= { 1, 2, 3, 4, 5 } },
    { option="--r3=", tag="r3", prec=3,
      sampling = "gaussian", values= { mean=0, variance=0.1 } },
    { option="--r4=", tag="r4", sampling = "random",
      filter = function(hyperparams)
	if hyperparams["r2"] == "1" then hyperparams["r4"] = "0" end return true
      end },
    { option=nil, tag="r5", sampling="random" }
  },
  filter = function(hyperparams) hyperparams['r5'] = '0.4' return true end,
  script = "",
  exec   = "echo",
  working_dir = "/tmp/",
  -- seed = ANY_SEED_VALUE (if not given, take random from bash)
  n = 50 }

     and any number of arguments which modify configuration file values. This arguments could access to:

     - all_hyperparams table which could be indexed by TAGs       => all_hyperparams.TAG.option='blah'
     - global_vars table which could be indexed by any global var => global_vars.seed=blah
   ]]

  if #arg < 1 then
    printf("%s\n", help_string)
    error("Syntax error")
  end

  printf("# HOST:\t %s\n", (io.popen("hostname", "r"):read("*l")))
  printf("# DATE:\t %s\n", (io.popen("date", "r"):read("*l")))
  printf("# CMD: \t %s\n",table.concat(arg, " "))

  ------------------------ Auxiliar functions ---------------------------------
  -- This sample random numbers from urandom device
  function sample_random_from_bash()
    return tonumber(io.popen("od -N2 -An -i /dev/urandom"):read("*l"))
  end

  -- This function check the correction of random_hyperparams
  local fixed_hyperparam_valid_options  = table.invert{"option", "tag", "value",
						       "hidden", "sampling" }
  local random_hyperparam_valid_options = table.invert{"option", "tag", "values",
						       "sampling", "prec", "filter",
						       "type", "size", "hidden" }
  local random_hyperparam_values_table_valid_options = table.invert{"min","max","step",
								    "mean", "variance",
								    "size" }
  function check_hyperparam(hyperparam)
    if not hyperparam.sampling then
      error("Each hyperhyperparameter needs a sampling")
    end
    if not hyperparam.tag then error("Each hyperparameter needs a tag") end
    if hyperparam.sampling == "fixed" then
      for name,v in pairs(hyperparam) do
	if not fixed_hyperparam_valid_options[name] then
	  error("Incorrect fixed hyperparam option name: " .. name)
	end
      end
      hyperparam.filter = function(hyperparams) return true end
    else
      if not hyperparam.prec then hyperparam.prec = 10 end
      if not hyperparam.filter then hyperparam.filter = function(hyperparams) return true end end
      for name,v in pairs(hyperparam) do
	if not random_hyperparam_valid_options[name] then
	  error("Incorrect random hyperparam option name: " .. name)
	end
      end
      if hyperparam.sampling == "gaussian" then
	if not hyperparam.values then
	  error("Each random hyperparameter needs values")
	end
	if not hyperparam.values.mean or not hyperparam.values.variance then
	  error("Gaussian sampling needs mean and variance")
	end
      elseif hyperparam.sampling == "uniform" or hyperparam.sampling == "log-uniform" then
	if not hyperparam.values then error("Each random hyperparameter needs values") end
	if type(hyperparam.values) ~= "table" then
	  error("Uniform sampling needs values table")
	end
	if type(hyperparam.values[1]) == "table" then
	  if hyperparam.type ~= "integer" and hyperparam.type ~= "real" then
	    error("Needs integer or real type field for TAG "..hyperparam.tag)
	  end
	  local size = 0
	  for _,p in ipairs(hyperparam.values) do
	    for name,v in pairs(p) do
	      if not random_hyperparam_values_table_valid_options[name] then
		error("Incorrect random values table option name: " .. name)
	      end
	    end
	    if not p.step then p.step = 1 end
	    if not p.min or not p.max then
	      error("Values need min and max (optionally step) hyperparameters")
	    end
	    p.size = (p.max - p.min)/p.step
	    if hyperparam.type == "integer" then p.size = math.floor(p.size) end
	    size   = size + p.size
	  end
	  hyperparam.size = size
	end
      elseif hyperparam.sampling == "random" then
	-- NOTHING TO DO
      else
	error("Incorrect sampling type")
      end
    end
  end

  -- This function sample one value from the given hyperparam distribution and random
  -- number generator
  function sample(hyperparam, rnd)
    local v
    if hyperparam.sampling == "fixed" then
      v = tostring(hyperparam.value)
    elseif hyperparam.sampling == "gaussian" then
      v = string.format("%."..hyperparam.prec.."f",
                        rnd:randNorm(hyperparam.values.mean, hyperparam.values.variance))
    elseif hyperparam.sampling == "random" then
      v = tostring(sample_random_from_bash())
    else -- uniform and log-uniform
      if type(hyperparam.values[1]) == "table" then
        local pos = rnd:rand(hyperparam.size)
        for k,p in ipairs(hyperparam.values) do
          pos = pos - p.size
          if pos <= 0 or k==#hyperparam.values then
            if hyperparam.type == "integer" then
	      local r
	      if hyperparam.sampling == "uniform" then
		r = rnd:randInt(0, p.size)*p.step + p.min
	      else
		if p.step ~= 1 then
		  error("Step must be one for log-uniform integers")
		end
		r = rnd:rand(math.log(p.max)-math.log(p.min)) + math.log(p.min)
		r = math.round(math.exp(r))
	      end
              v = tostring(r)
            else -- if hyperparam.type == "integer" ...
	      local r
	      if hyperparam.sampling == "uniform" then
		r = rnd:rand(p.max-p.min) + p.min
	      else
		r = rnd:rand(math.log(p.max)-math.log(p.min)) + math.log(p.min)
		r = math.exp(r)
	      end
	      v = string.format("%.".. hyperparam.prec .."f", r)
	    end -- if hyperparam.type == "integer" else ...
            break
          end -- if pos <= 0 or k == #hyperparam.values ...
        end -- for k,p in ipairs(hyperparam.values)
      else -- if type(hyperparam.values[1]) == "table" ...
	if hyperparam.sampling == "log-uniform" then
	  error("log-uniform sampling don't works with a finite table")
	end
        v = tostring(rnd:choose(hyperparam.values))
      end
    end
    return v
  end
  ------------------------------------------------------------------------------

  conf_table = (dofile(arg[1] or error("Needs a configuration file")) or
                error("Impossible to load configuration file"))
  local global_vars = {}
  global_vars.n = conf_table.n or error ("The configuration file needs "..
                                         "a n option")
  global_vars.seed          = tonumber(conf_table.seed or sample_random_from_bash())
  global_vars.working_dir   = conf_table.working_dir or error("Needs a working_dir option")
  global_vars.exec          = conf_table.exec or error("Needs an exec option")
  global_vars.script        = conf_table.script or error("Needs a script option")
  global_vars.filter        = conf_table.filter or (function(hyperparams) return true end)
  hyperparams_conf_tbl      = conf_table.hyperparams or {}
  
  all_hyperparams = {}

  -- ERROR CHECK --
  for _,hyperparam in ipairs(hyperparams_conf_tbl) do
    check_hyperparam(hyperparam)
    if all_hyperparams[hyperparam.tag] then
      error("The following tag is repeated: " .. hyperparam.tag)
    end
    all_hyperparams[hyperparam.tag] = hyperparam
  end

  -- Modify hyperparams by command line
  local valid_cmd_options = table.invert{ "seed", "working_dir",
                                          "exec", "script", "filter", "n" }
  for i=2,#arg do
    -- load the chunk
    local chunk_func=loadstring(arg[i]) or error("Impossible to load string: "..
                                                 arg[i])
    -- execute the chunk
    safe_call(chunk_func, { all_hyperparams = all_hyperparams,
                            global_vars     = global_vars })
    --
    printf("# Executed chunk string: %s\n", arg[i])
  end

  local n           = global_vars.n
  local seed        = global_vars.seed
  local working_dir = global_vars.working_dir
  local exec        = global_vars.exec
  local script      = global_vars.script
  local filter      = global_vars.filter
  local rnd         = random(seed)

  printf("# Seed %d\n", global_vars.seed)
  printf("# Sampling over %d hyperparameters\n", #hyperparams_conf_tbl)

  -- N random iterations
  for i=1,n do
    collectgarbage("collect")
    local hyperparam_values  = nil
    local args_table         = nil
    local filename_tags      = nil
    local filename           = nil
    local skip               = false
    local max_iters          = 1000
    local iter1_count        = 0
    -- First loop, until the combination of hyperparams were unique
    repeat
      iter1_count = iter1_count + 1
      if iter1_count > max_iters then
	error("Possible infinite loop, all possible hyperparam values "..
	      "are sampled yet")
      end
      -- Second loop, until hyperparam filter function returns true
      local iter2_count = 0
      repeat
	iter2_count = iter2_count + 1
	if iter2_count > max_iters then
	  error("Possible infinite loop due to global filter function\n"..
		"Check that it is not always returning false or nil")
	end
	hyperparam_values  = {}
        args_table         = {}
        filename_tags      = {}
        -- auxiliar function
        function put_value(option, tag, value, hidden)
          local value = tostring(value)
          if option then
	    local v = value
	    if #v>0 then v = "\""..v.."\"" end
            table.insert(args_table,string.format("%s%s", option, v))
          end
	  if not hidden then
	    table.insert(filename_tags,
			 string.format("%s:%s", tag,
				       string.gsub(value, "/", "_")))
	  end
        end
        for _,hyperparam in ipairs(hyperparams_conf_tbl) do
          -- Third loop, until current filter function returns true
	  local v
	  local iter3_count = 0
          repeat
	    iter3_count = iter3_count + 1
	    if iter3_count > max_iters then
	      error("Possible infinite loop due to filter function of: "..
		    hyperparam.tag .."\n"..
		    "Check that it is not always returning false or nil")
	    end
            v = sample(hyperparam, rnd)
            hyperparam_values[hyperparam.tag] = v
          until safe_call(hyperparam.filter, {}, hyperparam_values)
        end
      until safe_call(filter, {}, hyperparam_values)
      for _,hyperparam in ipairs(hyperparams_conf_tbl) do
	local tag = hyperparam.tag
	if type(hyperparam_values[tag]) ~= "string" then
	  print("WARNING!!! Please, check that all filter functions set"..
		" hyperparams to string type values, at least this "..
		"one: " .. tag)
	end
	put_value(hyperparam.option, tag, tostring(hyperparam_values[tag]),
		  hyperparam.hidden)
      end
      
      filename=string.format("%s/output-%s.log", working_dir,
                             table.concat(filename_tags, "_"))
      skip = (io.open(filename, "r") ~= nil)
      if skip then printf ("# Skipping file %s\n", filename) end
    until not skip
    printf("# ITERATION %d :: %s\n", i, table.concat(filename_tags, " "))
    printf("# \t output file: %s\n", filename)
    local args_str = table.concat(args_table, " ")
    local cmd      = string.format("%s %s %s", exec, script, args_str)
    printf("# \t executed string: %s\n", cmd)
    printf("# \t hyperparam values:\n")
    for _,hyperparam in ipairs(hyperparams_conf_tbl) do
      printf("# \t \t %s = %s\n", hyperparam.tag,
	     hyperparam_values[hyperparam.tag])
    end
    local f = io.popen(cmd)
    local g = io.open(filename, "w") or
      error(string.format("Impossible to open logfile '%s'", filename))
    for line in f:lines() do
      g:write(line.."\n")
      g:flush()
    end
    f:close()
    g:close()
  end
