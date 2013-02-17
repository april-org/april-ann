help_string =
  [[
     Command line args must be:

     april-ann  random-search.....lua  conf.lua  ["all_params.TAG.option='blah'" "global_vars.working_dir='blah'" "global_vars.n=blah" "global_vars.seed=blah"... ]

     This script executes a random search optimization of hyperparameters of any
     executable (not only april scripts). It receives one mandatory argument, a
     configuration file which must fullfil following template:

     return {
       fixed_params = {
         { option="--option-name",  value=ANY,  tag="outputtag", hidden=true },
         { option="--option-name2", value=ANY2, tag="outputtag"  },
         ...
       },
       random_params = {
         { option="--option-name", tag=ANY, sampling = "uniform",
           type="integer",
           values= { { min=ANY, max=ANY }, { min=ANY, max=ANY, step=ANY} } },
         { option="--option-name", tag=ANY, sampling = "uniform",
           prec=NUMBER, type="real",
           values= { { min=ANY, max=ANY }, { min=ANY, max=ANY, step=ANY} } },
         { option="--option-name", tag=ANY, sampling = "uniform", values= { a, b, c, ..., d } },
         { option="--option-name", tag=ANY, prec=NUMBER,
           sampling = "gaussian", values= { mean=ANY, variance=ANY } },
         { option="--option-name", tag=ANY, sampling = "random",
           check = FUNCTION },
         { option=nil, tag=ANY, sampling=.... }, -- this is a metaparameter example
         ...
       },
       check=function(params) return true end
       script = PATH_TO_SCRIPT,
       exec   = PATH_TO_APRIL_OR_OTHER_SYSTEM,
       working_dir = PATH_TO_WORKING_DIR,
       seed = ANY_SEED_VALUE (if not given, take random from bash)
                              n = number of iterations,
     }

     and any number of arguments which modify configuration file values. This arguments could access to:

     - all_params table which could be indexed by TAGs            => all_params.TAG.option='blah'
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

  -- This function check the correction of fixed_params
  local fixed_param_valid_options = table.invert{"option", "tag", "value",
						 "hidden" }
  function check_fixed(param)
    if not param.tag then error("Each fixed parameter needs a tag") end
    if not param.value then error("Each fixed parameter needs a value") end
    for name,v in pairs(param) do
      if not fixed_param_valid_options[name] then
        error("Incorrect fixed param option name: " .. name)
      end
    end
  end

  -- This function check the correction of random_params
  local random_param_valid_options = table.invert{"option", "tag", "values",
                                                  "sampling", "prec", "check",
                                                  "type"}
  local random_param_values_table_valid_options = table.invert{"min","max","step",
                                                               "mean", "variance"}
  function check_random(param)
    if not param.tag then error("Each random parameter needs a tag") end
    if not param.sampling then error("Each random parameter needs a sampling") end
    if not param.prec then param.prec = 10 end
    if not param.check then param.check = function(params) return true end end
    for name,v in pairs(param) do
      if not random_param_valid_options[name] then
        error("Incorrect random param option name: " .. name)
      end
    end
    if param.sampling == "gaussian" then
      if not param.values then error("Each random parameter needs values") end
      if not param.values.mean or not param.values.variance then
        error("Gaussian sampling needs mean and variance")
      end
    elseif param.sampling == "uniform" then
      if not param.values then error("Each random parameter needs values") end
      if type(param.values) ~= "table" then
        error("Uniform sampling needs values table")
      end
      if type(param.values[1]) == "table" then
        if param.type ~= "integer" and param.type ~= "real" then
          error("Needs integer or real type option: ")
        end
        local size = 0
        for _,p in ipairs(param.values) do
          for name,v in pairs(p) do
            if not random_param_values_table_valid_options[name] then
              error("Incorrect random values table option name: " .. name)
            end
          end
          if not p.step then p.step = 1 end
          if not p.min or not p.max then
            error("Values need min and max (optionally step) parameters")
          end
          p.size = (p.max - p.min)/p.step
          if param.type == "integer" then p.size = math.floor(p.size) end
          size   = size + p.size
        end
        param.size = size
      end
    elseif param.sampling == "random" then
      -- NOTHING TO DO
    else
      error("Incorrect sampling type")
    end
  end

  -- This function sample one value from the given param distribution and random
  -- number generator
  function sample(param, rnd)
    local v
    if param.sampling == "gaussian" then
      v = string.format("%."..param.prec.."f",
                        rnd:randNorm(param.values.mean, param.values.variance))
    elseif param.sampling == "random" then
      v = tostring(sample_random_from_bash())
    else
      if type(param.values[1]) == "table" then
        local pos = rnd:rand(param.size)
        for k,p in ipairs(param.values) do
          pos = pos - p.size
          if pos <= 0 or k==#param.values then
            if param.type == "integer" then
              v = tostring(rnd:randInt(0, p.size)*p.step + p.min)
            else
              v = string.format("%.".. param.prec .."f",
                                rnd:rand(p.max-p.min) + p.min)
            end
            break
          end
        end
      else
        v = tostring(rnd:choose(param.values))
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
  global_vars.check         = conf_table.check or (function(params) return true end)
  fixed_params              = conf_table.fixed_params or {}
  random_params             = conf_table.random_params or {}

  all_params = {}

  -- ERROR CHECK --
  for _,param in ipairs(fixed_params) do
    check_fixed(param)
    if all_params[param.tag] then
      error("The following tag is repeated: " .. param.tag)
    end
    all_params[param.tag] = param
  end
  for _,param in ipairs(random_params) do
    check_random(param)
    if all_params[param.tag] then
      error("The following tag is repeated: " .. param.tag)
    end
    all_params[param.tag] = param
  end

  -- Modify params by command line
  local valid_cmd_options = table.invert{ "seed", "working_dir",
                                          "exec", "script", "check", "n" }
  for i=2,#arg do
    -- load the chunk
    local chunk_func=loadstring(arg[i]) or error("Impossible to load string: "..
                                                 arg[i])
    -- execute the chunk
    safe_call(chunk_func, { all_params  = all_params,
                            global_vars = global_vars })
    --
    printf("# Executed chunk string: %s\n", arg[i])
  end

  local n           = global_vars.n
  local seed        = global_vars.seed
  local working_dir = global_vars.working_dir
  local exec        = global_vars.exec
  local script      = global_vars.script
  local check       = global_vars.check
  local rnd         = random(seed)

  printf("# Seed %d\n", global_vars.seed)
  printf("# Sampling over %d hyperparameters\n", #random_params)

  -- N random iterations
  for i=1,n do
    collectgarbage("collect")
    local params_check   = nil
    local args_table     = nil
    local filename_tags  = nil
    local filename       = nil
    local skip           = false
    -- First loop, until the combination of params were unique
    repeat
      -- Second loop, until param check function returns true
      repeat
        params_check  = {}
        args_table    = {}
        filename_tags = {}
        -- auxiliar function
        function put_value(option, tag, value, hidden)
          local value = tostring(value)
          if option then
            table.insert(args_table,string.format("%s%s", option, value))
          end
	  if not hidden then
	    table.insert(filename_tags,
			 string.format("%s:%s", tag,
				       string.gsub(value, "/", "_")))
	  end
        end
        for _,param in ipairs(fixed_params) do
          put_value(param.option, param.tag, param.value, param.hidden)
          params_check[param.tag] = param.value
        end
        for _,param in ipairs(random_params) do
          -- Third loop, until current param option check function returns true
	  local v
          repeat
            v = sample(param, rnd)
            params_check[param.tag] = v
          until safe_call(param.check, {}, params_check)
	  put_value(param.option, param.tag, v)
        end
      until safe_call(check, {}, params_check)
      filename=string.format("%s/output-%s.log", working_dir,
                             table.concat(filename_tags, "_"))
      skip = (io.open(filename, "r") ~= nil)
      if skip then printf ("# Skipping file %s\n", filename) end
    until not skip
    printf("# iteration %d :: %s\n", i, table.concat(filename_tags, " "))
    printf("# \t output file: %s\n", filename)
    local args_str = table.concat(args_table, " ")
    local cmd = string.format("%s %s %s", exec, script, args_str)
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
