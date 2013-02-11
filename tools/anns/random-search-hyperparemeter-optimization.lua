-- Template for configuration
-- return {
--    fixed_params = {
--             { option="--option-name",  value=ANY,  tag="outputtag"  },
--             { option="--option-name2", value=ANY2, tag="outputtag"  },
--             ...
--    },
--    random_params = {
--             { option="--option-name", tag=ANY, sampling = "uniform", type="integer"|"real", values= { { min=ANY, max=ANY }, { min=ANY, max=ANY, step=ANY} } },
--             { option="--option-name", tag=ANY, sampling = "uniform", values= { a, b, c, ..., d } },
--             { option="--option-name", tag=ANY, sampling = "gaussian", values= { mean=ANY, variance=ANY } },
--             ...
--   },
--   script = PATH_TO_SCRIPT,
--   exec   = PATH_TO_APRIL_OR_OTHER_SYSTEM,
--   working_dir = PATH_TO_WORKING_DIR,
--   seed = ANY_SEED_VALUE (if not given, take "echo $RANDOM" as seed)
--   n = number of iterations,
-- }

printf("# HOST:\t %s\n", (io.popen("hostname", "r"):read("*l")))
printf("# DATE:\t %s\n", (io.popen("date", "r"):read("*l")))
printf("# CMD: \t %s\n",table.concat(arg, " "))

------------------------ Auxiliar functions ---------------------------------
function check_fixed(param)
  if not param.option then error("Each fixed parameter needs an option") end
  if not param.tag then error("Each fixed parameter needs a tag") end
  if not param.value then error("Each fixed parameter needs a value") end
end

function check_random(param)
  if not param.option then error("Each random parameter needs an option") end
  if not param.tag then error("Each random parameter needs a tag") end
  if not param.sampling then error("Each random parameter needs a sampling") end
  if not param.values then error("Each random parameter needs values") end
  if param.sampling == "gaussian" then
    if not param.values.mean or not param.values.variance then
      error("Gaussian sampling needs mean and variance")
    end
  elseif param.sampling == "uniform" then
    if type(param.values) ~= "table" then
      error("Uniform sampling needs values table")
    end
    if type(param.values[1]) == "table" then
      if param.type ~= "integer" and param.type ~= "real" then
	error("Needs integer or real type option: ")
      end
      local size = 0
      for _,p in ipairs(param.values) do
	if not p.step then p.step = 1 end
	if not p.min or not p.max then
	  error("Values need min and max (optionally step) parameters")
	end
	p.size = math.floor((p.max - p.min)/p.step)
	size   = size + p.size
      end
      param.size = size
    end
  else
    error("Incorrect sampling type")
  end
end

function sample(param, rnd)
  local v
  if param.sampling == "gaussian" then
    v = rnd:randNorm(param.values.mean, param.values.variance)
  else
    if type(param.values[1]) == "table" then
      local pos = rnd:randInt(1, param.size)
      for _,p in ipairs(param.values) do
	pos = pos - p.size
	if pos <= 0 then
	  if param.type == "integer" then
	    v = rnd:randInt(0, p.size)*p.step + p.min
	  else
	    v = rnd:rand(p.max-p.min) + p.min
	  end
	  break
	end
      end
    else
      v = rnd:choose(param.values)
    end
  end
  return v
end
------------------------------------------------------------------------------

conf_table = (dofile(arg[1] or error("Needs a configuration file")) or
	      error("Impossible to load configuration file"))
local num_iterations = conf_table.n or error ("The configuration file needs "..
					      "a num_iterations option")
local seed          = tonumber(conf_table.seed or
			       io.popen("echo $RANDOM"):read("*l"))
local working_dir   = conf_table.working_dir or error("Needs a working_dir option")
local exec          = conf_table.exec or error("Needs an exec option")
local script        = conf_table.script or error("Needs a script option")
local fixed_params  = conf_table.fixed_params or {}
local random_params = conf_table.random_params or {}
local rnd           = random(seed)

-- ERROR CHECK --
for _,param in ipairs(fixed_params) do
  check_fixed(param)
end
for _,param in ipairs(random_params) do
  check_random(param)
end

for i=1,num_iterations do
  local args_table = {}
  local filename_args = {}
  function put_value(option, tag, value)
    table.insert(args_table, string.format("%s%s", option, tostring(value)))
    table.insert(filename_args, string.format("%s%s", tag, tostring(value)))
  end
  for _,param in ipairs(fixed_params) do
    put_value(param.option, param.tag, param.value)
  end
  for _,param in ipairs(random_params) do
    put_value(param.option, param.tag, sample(param, rnd))
  end
  local filename=string.format("%s/output-%s.log", working_dir,
			       table.concat(filename_args, "_"))
  local args_str = table.concat(args_table, " ")
  local cmd = string.format("%s %s %s", exec, script, args_str)
  local f = io.popen(cmd)
  if not io.open(filename, "r") then
    local g = io.open(filename, "w")
    for line in f:lines() do
      g:write(line.."\n")
      g:flush()
    end
    f:close()
    g:close()
  else
    printf ("# Skipping file %s\n", filename)
  end
end
