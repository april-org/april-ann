if #arg ~= 5 then
  print("Syntax error!")
  printf("\t%s FILENAME NUMCOL1 NUMCOL2 SEED CONF\n", arg[0])
  printf("\t\tuse - as FILENAME for stdin\n")
  printf("\t\tNUMCOL1 and NUMCOL2 are column indexes of your data\n")
  printf("\t\tSEED is a random number generator seed for bootstrap confidences\n")
  printf("\t\tCONF is the confidence value (0.95 for example)\n")
  os.exit(1)
end
filename = arg[1]
col1     = tonumber(arg[2])
col2     = tonumber(arg[3])
seed     = tonumber(arg[4])
conf     = tonumber(arg[5])
reps     = 1000

if conf > 0.95 then reps = 10000  end
if conf > 0.99 then reps = 100000 end

local f
if filename == "-" then f = io.stdin
else f = io.open(filename, "r") end

function zero_if_null(v)
  if v == "None" or v == "Undefined" or v == "Unintialized" or v == "Null" then
    v = 0
  end
  return v
end

local N      = 0
local x      = {}
local y      = {}
for line in f:lines() do
  local t  = string.tokenize(line)
  if string.sub(t[1], 1, 1) ~= "#" then
    t[col1] = zero_if_null(t[col1])
    t[col2] = zero_if_null(t[col2])
    local v1 = tonumber(t[col1]) or
      error("Impossible to convert to number col1: " .. t[col1])
    local v2 = tonumber(t[col2]) or
      error("Impossible to convert to number col2: " .. t[col2])
    table.insert(x, v1)
    table.insert(y, v2)
    N = N + 1
  end
end

fprintf(io.stderr, "%d repetitions\n", reps)

local rnd    = random(seed)
local data   = {}
for rep=1,reps do
  local xy_sum = 0
  local x_sum  = 0
  local y_sum  = 0
  local x2_sum = 0
  local y2_sum = 0
  for k=1,#x do
    local j = rnd:randInt(1,#x)
    local v1 = x[j]
    local v2 = y[j]
    xy_sum = xy_sum + v1*v2
    x_sum  = x_sum  + v1
    y_sum  = y_sum  + v2
    x2_sum = x2_sum + v1*v1
    y2_sum = y2_sum + v2*v2
  end
  
  local sx  = math.sqrt(N * x2_sum - x_sum*x_sum)
  local sy  = math.sqrt(N * y2_sum - y_sum*y_sum)
  local rxy = (N * xy_sum - x_sum*y_sum) / ( sx*sy )
  
  table.insert(data,{ rxy = rxy, sx = sx, sy = sy,
		      y_sum = y_sum,
		      x_sum = x_sum })
  if math.mod(rep,100)==0 then fprintf(io.stderr, "\r%3.0f%%", rep/reps*100) end
  io.stderr:flush()
end
fprintf(io.stderr, " done\n")
table.sort(data, function(a,b) return a.rxy < b.rxy end)
local med_conf_size = reps*(1.0 - conf)*0.5
local a = math.max(1,    math.round(med_conf_size))
local b = math.min(reps, math.round(reps - med_conf_size))

local rxy1 = data[a].rxy
local rxy2 = data[b].rxy

-- linear regression fitting: y = alpha + beta * x
local beta1   = rxy1 * data[a].sy / data[a].sx
local alpha1  = data[a].y_sum/N - beta1 * data[a].x_sum/N

local beta2   = rxy2 * data[b].sy / data[b].sx
local alpha2  = data[b].y_sum/N - beta2 * data[b].x_sum/N

printf("rxy=    % .4f +- % .4f [% .4f, % .4f]\n",
       (rxy1 + rxy2)/2, math.abs(rxy1-rxy2)*0.5, rxy1, rxy2)
printf("alpha=  % .4f +- % .4f [% .4f, % .4f]\n",
       (alpha1 + alpha2)/2, math.abs(alpha1-alpha2)*0.5, alpha1, alpha2)
printf("beta=   % .4f +- % .4f [% .4f, % .4f]\n",
       (beta1 + beta2)/2, math.abs(beta1+beta2)*0.5, beta1, beta2)
