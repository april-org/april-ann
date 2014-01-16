if #arg ~= 5 then
  print("Syntax error!")
  printf("\t%s [FILENAME | -] NUMCOL1 NUMCOL2 SEED CONF\n", arg[0])
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
reps     = 100

if conf > 0.5  then reps = 1000 end
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

local points = iterator(f:lines()):
map(string.tokenize):
-- remove commentaries
filter(function(t) return string.sub(t[1],1,1) ~= "#" end):
-- filter null values
map(function(t) return zero_if_null(t[col1]),zero_if_null(t[col2]) end):
-- check numbers
map(function(a,b)
      local a =
	tonumber(a) or error("Impossible to convert to number col1: " .. a)
      local b =
	tonumber(b) or error("Impossible to convert to number col2: " .. b)
      return {a,b}
    end):
-- table conversion
table()

local N   = #points
local rnd = random(seed)

fprintf(io.stderr, "%d repetitions, %d points\n", reps, N)

local data = stats.bootstrap_resampling{
  verbose         = true,
  population_size = N,
  repetitions     = reps,
  sampling        = function() return table.unpack(points[rnd:randInt(1,N)]) end,
  initial         = function() return stats.correlation.pearson() end,
  reducer         = function(acc,x,y) return acc:add(x,y) end,
  postprocess     = function(acc) return acc:compute() end,
}

table.sort(data, function(a,b) return a < b end)
local rxy1,rxy2 = stats.confidence_interval(data, conf)

local alpha,beta = util.linear_least_squares(points)

printf("rxy= % .4f +- % .4f [% .4f, % .4f]\n",
       (rxy1 + rxy2)/2, math.abs(rxy1-rxy2)*0.5, rxy1, rxy2)
printf("y=   % .4f + % .4f x\n", alpha, beta)
