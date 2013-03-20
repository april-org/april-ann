if #arg < 3 then
  print("Syntax error!")
  printf("\t%s FILENAME|-  NUMCOL  LAG1  [LAG2  [LAG3  [...]]]\n", arg[0])
  os.exit(1)
end
filename = arg[1]
col      = tonumber(arg[2])
lags     = {}
for i=3,#arg do table.insert(lags, tonumber(arg[i])) end

if filename == "-" then filename = nil end

function zero_if_null(v)
  if v == "None" or v == "Undefined" or v == "Unintialized" or v == "Null" then
    v = 0
  end
  return v
end

local data = {}
local sum  = 0
local sum2 = 0
for line in io.uncommented_lines(filename) do
  local t = string.tokenize(line)
  local v = zero_if_null(t[col])
  table.insert(data, v)
  sum  = sum  + v
  sum2 = sum2 + v*v
end

local mean = sum/#data
local var  = (sum2 - sum*sum/#data)/(#data - 1)

printf("mean: %g\nvariance: %g\n\n", mean, var)

for _,lag in ipairs(lags) do
  local i   = 1
  local j   = lag
  local sum = 0
  while j <= #data do
    local x,y = data[i],data[j]
    sum = sum + (x - mean) * (y - mean)
    i, j = i + 1, j + 1
  end
  printf("LAG %d  %g\n", lag, (sum/#data)/var)
end
