if #arg ~= 3 then
  print("Syntax error!")
  printf("\tExecute as: %s FILENAME NUMCOL1 NUMCOL2\n", arg[0])
  printf("\twhere NUMCOL1 and NUMCOL2 are column indexes of your data\n")
  printf("\tuse - as FILENAME for stdin\n")
  os.exit(1)
end
filename = arg[1]
col1     = tonumber(arg[2])
col2     = tonumber(arg[3])

local f
if filename == "-" then f = io.stdin
else f = io.open(filename, "r") end

local N      = 0
local xy_sum = 0
local x_sum  = 0
local y_sum  = 0
local x2_sum = 0
local y2_sum = 0
for line in f:lines() do
  local t  = string.tokenize(line)
  if string.sub(t[1], 1, 1) ~= "#" then
    local v1 = tonumber(t[col1]) or
      error("Impossible to convert to number col1: " .. t[col1])
    local v2 = tonumber(t[col2]) or
      error("Impossible to convert to number col2: " .. t[col2])
    xy_sum = xy_sum + v1*v2
    x_sum  = x_sum  + v1
    y_sum  = y_sum  + v2
    x2_sum = x2_sum + v1*v1
    y2_sum = y2_sum + v2*v2
    N      = N + 1
  end
end

local sx  = math.sqrt(N * x2_sum - x_sum*x_sum)
local sy  = math.sqrt(N * y2_sum - y_sum*y_sum)
local rxy = (N * xy_sum - x_sum*y_sum) / ( sx*sy )

-- linear regression fitting: y = alpha + beta * x
local beta   = rxy * sy / sx
local alpha  = y_sum/N - beta * x_sum/N

printf("rxy=    %.2f\nalpha= %.2f\nbeta=  %.2f\n", rxy, alpha, beta)
