-- COMPUTES BOX AND WHISKERS, PLUS MEAN AND VARIANCE
datafilename = arg[1]
column       = tonumber(arg[2])
if datafilename == "-" then datafilename = nil end
data = {}
local sum  = 0
local sum2 = 0
for line in io.uncommented_lines(datafilename) do
  local t = string.tokenize(line)
  local v = tonumber(t[column])
  table.insert(data, v)
  sum  = sum  + v
  sum2 = sum2 + v*v
end
table.sort(data)
local q0 = 1
local q1 = math.round(#data/4)
local q2 = math.round(#data/2)
local q3 = math.round(#data*3/4)
local q4 = #data
local mean     = sum/#data
local variance = (sum2 - sum*sum/#data)/(#data - 1)
print(data[q0], data[q1], data[q2], data[q3], data[q4], mean, variance)
