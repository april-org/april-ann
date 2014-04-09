featuresfilename     = arg[1]

f = {} for i=2,#arg do f[i-1] = io.open(arg[i], "r") end

-- leemos las features
features     = {}
local n      = 1
local fbegin = "FEATURES_TXT_BEGIN"
local fend   = "FEATURES_TXT_END"
featsf=io.open(featuresfilename)
for line in featsf:lines() do
  if string.sub(line, 1, #fbegin) == fbegin then
    features[n] = {}
    local t = string.tokenize(line)
    printf("%s %s %s %d %s",
	   t[1], t[2], t[3], tonumber(t[4])+#arg-1, table.concat(t, " ", 5, #t))
    for i=1,#f do printf(" FEAT%d", i) end printf("\n")
  elseif string.sub(line, 1, #fend) == fend then
    n = n + 1
    printf("%s", line)
    for i=1,#f do printf(" FEAT%d", i) end printf("\n")
  else
    printf("%s", line)
    for i=1,#f do  printf(" " .. f[i]:read("*l")) end printf("\n")
  end
end
featsf:close()
for i=1,#f do f[i]:close() end
