feature_pos          = tonumber(arg[1])
featuresfilename     = arg[2]
f                    = io.open(arg[3], "r") -- new feature file
alpha_linear_comb    = tonumber(arg[4] or 0.0) -- alpha for nbest list feature, 1 - alpha for new feature

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
    if not tonumber(t[4]) then fprintf(io.stderr, "%s\n", line) end
    printf("%s %s %s %d %s",
	   t[1], t[2], t[3], t[4], table.concat(t, " ", 5, #t))
    printf("\n")
  elseif string.sub(line, 1, #fend) == fend then
    n = n + 1
    printf("%s", line)
    printf("\n")
  else
    local t = string.tokenize(line)
    t[feature_pos] = (t[feature_pos]*alpha_linear_comb) + ((1-alpha_linear_comb)*tonumber(f:read("*l")))
    printf("%s\n", table.concat(t, " "))
  end
end
featsf:close()
f:close()
