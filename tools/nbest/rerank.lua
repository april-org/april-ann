weightsfilename = arg[1]
nbestfilename   = arg[2]
features        = arg[3]
out_reranked    = arg[4] or "reranked.nbest.out.gz"
out_feats       = arg[5] or nil

local featsf = nil
if out_feats then featsf = io.open(out_feats, "w") end
local g = io.open(out_reranked, "w")

-- leemos los pesos
weights = string.tokenize(io.open(weightsfilename, "r"):read("*l"))
for i=1,#weights do weights[i] = tonumber(weights[i]) end

if features ~= nil then
  features = io.open(features)
end

-- leemos las nbest y hacemos el rescoring
local currentn   = 0
local best       = nil
local best_score = nil
local f          = io.open(nbestfilename)
local reranked   = {}
if features then features:read("*l") end
for line in f:lines() do
  local n,line,feats = string.match(line, "(.*)|||(.*)|||(.*)|||")
  n = tonumber(n)
  if features then
    if n ~= currentn then
      features:read("*l")
      features:read("*l")
    end
    feats = features:read("*l")
  end
  if n ~= currentn and best then
    print(table.concat(string.tokenize(best), " "))
    table.sort(reranked, function(a,b) return a[3] > b[3] end)
    if featsf then
      fprintf(featsf, "FEATURES_TXT_BEGIN_0 %d %d %d",
	      currentn, #reranked, #weights)
      for j=1,#weights do fprintf(featsf, " f_%d", j) end
      fprintf(featsf, "\n")
      for i=1,#reranked do
	fprintf(featsf, "%s\n", reranked[i][2])
      end
      fprintf(featsf, "FEATURES_TXT_END_0\n")
    end
    for i=1,#reranked do
      fprintf(g, "%d|||%s|||%s|||%f\n",
	      currentn,
	      reranked[i][1],
	      reranked[i][2],
	      reranked[i][3])
    end
    reranked = {}
    best_score = nil
    best       = nil
    currentn   = n
  end
  score = 0
  local feats = string.tokenize(feats)
  for k=1,#weights do
    score = score + weights[k]*tonumber(feats[k])
  end
  if not best_score or score > best_score then
    best_score = score
    best       = line
  end
  table.insert(reranked, { line, table.concat(feats, " "), score })
end
print(table.concat(string.tokenize(best), " "))
f:close()
table.sort(reranked, function(a,b) return a[3] > b[3] end)
if featsf then
  fprintf(featsf, "FEATURES_TXT_BEGIN_0 %d %d %d",
	  currentn, #reranked, #weights)
  for j=1,#weights do fprintf(featsf, " f_%d", j) end
  fprintf(featsf, "\n")
  for i=1,#reranked do
    fprintf(featsf, "%s\n", reranked[i][2])
  end
  fprintf(featsf, "FEATURES_TXT_END_0\n")
end
for i=1,#reranked do
  fprintf(g, "%d|||%s|||%s|||%f\n",
	  currentn,
	  reranked[i][1],
	  reranked[i][2],
	  reranked[i][3])
end
reranked = {}
g:close()
if featsf then featsf:close() end
