local dir = string.get_path(arg[0])
local data = ImageIO.read(dir .. "../../../../TEST/digitos/digits.png"):to_grayscale():invert_colors():matrix()
local TOP_PCA = 10
local EPSILON = 1e-05
local PCA_EPSILON = 0.024226
local KNN     = 7
local DO_PCA  = false

local train_data = matrix(800,256)
local val_data   = matrix(200,256)
local i=1
for sw in data:sliding_window{ size={16,16},
			       step={16,16},
			       numSteps={80,10} }:iterate() do
  train_data(i,':'):rewrap(16,16):copy(sw)
  i=i+1
end
local i=1
for sw in data:sliding_window{ size={16,16},
			       step={16,16},
			       numSteps={20,10},
			       offset={1280,0} }:iterate() do
  val_data(i,':'):rewrap(16,16):copy(sw)
  i=i+1
end

if DO_PCA then
  local U,S = stats.pca(train_data:clone("col_major"))
  U=U:clone("row_major")
  S=S:clone("row_major")
  train_data = stats.pca_whitening(train_data,U,S,PCA_EPSILON)
  -- print(stats.pca_threshold(S,0.99))
  train_data = train_data(':',{1,TOP_PCA})
  
  val_data = stats.pca_whitening(val_data,U,S,PCA_EPSILON)
  val_data = val_data(':',{1,TOP_PCA})
end

--
local D = train_data:dim(2)
local kdt = knn.kdtree(D,random(213824))
kdt:push(train_data)
kdt:build()

local errors=0
local errors2=0
for i=1,val_data:dim(1) do
  local pat = val_data(i,':')
  local result = kdt:searchKNN(KNN,pat)
  -- MAJORITY VOTE
  local best_class = knn.kdtree.classifyKNN(result,
					    function(id) return (id-1)%10 end)
  local dist=result[1][2]
  -- print(i,best_class,counts[best_class])
  local target_class = (i-1) % 10
  -- POSTERIORS
  local posteriors = knn.kdtree.posteriorKNN(result,
					     function(id) return (id-1)%10 end)
  local best_class2 = iterator(pairs(posteriors)):map(table.pack):
  reduce(function(acc,b)
           return (b[2]>acc[2] and { b[1], b[2] }) or acc
         end, { -1, -math.huge })[1]
  -- CLASSIFY
  if best_class2 ~= target_class then errors=errors+1 end
  --
  -- iterator(ipairs(posteriors)):select(2):map(table.unpack):apply(print)
  -- NAIVE CLASSIFIER
  local naive_result = {}
  for j=1,train_data:dim(1) do
    local trainpat = train_data(j,':')
    local dist = (trainpat - pat):pow(2):sum()
    table.insert(naive_result, {j, dist})
  end
  table.sort(naive_result, function(a,b) return a[2] < b[2] end)
  naive_result = table.slice(naive_result, 1, KNN)
  for i=1,KNN do
    assert(math.abs(result[i][2] - naive_result[i][2])/result[i][2] < EPSILON)
  end
end

print(100*errors/val_data:dim(1) .. "%", errors)
