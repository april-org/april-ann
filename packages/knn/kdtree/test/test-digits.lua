local dir = string.get_path(arg[0])
local data = ImageIO.read(dir .. "../../../../TEST/digitos/digits.png"):to_grayscale():invert_colors():matrix()
local TOP_PCA = 10
local EPSILON = 0.024226
local KNN     = 2
local DO_PCA  = true

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
  train_data = stats.pca_whitening(train_data,
				   U:clone("row_major"),
				   S:clone("row_major"),
				   EPSILON)
  -- print(stats.pca_threshold(S,0.99))
  train_data = train_data(':',{1,TOP_PCA})
  
  val_data = stats.pca_whitening(val_data,
				 U:clone("row_major"),
				 S:clone("row_major"),
				 EPSILON)
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
  local counts = {}
  local max = 0
  local best_class = 0
  for i=1,#result do
    local c = (result[i][1]-1) % 10
    counts[c] = (counts[c] or 0) + 1
    if counts[c] > max then best_class,max = c,counts[c] end
  end
  local dist=result[1][2]
  -- print(i,best_class,counts[best_class])
  local target_class = (i-1) % 10
  if best_class ~= target_class then errors=errors+1 end
  -- NAIVE CLASSIFIER
  local best_class,min=nil,math.huge
  for j=1,train_data:dim(1) do
    local trainpat = train_data(j,':')
    local dist = (trainpat - pat):pow(2):sum()
    if dist < min then min,best_class=dist,(j-1)%10 end
  end
  if target_class ~= best_class then errors2 = errors2 + 1 end
  assert(math.abs(min - dist)/dist < 1e-05)
end

print(100*errors/val_data:dim(1) .. "%", errors)
print(100*errors2/val_data:dim(1) .. "%", errors2)
