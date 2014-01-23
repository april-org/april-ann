knn = knn or {}
knn.kdtree = knn.kdtree or {}
function knn.kdtree.classifyKNN(result, get_class_function)
  local best_class,max,counts = nil,0,{}
  for i=1,#result do
    local c = get_class_function(result[i][1])
    counts[c] = (counts[c] or 0) + 1
    if counts[c] > max then best_class,max = c,counts[c] end
  end
  return best_class
end
