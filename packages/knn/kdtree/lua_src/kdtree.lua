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

function knn.kdtree.posteriorKNN(result, get_class_function)
  local posteriors = {}
  local cte
  for i=1,#result do
    local c = get_class_function(result[i][1])
    local logp = -result[i][2]
    posteriors[c] = (posteriors[c] and math.logadd(posteriors[c],logp)) or logp
    cte = (cte and math.logadd(cte,logp)) or logp
  end
  for i,v in pairs(posteriors) do posteriors[i] = v - cte end
  return posteriors
end
