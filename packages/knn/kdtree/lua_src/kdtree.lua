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
  local best,argbest=-math.huge,-1
  local max=iterator(result):map(function(x) return -x[2] end):max()
  for i=1,#result do
    local c = get_class_function(result[i][1])
    local logp = -result[i][2] - max
    posteriors[c] = (posteriors[c] and math.logadd(posteriors[c],logp)) or logp
    cte = (cte and math.logadd(cte,logp)) or logp
  end
  for i,v in pairs(posteriors) do
    local logp = v - cte
    posteriors[i] = logp
    if logp > best then best,argbest=logp,i end
  end
  return posteriors,argbest,best
end

function knn.kdtree.regressionKNN(result, get_target_function)
  local posteriors = {}
  local cte
  for i=1,#result do
    local logp = -result[i][2]
    posteriors[i] = logp
    cte = (cte and math.logadd(cte,logp)) or logp
  end
  local wrap_table = {
    matrix = function(a) return a end,
    matrixComplex = function(a) return a end,
    table = function(a) return matrix(#a, a) end,
    number = function(a) return matrix(1,{a}) end,
  }
  local wrap_func = function(a)
    return assert(wrap_table[type(a)](a),
		  "Incorrect type returned by get_target_function, expected a matrix, a table or a number")
  end
  local logp = posteriors[1] - cte
  posteriors[1] = logp
  local result = wrap_func(get_target_function(result[1][1])):clone():
  scal(math.exp(logp))
  for i=2,#posteriors do
    local logp = posteriors[i] - cte
    posteriors[i] = logp
    local target = wrap_func(get_target_function(result[i][1]))
    result:axpy(math.exp(logp), target)
  end
  return result,posteriors
end
