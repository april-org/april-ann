metrics = metrics or {}
--

local function check_matrix(m)
  assert(class.is_a(m,matrix), "Needs a matrix as argument")
  local dim = m:dim()
  assert( #dim == 1 or
            (#dim == 2 and
               (dim[1] == 1 or dim[2] == 1)),
          "Needs a row or column vector" )
  return #dim == 1 and m:contiguous():rewrap(m:size(),1) or m
end

--

local roc,roc_methods = class("metrics.roc")
metrics.roc = roc -- global environment

metrics.roc.test = nil -- implementation in next do...end block
do
  local function compute_delong_V_vectors(normal_idx, abnormal_idx, data)
    local Na     = abnormal_idx:size()
    local Nn     = normal_idx:size()
    local V10    = matrix(Na)
    local V01    = matrix(Nn)
    local normal_data   = data:index(1, normal_idx)
    local abnormal_data = data:index(1, abnormal_idx)
    for i=1,Na do
      local x  = abnormal_data[i]
      local lt = normal_data:lt(x):count_ones()
      local gt = normal_data:gt(x):count_ones()
      local eq = Nn - lt - gt
      V10[i] = lt + 0.5 * eq
    end
    V10:scal(1.0/Nn)
    for i=1,Nn do
      local y  = normal_data[i]
      local lt = abnormal_data:lt(y):count_ones()
      local gt = abnormal_data:gt(y):count_ones()
      local eq = Nn - lt - gt
      V01[i] = gt + 0.5 * eq
    end
    V01:scal(1.0/Na)
    --
    return V10,V01
  end
  
  local function compute_delong_covariance(params, r1, r2,
                                           t_data, p1_data, p2_data)
    local AUC2 = r1:compute_area()
    local AUC1 = r2:compute_area()
    local normal_idx   = t_data:eq(0.0):to_index()
    local abnormal_idx = t_data:eq(1.0):to_index()
    local V10,V01 = {},{}
    V10[1],V01[1] = compute_delong_V_vectors(normal_idx,
                                             abnormal_idx, p1_data)
    V10[2],V01[2] = compute_delong_V_vectors(normal_idx,
                                             abnormal_idx, p2_data)
    V10[1]:scalar_add(-AUC1)
    V01[1]:scalar_add(-AUC1)
    V10[2]:scalar_add(-AUC2)
    V01[2]:scalar_add(-AUC2)
    local Na  = abnormal_idx:size()
    local Nn  = normal_idx:size()
    local S10 = matrix(2,2)
    local S01 = matrix(2,2)
    for i=1,2 do
      local row10 = S10[i]
      local row01 = S01[i]
      for j=1,2 do
        row10[j] = V10[i]:dot(V10[j]) / (Na-1)
        row01[j] = V01[i]:dot(V01[j]) / (Nn-1)
      end
    end
    local S = S10/Na + S01/Nn
    return S
  end
  
  local function not_stratified_bootstrap(params, t_data, p1_data, p2_data)
    return stats.boot{
      seed = params.seed,
      random = params.random,
      size = t_data:size(),
      R = params.R or 1000,
      k = 1,
      ncores = params.ncores,
      verbose = params.verbose,
      statistic = function(idx)
        local p1_data = p1_data:index(1, idx)
        local p2_data = p2_data:index(1, idx)
        local t_data  = t_data:index(1,idx)
        local r1 = metrics.roc(p1_data, t_data)
        local r2 = metrics.roc(p2_data, t_data)
        return r1:compute_area() - r2:compute_area()
      end,
    }
  end

  local function stratified_bootstrap(params, t_data, p1_data, p2_data)
    local t0_idx,t1_idx
    do
      local gt_0 = t_data:gt(0)
      t1_idx = gt_0:to_index()
      t0_idx = gt_0:complement():to_index()
    end
    return stats.boot{
      seed = params.seed,
      random = params.random,
      size = { t0_idx:size(), t1_idx:size() },
      R = params.R or 1000,
      k = 1,
      ncores = params.ncores,
      verbose = params.verbose,
      statistic = function(idx0,idx1)
        local idx = matrix.join(1, { t0_idx:index(1,idx0),
                                     t1_idx:index(1,idx1) })
        local p1_data = p1_data:index(1, idx)
        local p2_data = p2_data:index(1, idx)
        local t_data  = t_data:index(1,idx)
        local r1 = metrics.roc(p1_data, t_data)
        local r2 = metrics.roc(p2_data, t_data)
        return r1:compute_area() - r2:compute_area()
      end,
    }
  end
  
  local hyp_test = stats.hypothesis_test
  
  -- two-sided test, checks if both curves have a different AUC
  -- http://cran.r-project.org/web/packages/pROC/pROC.pdf
  metrics.roc.test =
    function(r1, r2, params)
      assert(r1.data:dim(1) == r2.data:dim(1), "Incompatible ROC curves")
      local params  = params or {}
      local method  = params.method or "bootstrap" params.method = nil
      local stratified = (params.stratified==nil and true) or params.stratified
      params.stratified = nil
      local r1_data = r1.data
      local r2_data = r2.data
      local t_data  = r1_data:select(2,2):contiguous()
      local p1_data = r1_data:select(2,1):contiguous()
      local p2_data = r2_data:select(2,1):contiguous()
      assert(t_data == r2_data:select(2,2), "Different response in both curves")
      if method == "bootstrap" then
        local result
        if not stratified then
          result = not_stratified_bootstrap(params, t_data, p1_data, p2_data)
        else
          result = stratified_bootstrap(params, t_data, p1_data, p2_data)
        end
        -- normal distribution test
        local sd = stats.std(result)
        local mean = r1:compute_area() - r2:compute_area()
        local D = var
        return hyp_test(D, stats.dist.normal())
      elseif method == "delong" then
        local S = compute_delong_covariance(params, r1, r2,
                                            t_data, p1_data, p2_data)
        local contrast = matrix(2,2,{1,-1,-1,1})
        local var = S:cmul(contrast):sum()
        local mean = r1:compute_area() - r2:compute_area()
        local z = mean/math.sqrt(var)
        return hyp_test(z, stats.dist.normal(), stats.dist.normal(mean,var))
      else
        error("Unknown method " .. method)
      end
    end
end

april_set_doc(roc,
              {
                class="class",
                summary="ROC curve class, for one-class problems",
})

roc.constructor =
  april_doc{
    class = "method",
    summary = "Constructor without arguments",
  } ..
  april_doc{
    class = "method",
    summary = "Constructor with matrix arguments",
    params = {
      { "A matrix with classifier output probabilities in natural scale", },
      { "A matrix with target class: 0 or 1", },
    },
  } ..
  function(self,outputs,targets)
    self.P = 0
    self.N = 0
    if outputs or targets then self:add(outputs,targets) end
  end

roc_methods.add =
  april_doc{
    class = "method",
    summary = "Adds an output and target class for ROC computation",
    params = {
      { "A matrix with classifier output probabilities in natural scale", },
      { "A matrix with target class: 0 or 1", },
    },
  } ..
  function(self,outputs,targets)
    local outputs = check_matrix(outputs)
    local targets = check_matrix(targets)
    local data,P,N = self.data,0,0
    if not data then
      self.data = matrix.join(2, outputs, targets)
    else
      self.data = matrix.join(1, data, matrix.join(2, outputs, targets))
    end
    local ones = targets:gt(0.5):count_ones()
    self.P = self.P + ones
    self.N = self.N + targets:size() - ones
  end

roc_methods.compute_curve =
  april_doc{
    class = "method",
    summary = "Computes the ROC curve with all added data",
    outputs = {
      {"A matrix with N rows and 4 columns, the first column is FPR, second",
       "is TPR, the third is the threshold, and the last is the true value"},
    },
  } ..
  function(self)
    local data = self.data
    local P = self.P
    local N = self.N
    local data = data:index(1,data:select(2,1):order())
    local result = {}
    local TP,FP = 0,0
    local prev_th,j = -math.huge,0
    for i,row in matrix.ext.iterate(data, 1, -1) do
      if row[1] ~= prev_th then
        local TPR,FPR = TP/P,FP/N
	table.insert(result, FPR)    -- 1
	table.insert(result, TPR)    -- 2
	table.insert(result, row[1]) -- 3
	table.insert(result, row[2]) -- 4
        j=j+1
        prev_th = row[1]
      end
      if row[2] > 0.5 then
        TP = TP + 1
      else
        FP = FP + 1
      end
    end
    table.insert(result, 1)
    table.insert(result, 1)
    table.insert(result, -1)
    table.insert(result, -1)
    return matrix(j+1,4,result)
  end

roc_methods.compute_area =
  april_doc{
    class="method",
    summary="Computes the Area Under the Curve",
    outputs={ "The area" },
  } ..
  function(self)
    local cv = self:compute_curve()
    local area = 0
    for i = 2,cv:dim(1) do
      area = area +
        (cv:get(i,1) - cv:get(i-1,1)) *
        (cv:get(i,2) + cv:get(i-1,2))*0.5
    end
    return area
  end

roc_methods.reset =
  april_doc{
    class="method",
    summary="Resets all the intermediate data",
  } ..
  function(self)
    self.data = nil
    self.P    = 0
    self.N    = 0
    collectgarbage("collect")
  end

return roc
