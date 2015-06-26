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

do
  -- two-sided test, checks if both curves have a different AUC
  -- http://cran.r-project.org/web/packages/pROC/pROC.pdf
  metrics.roc.test =
    function(r1, r2, params)
      assert(r1.data:dim(1) == r2.data:dim(1), "Incompatible ROC curves")
      local result
      do
        local params = params or {}
        local r1_data = r1.data
        local r2_data = r2.data
        local t_data  = r1_data:select(2,2):contiguous()
        local p1_data = r1_data:select(2,1):contiguous()
        local p2_data = r2_data:select(2,1):contiguous()
        assert(t_data == r2_data:select(2,2), "Different response in both curves")
        result = stats.boot{
          seed = params.seed,
          random = params.random,
          size = r1.data:dim(1),
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
      local s = stats.std(result)
      local D = math.abs( (r1:compute_area() - r2:compute_area())/s )
      return 2*(stats.pnorm(-D):exp()[1]),D
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
