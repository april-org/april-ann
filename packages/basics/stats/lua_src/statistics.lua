stats = stats or {} -- global environment
stats.running = stats.running or {}

april_set_doc(stats.running,{
                class = "namespace",
                summary = "Table with running statistics classes",
})

-----------------------------------------------------------------------------

local mop = matrix.op
local sdiag = matrix.sparse.diag

-- x must be a 2D matrix
local function center(x,mu)
  local x_dim = x:dim()
  local N = x_dim[1]
  if #x_dim == 1 then
    mu = mu or x:sum()/N
    return x - mu, mu
  else
    mu = mu or x:sum(1):scal(1/N)
    return matrix.ext.broadcast(bind(x.axpy, nil, -1.0), x, mu), mu
  end
end

stats.standardize =
  april_doc{
    class = "function",
    summary = "Standardize data to have zero-mean one-variance",
    description = "Data is ordered by rows, features by columns.",
    params = { "A 2D matrix", "A table with center and/or scale matrices [optional]"},
    outputs = { "Another new allocated matrix", "Center matrix", "Scale matrix" },
  } ..
  function(x, params)
    local params = params or {}
    assert(#x:dim() == 2, "Needs a 2D matrix")
    local N = x:dim(1)
    local mu,sigma = params.center,params.scale
    if not sigma then
      local sigma2,mup = stats.var(x,1)
      mu = mu or mup
      sigma = sigma2:sqrt()
    elseif not mu then
      mu = stats.amean(x,1)
    end
    local x = matrix.ext.broadcast(bind(x.axpy, nil, -1.0), x, mu)
    x = matrix.ext.broadcast(x.cmul, x, 1/sigma)
    return x,mu,sigma
  end

stats.center =
  april_doc{
    class = "function",
    summary = "Centers data by rows, computing mean of every column",
    description = "Data is ordered by rows, features by columns.",
    params = { "A 2D matrix", "Center matrix" },
    outputs = { "Another new allocated matrix", "The center matrix" },
  } ..
  function(x,mu)
    assert(#x:dim() == 2, "Needs a 2D matrix")
    return center(x,mu)
  end

stats.var =
  april_doc{
    class = "function",
    summary = "Computes variance over a dimension",
    params = { "A matrix",
               "A dimension number [optional].", },
    outputs = {
      "A new allocated matrix or a number if not dim given",
      "The mean used to center the data"
    },
  } ..
  function(x,dim)
    local mean = stats.amean(x,dim)
    local x,x_row,sz = x:clone()
    if dim then
      sz = x:dim(dim)
      for i=1,sz do x_row=x:select(dim,i,x_row):axpy(-1.0, mean) end
    else
      x:scalar_add(-mean)
      sz = x:size()
    end
    return x:pow(2):sum(dim)/(sz-1),mean
  end

stats.std =
  april_doc{
    class = "function",
    summary = "Computes standard deviation over a dimension",
    params = { "A matrix",
               "A dimension number [optional].", },
    outputs = {
      "A new allocated matrix or a number if not dim given",
      "The mean used to center the data",
    },
  } ..
  function(x,dim)
    local result,center = stats.var(x,dim)
    if dim then result:sqrt()
    else result = math.sqrt(result)
    end
    return result,center
  end

stats.cov =
  april_doc{
    class = "function",
    summary = "Compute covariance matrix of two matrices.",
    description = "Data is ordered by rows, features by columns.",
    params = {
      "A 2D matrix or a vector (x)",
      "Another 2D matrix or a vector (y)",
      "An [optional] table with 'centered' boolean, 'true_mean' boolean",
    },
    outputs = { "Covariance matrix" }
  } ..
  april_doc{
    class = "function",
    summary = "Compute covariance matrix.",
    description = "Data is ordered by rows, features by columns.",
    params = {
      "A 2D matrix or a vector (x)",
      "An [optional] table with 'centered' boolean, 'true_mean' boolean",
    },
    outputs = { "Covariance matrix",
                "The x center vector if not centered flag [optional]",
                "The y center vector if not centered flag [optional]" }
  } ..
  function(x,...)
    local y,params = ...
    if type(y) == "table" or not y then y,params = x,y end
    collectgarbage("collect")
    assert(class.is_a(x,matrix) and class.is_a(y,matrix),
           "Needs at least two matrix arguments")
    local params = get_table_fields(
      {
        centered = { type_match = "boolean", default = nil },
        true_mean = { type_match = "boolean", default =nil },
      }, params)
    assert(not params.true_mean or params.centered,
           "true_mean=true is mandatory of centered=true")
    local x_dim,y_dim = x:dim(),y:dim()
    assert((#x_dim <= 2) and (#y_dim <= 2), "Needs 2D matrices or vectors")
    assert(x_dim[1] == y_dim[1] and x_dim[2] == y_dim[2],
           "Require same shape matrices")
    local mu_x,mu_y
    local N,M = table.unpack(x_dim)
    if not params.centered then
      local oldx = x
      x,mu_x = center(x)
      if rawequal(xold,y) then y,mu_y = x,mu_x else y,mu_y = center(y) end
    end
    local sz = N-1
    if params.true_mean then sz = N end
    return (x:transpose() * y):scal(1/sz):rewrap(M or 1,M or 1),mu_x,mu_y
  end

stats.cor =
  april_doc{
    class = "function",
    summary = "Compute correlation matrix of two matrices.",
    description = "Data is ordered by rows, features by columns.",
    params = {
      "A 2D matrix or a vector (x)",
      "Another 2D matrix or a vector (y)",
      "An [optional] table with 'centered' boolean",
    },
    outputs = { "Correlation matrix" }
  } ..
  april_doc{
    class = "function",
    summary = "Compute correlation matrix.",
    description = "Data is ordered by rows, features by columns.",
    params = {
      "A 2D matrix or a vector (x)",
      "An [optional] table with 'centered' boolean",
    },
    outputs = { "Correlation matrix",
                "The x center vector if not centered flag [optional]",
                "The y center vector if not centered flag [optional]" }
  } ..
  function(x,...)
    local y,params = ...
    if type(y) == "table" or not y then y,params = x,y end
    local params = params or {}
    local mu_x,mu_y
    if not params.centered then
      local xold = x
      x,mu_x = center(x)
      if rawequal(xold,y) then y,mu_y = x,mu_x else y,mu_y = center(y) end
    end
    local function cstd(m) return sdiag((m^2):sum(1):scal(1/(m:dim(1)-1)):sqrt():div(1):squeeze()) end
    local sigma = stats.cov(x,y,{ centered=true })
    local sx = cstd(x)
    local sy = rawequal(x,y) and sx or cstd(y)
    return sx * sigma * sy,mu_x,mu_y
  end

stats.acf =
  april_doc{
    class = "function",
    summary = "Compute auto-correlation of one or more series.",
    description = "Data is ordered by rows, series by columns.",
    params = {
      "A 2D matrix or a vector (x)",
      { "An [optional] table with 'lag_max' number,",
        "'lag_step' number, 'lag_start' number,",
        "'cor' function (one of stats.cor [default], stats.cov)." },
    },
    outputs = { "A matrix with auto-correlation of the series",
                "A matrixInt32 with lag values" },
  } ..
  function(x,params)
    assert(class.is_a(x, matrix), "Needs a matrix argument")
    local x_dim = x:dim()
    assert(x_dim[1] > 1, "Needs two or more rows")
    assert(#x_dim <= 2, "Requires 2D matrix or a vector")
    local params = get_table_fields(
      {
        lag_max = { type_match = "number", default = x_dim[1]-2 },
        lag_step = { type_match = "number", default = 1 },
        lag_start = { type_match = "number", default = 1 },
        cor = { type_match = "function", default = stats.cor },
      }, params)
    local lag_start,lag_max,lag_step = params.lag_start,params.lag_max,params.lag_step
    if #x_dim == 1 then x = x:rewrap(x:size(),1) end
    local N,M = x_dim[1],x:dim(2)
    local result = matrix(math.floor((lag_max + 1 - lag_start) / lag_step), M)
    local acf_func = params.cor
    for j=1,M do
      local i=1
      for lag = lag_start, lag_max, lag_step do
        local a,b = x({1,N-lag},j), x({lag+1,N},j)
        local y = acf_func(a,b)
        result[{i,j}] = y
        i=i+1
      end
    end
    local lags = matrixInt32(iterator(range(lag_start,lag_max,lag_step)):table())
    return result,lags
  end

-- arithmetic mean of a matrix
stats.amean =
  april_doc{
    class = "function",
    summary = "Computes the arithmetic mean over a given dimension",
    params = {
      "A matrix",
      "A dimension number [optional]",
    },
    outputs = {
      "A matrix if given a dimension, a number otherwise",
    },
  } ..
  function(m, D)
    local r = m:sum(D)
    if D then
      return r:scal(1/m:dim(D))
    else
      return r/m:size()
    end
  end

-- geometric mean of a matrix with positive elements
stats.gmean =
  april_doc{
    class = "function",
    summary = "Computes the geometric mean over a given dimension",
    params = {
      "A matrix with positive elements",
      "A dimension number [optional]",
    },
    outputs = {
      "A matrix if given a dimension, a number otherwise",
    },
  } ..
  function(m, D)
    local r = mop.log(m):sum(D)
    if D then
      return r:scal(1.0/m:dim(D)):exp()
    else
      return math.exp(r / m:size())
    end
  end

-- harmonic mean of a matrix with non-zero elements
stats.hmean =
  april_doc{
    class = "function",
    summary = "Computes the harmonic mean over a given dimension",
    params = {
      "A matrix with non-zero elements",
      "A dimension number [optional]",
    },
    outputs = {
      "A matrix if given a dimension, a number otherwise",
    },
  } ..
  function(m, D)
    local r = (1 / m):sum(D)
    if D then
      return r:div( m:dim(D) )
    else
      return m:size() / r
    end
  end

-----------------------------------------------------------------------------

local mean_var,mean_var_methods = class("stats.running.mean_var")
stats.running.mean_var = mean_var -- global environment

april_set_doc(stats.running.mean_var, {
		class       = "class",
		summary     = "Class to compute mean and variance",
		description ={
		  "This class is designed to compute mean and variance",
		  "by adding a sequence of data values (or tables)",
		}, })

-----------------------------------------------------------------------------

april_set_doc(stats.running.mean_var, {
		class = "method", summary = "Constructor",
		description ={
		  "Constructor of a mean_var object",
		},
		params = {
		  "A number [optional]. If given, the assumed_mean approach",
		  "will be followed.",
		},
		outputs = { "A mean_var object" }, })

function mean_var:constructor()
  self:clear()
end

-----------------------------------------------------------------------------

mean_var_methods.clear =
  april_doc{
    class = "method",
    summary = "Re-initializes the object"
  } ..
  function(self)
    self.old_m = 0
    self.old_s = 0
    self.new_m = 0
    self.new_s = 0
    self.N     = 0
    return self
  end

-----------------------------------------------------------------------------

mean_var_methods.add =
  april_doc{
    class = "method", summary = "Adds one value",
    params = {
      "A number",
    },
    outputs = { "The caller mean_var object (itself)" },
  } ..
  april_doc{
    class = "method", summary = "Adds a sequence of values",
    params = {
      "A Lua table (as array of numbers)",
    },
    outputs = { "The caller mean_var object (itself)" },
  } ..
  april_doc{
    class = "method",
    summary = "Adds a sequence of values from an iterator function",
    params = {
      "An iterator function",
    },
    outputs = { "The caller mean_var object (itself)" },
  } ..
  function (self, ...)
    local arg = { ... }
    local v = arg[1]
    if type(v) == "table" then
      return self:add(ipairs(v))
    elseif type(v) == "function" then
      local f,s,v = table.unpack(arg)
      local tmp = table.pack(f(s,v))
      while tmp[1] ~= nil do
        v = tmp[1]
        if #tmp > 1 then table.remove(tmp,1) end
        for _,aux in ipairs(tmp) do self:add(aux) end
        tmp = table.pack(f(s,v))
      end
    elseif type(v) == "number" then
      self.N = self.N + 1
      -- see Knuth TAOCP vol 2, 3rd edition, page 232
      if self.N == 1 then
        self.old_m,self.new_m = v,v
        self.old_s = 0.0
      else
        local old_diff = (v - self.old_m)
        self.new_m = self.old_m + old_diff/self.N
        self.new_s = self.old_s + old_diff*(v - self.new_m)
        -- setup for next iteration
        self.old_m = self.new_m
        self.old_s = self.new_s
      end
    else
      error("Incorrect type="..type(v)..". Expected number, table or function")
    end
    return self
  end

-----------------------------------------------------------------------------

mean_var_methods.size =
  april_doc{
    class = "method",
    summary = "Return the number of elements added",
    outputs = { "The number of elements added" },
  } ..
  function(self)
    return self.N
  end

-----------------------------------------------------------------------------

mean_var_methods.compute =
  april_doc{
    class = "method",
    summary = "Computes mean and variance of given values",
    outputs = {
      "A number, the mean of the data",
      "A number, the variance of the data",
    },
  } ..
  function(self)
    return self.new_m,self.new_s/(self.N-1)
  end

--------------------
-- Confusion Matrix
-- -----------------
local confus_matrix,confus_matrix_methods = class("stats.confusion_matrix")
stats.confusion_matrix = confus_matrix -- global environment

april_set_doc(stats.confusion_matrix, {
                class       = "class",
                summary     = "class for computing confusion matrix and classification metrics",
                description ={
                  "This class is designed to store a confusion matrix and compute main metrics for classification stats",
                },
})

april_set_doc(stats.confusion_matrix, {
                class ="method",
                summary     = "Constructor of confusion matrix.",
                description ={
                  "This class is designed to store a confusion matrix and compute main metrics for classification stats",
                },
                params = {
                  "A number of classes [mandatory].",
                  "A table of size num_classes, with the elements on the set.",
                  outputs = {"A confusion_matrix object"}

                }
})
function confus_matrix:constructor(num_classes, class_dict)
  local confusion = {}
  for i = 1, num_classes do
    local t = {}
    for j = 1, num_classes do
      table.insert(t, 0)
    end
    table.insert(confusion, t)
  end


  if (class_dict) then
    --assert(#class_dict == num_classes, "The map table doesn't have the exact size")
    map_dict = class_dict

    --for i, v in ipairs(map_table) do
    --  map_dict[v] = i
    --end

  end

  self.num_classes = num_classes
  self.confusion = confusion
  self.hits = 0
  self.misses = 0
  self.samples = 0
  -- FIXME: IS NOT POSSIBLE USE MAP DICT AS NIL
  self.map_dict = map_dict or false
end

confus_matrix_methods.clone =
  april_doc{
    class ="method",
    summary     = "Clone onstructor of confusion matrix.",
    description ={
      "This class is designed to store a confusion matrix and compute main metrics for classification stats",
    },
    params = {
    }
  } ..
  function(self)
    
    local obj = table.deep_copy(self)

    return class_instance(obj, stats.confusion_matrix)
  end

confus_matrix_methods.reset =
  april_doc{
    class = "method", summary = "Reset to 0 all the counters",
  } ..
  function(self)
    for i = 1, self.num_classes do
      local t = {}
      for j = 1, self.num_classes do
        self.confusion[i][j] = 0
      end
    end
    self.hits = 0
    self.misses = 0
    self.samples = 0
  end

function confus_matrix_methods:checkType(clase)
  return type(clase) == "number" and clase >= 1 and clase <= self.num_classes or false
end

---------------------------------------------
function confus_matrix_methods:addSample(pred, gt)

  if self.map_dict then

    pred = map_dict[pred]
    gt   = map_dict[gt]
  end
  if not self:checkType(pred) or not self:checkType(gt) then
    printf("Error %f %f, %d\n", pred, gt, self.samples)
    return
      --error("The class is not correct")
  end

  if gt == pred then
    self.hits = self.hits + 1
  else
    self.misses = self.misses + 1
  end
  self.samples = self.samples + 1

  self.confusion[gt][pred] = self.confusion[gt][pred] + 1
end

------------------------------------------------

confus_matrix_methods.printConfusionRaw =
  april_doc{
    class = "method", summary = "Print the counters for each class",
  } ..
  function(self)
    for i,v in ipairs(self.confusion) do
      print(table.concat(v, "\t"))
    end
  end

confus_matrix_methods.printConfusion =
  april_doc{
    class = "method", summary = "Print the counters for each class and PR and RC",
    params = { "A num_classes string table [optional] with the tags of each class",}
  } ..
  function(self, tags)

    local total_pred = {}


    printf("\t|\t Predicted ")
    for i = 1, self.num_classes do
      printf("\t\t")
    end

    printf("|\n")
    printf("______\t|")
    for i = 1, self.num_classes do
      printf("\t___\t")
    end

    printf("\t___\t\t|\n")
    for i,v in ipairs(self.confusion) do

      local tag = i
      if tags then
        tag = tags[i]
      end
      printf("%s\t|\t", tag)
      
      local recall, hits, total = self:getRecall(i)
      printf("%s\t|\t %d/%d %0.4f\t|\n", table.concat(v, "\t|\t"), hits, total, recall)
    end
    printf("______\t|")
    for i = 1, self.num_classes do
      printf("\t___\t")
    end

    printf("\t___\t|\n")
    printf("\t\t|")
    for i = 1, self.num_classes do
      printf("\t%0.4f\t|", self:getPrecision(i))
    end

    local acc, hits, total = self:getAccuracy()
    printf("\t%d/%d %0.4f\t|\n", hits, total, acc)
  end

confus_matrix_methods.tostring =
  function(self)

    local total_pred = {}

    local t = {}

    table.insert(t, "\t|\t Predicted ")

    for i = 1, self.num_classes do
      table.insert(t, ("\t\t"))
    end

    table.insert(t,"|\n")
    table.insert(t, "______\t|")
    for i = 1, self.num_classes do
      table.insert(t, "\t___\t")
    end

    table.insert(t, "\t___\t\t|\n")
    for i,v in ipairs(self.confusion) do

      local tag = i
      table.insert(t, string.format("%s\t|\t", tag))
      
      local recall, hits, total = self:getRecall(i)
      table.insert(t, string.format("%s\t|\t %d/%d %0.4f\t|\n", table.concat(v, "\t|\t"), hits, total, recall))
    end
    table.insert(t, "______\t|")
    for i = 1, self.num_classes do
      table.insert(t, "\t___\t")
    end

    table.insert(t, "\t___\t|\n")
    table.insert(t, "\t\t|")
    for i = 1, self.num_classes do
      table.insert(t, string.format("\t%0.4f\t|", self:getPrecision(i)))
    end

    local acc, hits, total = self:getAccuracy()
    table.insert(t, string.format("\t%d/%d %0.4f\t|\n", hits, total, acc))

    return table.concat(t,"")
  end
function confus_matrix_methods:printInf()
  
  printf("Samples %d, hits = %d, misses = %d (%0.4f)\n", self.samples, self.hits, self.misses, self.misses/self.total)
  for i = 1, self.num_classes do
    
    print("Predicted %d", i)
    local total = 0
    for j = 1, self.num_classes do
      total = total + self.confusion[i][j]
    end

    for j = 1, self.num_classes do
      printf(" - class %d %d/%d (%0.4f)", j, self.confusion[i][j], total, self.confusion[i][j]/total)
    end
    print()
  end
end

--------------------------------------------
-- Datasets and Tables Iterators
--------------------------------------------
function stats.confusion_matrix.twoTablesIterator(table_pred, table_gt)
  local i = 0
  local n = #table_pred
  return function()
    i = i+1
    if i <= n then return table_pred[i],table_gt[i] end
  end

end
----------------------------------------------------------------
function stats.confusion_matrix.oneTableIterator(typeTable)

  local i = 0
  local n = #typeTable
  return function()
    i = i+1
    if i <= n then return typeTable[i][1], typeTable[i][2] end
  end

end
--------------------------------------------------------------
function stats.confusion_matrix.oneDatasetIterator(typeDataset)
  local i = 0
  local n = typeDataset:numPatterns()

  return function()
    i = i+1

    if i <= n then return typeDataset:getPattern(i)[1], typeDataset:getPattern(i)[2] end
  end
end

function stats.confusion_matrix.twoDatasetsIterator(predDs, gtDs)
  local i = 0
  assert(predDs:numPatterns() == gtDs:numPatterns(), "Datasets doesn't have the same size")

  local n = predDs:numPatterns()

  return function()
    i = i+1
    if i <= n then return predDs:getPattern(i)[1], gtDs:getPattern(i)[1] end
    
  end
end

---------------------------------------------------------------------------------------------------------
confus_matrix_methods.addData =
  april_doc{
    class = "method",
    summary = "Add the info of Predicted and Ground Truth set",
    description = {
      "This class recieves two tables with the predicted class and the",
      "ground truth of that class.",
      "Also it can recieve a iterator function that returns two elements:",
      "predicted sample and groundtruth sample"
    },
    params = {
      "This parameter can be a table of the predicted tags or an iterator function",
      "This parameter is used if the first parameter is the Predicted table, otherwise it should be nil"},
  } ..
  function(self, param1, param2)

    local iterator
    if( type(param1) == 'function') then
      iterator = param1
      assert(param2 == nil)
    elseif (type(param1) == 'dataset') then
      iterator = stats.confusion_matrix.twoDatasetsIterator(param1, param2)
    else
      iterator = stats.confusion_matrix.twoTablesIterator(param1, param2)
      assert(type(param1) == "table" and type(param2) == "table", "The type of the params is not correct")
      assert(#param1, #param2, "The tables does not have the same dimension")
    end


    for pred, gt in iterator do
      self:addSample(pred, gt)
    end
  end

---------------------------------------------------------------
confus_matrix_methods.getError =
  april_doc{
    class = "method", summary = "Return the global classification error (misses/total)",
    outputs = { "The global classification error." }, 
  } ..
  function(self)
    return self.misses/self.samples, self.misses, self.samples
  end

confus_matrix_methods.getWeightedError =
  april_doc{
    class = "method", summary = "Return the classification error weighted by given values",
    params = {"A table of size weight"},
    outputs = { "The global classification error." }, 
  } ..
  function(self, weights)
    
    local totalError = 0.0
    for i,w in ipairs(weights) do
      totalError = totalError+(1-w*self:getRecall(i))
    end

    return totalError
  end

confus_matrix_methods.getAvgError =
  april_doc{
    class = "method", summary = "Return the average error.",
    outputs = { "The average classification error." }, 
  } ..
  function(self, weights)
    local totalError = 0.0
    local w = 1.0/self.num_classes
    local i
    for i = 1, self.num_classes do
      totalError = totalError+(1-self:getRecall(i))
    end

    return totalError*w
  end

confus_matrix_methods.getAccuracy =
  april_doc{
    class = "method", summary = "Return the accuracy (hits/total)",
    outputs = { "The global accuracy." },
  } ..
  function(self)
    return self.hits/self.samples, self.hits, self.samples
  end

--------------------------------------------------------------
function confus_matrix_methods:getConfusionTables()
  return self.confusion
end
------------------------------------------------------------
--
confus_matrix_methods.getPrecision =
  april_doc{
    class = "method", summary = "Return the accuracy (hits/total)",
    params = {"The index of the class for computing the Precision"},
    outputs = { "The selected class Precision." },
  } ..
  function(self, tipo)

    local tp = 0
    local den = 0

    -- Moving by columns
    for i=1, self.num_classes do
      v = self.confusion[i][tipo]
      if i == tipo then
        tp = v

      end
      den = den + v
    end     
    if den == 0 then
      return 0, tp, den
    end
    return tp/den, tp, den
  end

confus_matrix_methods.getRecall =
  april_doc{
    class = "method", summary = "Return the accuracy (hits/total)",
    params = {"The index of the class for computing the Recall"},
    outputs = { "The selected class Recall." },
  } ..
  function(self, tipo)
    
    local tp = 0
    local den = 0

    -- Moving by columns
    for j=1, self.num_classes do
      v = self.confusion[tipo][j]
      if j == tipo then
        tp = v

      end
      den = den + v
    end 

    if den == 0 then
      return 0, tp, den
    end
    return tp/den, tp, den
  end

confus_matrix_methods.getFMeasure =
  april_doc{
    class = "method", summary = "Return the accuracy (hits/total)",
    params = {"The index of the class for computing the Precision"},
    outputs = { "The selected class Precision." },
  } ..
  function(self, tipo, beta)
    local nBeta = beta or 1
    nBeta = nBeta*nBeta
    local PR = self:getRecall(tipo)
    local RC = self:getPrecision(tipo)

    return (1+nBeta)*(PR*RC)/(nBeta*PR+RC)
  end



-------------------------------------------------------
confus_matrix_methods.clearGTClass =
  april_doc{
    class = "method", summary = "Clear the counters of one class",
    description= "This function is useful when you don't want to count determinated class.",
    params = {"The index of the class to be clear"},
  } ..
  function(self, tipo)

    local n_samples = 0
    local hits = 0
    local misses = 0
    -- Moving by columns
    for i=1, self.num_classes do
      n_samples = n_samples + self.confusion[tipo][i]
      if i == tipo then
        hits = self.confusion[tipo][i]
      else
        misses = misses + self.confusion[tipo][i]
      end
      self.confusion[tipo][i] = 0
    end
    
    self.samples = self.samples - n_samples
    self.hits    = self.hits - hits
    self.misses  = self.misses - misses
  end

confus_matrix_methods.clearClass =
  april_doc{
    class = "method", summary = "Clear the counters of one pair classes",
    description= "This function is useful when you don't want to count determinated pair class.",
    params = {"The index of the Ground Truth class.","The index of the predicted class"},
  } ..
  function(self, gt, pred)

    local samples = self.confusion[gt][pred]
    local n_samples = 0
    if gt == pred then
      self.hits = self.hits - samples
    else
      self.misses = self.misses - samples
    end
    self.confusion[gt][pred] = 0
    
    self.samples = self.samples - samples
  end

confus_matrix_methods.clearPredClass =
  april_doc{
    class = "method", summary = "Clear the counters of one class",
    description= "This function is useful when you don't want to count determinated class.",
    params = {"The index of the class to be clear"},
  } ..
  function(self, tipo)
    
    local n_samples = 0
    -- Moving by Rows
    for i=1, self.num_classes do
      n_samples = n_samples + self.confusion[i][tipo]
      self.confusion[i][tipo] = 0
    end
    
    self.samples = self.samples - n_samples
    
  end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

stats.boot = {}
april_set_doc(stats.boot,
	      {
		class = "function",
		summary = "Produces a bootstrap resampling table",
		description= {
		  "This function is useful to compute confidence intervals",
		  "by using bootstrapping technique. The function receives",
		  "the population size and a function which returns statistics",
                  "of a sample.",
		  "A table with the computation of the post-process function",
		  "for every repetition will be returned.",
		},
		params = {
		  size = "Population size, it can be a table with several sizes",
                  resample = "Resample value [optional] by default it is 1",
		  R = "Number of repetitions, recommended minimum of 1000",
		  statistic = {
		    "A function witch receives as many matrixInt32 (with",
                    "sample indices) as given number of population sizes",
                    "and computes statistics (k>=1 statistics) over",
                    "the sample.",
		  },
                  k = "Expected number of returned values in statistic function [optional], by default it is 1",
		  verbose = "True or false",
                  ncores = "Number of cores [optional], by default it is 1",
                  seed = "A random seed [optional]",
                  random = "A random numbers generator [optional]",
                  ["..."] = "Second and beyond parameters are extra arguments for statistic function.",
		},
		outputs = {
		  "A matrix with Rxk, being R repetitions."
		},
})

-- self is needed because of __call metamethod, but it will be ignored
local function boot(self,params,...)
  local params = get_table_fields(
    {
      size        = { mandatory = true, },
      resample    = { mandatory = false, type_match = "number", default = 1 },
      R           = { type_match = "number",   mandatory = true },
      statistic   = { type_match = "function", mandatory = true },
      k           = { type_match = "number", mandatory = false, default = 1 },
      verbose     = { mandatory = false },
      ncores      = { mandatory = false, type_match = "number", default = 1 },
      seed        = { mandatory = false, type_match = "number" },
      random      = { mandatory = false, isa_match  = random },
    },
    params)
  assert(not params.seed or not params.random,
         "Fields 'seed' and 'random' are forbidden together")
  local extra       = table.pack(...)
  local size        = params.size
  local resample    = params.resample
  local repetitions = params.R
  local statistic   = params.statistic
  local ncores      = params.ncores
  local seed        = params.seed
  local rnd         = params.random or random(seed)
  local tsize       = type(size)
  local k           = params.k
  local result      = matrix(repetitions, k):zeros()
  assert(resample > 0.0 and resample <= 1.0, "Incorrect resample value")
  assert(tsize == "number" or tsize == "table",
         "Needs a 'size' field with a number or a table of numbers")
  if tsize == "number" then size = { size } end
  local resamples = {}
  for i=1,#size do
    resamples[i] = math.round(size[i]*resample)
    assert(resamples[i] > 0, "Resample underflow, increase resample value")
  end
  local get_row,N
  -- resample function executed in parallel using parallel_foreach
  local last_i = 0
  local rnd_matrix = function(sz,rsmpls) return matrixInt32(rsmpls):uniform(1,sz,rnd) end
  local tmpname = os.tmpname()
  result:toMMap(tmpname)
  result = matrix.fromMMap(tmpname)
  local resample = function(i, id)
    collectgarbage("collect")
    -- this loop allows to synchronize the random number generator, allowing to
    -- produce the same exact result independently of ncores value
    for j=last_i+1,i-1 do
      for r=1,#size do
        for k=1,resamples[r] do rnd:randInt(size[r]-1) end
      end
    end
    last_i = i
    --
    local sample = iterator.zip(iterator(size),
                                iterator(resamples)):map(rnd_matrix):table()
    local r = table.pack( statistic(multiple_unpack(sample, extra)) )
    april_assert(#r == k,
                 "Unexpected number of returned values in statistic, expected %d, found %d",
                 k, #r)
    result[i]:copy_from_table(r)
    if (not id or id == 0) and params.verbose and i % 20 == 0 then
      fprintf(io.stderr, "\r%3.0f%%", i/repetitions*100)
      io.stderr:flush()
    end
  end
  local ok,msg = xpcall(parallel_foreach, debug.traceback,
                        ncores, repetitions, resample)
  result = result:clone()
  os.remove(tmpname)
  if not ok then error(msg) end
  if params.verbose then fprintf(io.stderr, " done\n") end
  return result
end
setmetatable(stats.boot, { __call = boot })

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

stats.boot.ci =
  april_doc{
    class = "function",
    summary = "Returns the extremes of a confidence interval",
    description= {
      "This function returns the extremes of a confidence interval",
      "given the result of stats.boot function and the confidence value.",
      "It could compute the interval over a slice of the table.",
      "This functions uses stats.boot.percentile() to compute the interval.",
    },
    params = {
      "The result of stats.boot function.",
      "The confidence [optional], by default it is 0.95.",
      "The statistic index for which you want compute the CI [optional], by default it is 1",
    },
    outputs = {
      "The left limit of the interval",
      "The right limit of the interval",
    },
  } ..
  -- returns the extremes of the interval
  function(data, confidence, index)
    local confidence,index  = confidence or 0.95, index or 1
    assert(confidence > 0 and confidence < 1,
           "Incorrect confidence value, it must be in range (0,1)")
    local Pa = (1.0 - confidence)*0.5
    local Pb = 1.0 - Pa
    return stats.boot.percentile(data, { Pa, Pb }, index)
  end

stats.boot.percentile =
  april_doc{
    class = "function",
    summary = "Returns a percentile value from bootstrap output following NIST method",
    description= {
      "This function returns a percentile value",
      "given the result of stats.boot function and the percentile number.",
      "It could compute the percentile over a slice of the table.",
    },
    params = {
      "The result of stats.boot function.",
      "The percentile [optional], by default it is 0.5. It can be a table of several percentiles",
      "The statistic index for which you want compute the percentile [optional], by default it is 1",
    },
    outputs = {
      "The percentile value",
      "Another pecentil value",
      "..."
    },
  } ..
  -- returns the percentile
  function(data, percentile, index)
    if type(percentile) ~= "table" then percentile= { percentile } end
    return stats.percentile(data:select(2, index or 1),
                            table.unpack(percentile))
  end

stats.percentile =
  april_doc{
    class = "function",
    summary = "Returns a percentile value from a matrix (should be a vector)",
    params = {
      "A one-dimensional matrix (a vector).",
      "A percentile",
      "Another percentile",
      "...",
    },
    outputs = {
      "The percentile value",
      "Another percentile value",
      "..."
    },
  } ..
  -- returns the percentile
  function(data, ...)
    local data = data:squeeze()
    assert(data:num_dim() == 1, "Needs a data vector")
    local order = data:order()
    local result_tbl = {}
    for _,v in ipairs(table.pack(...)) do
      assert(v >= 0.0 and v <= 1.0,
             "Incorrect percentile value, it must be in range [0,1]")
      local N = data:size()
      local pos = (N+1)*v
      local pos_floor,pos_ceil,result = math.floor(pos),math.ceil(pos)
      local result
      if pos_floor == 0 then
        result = data[order[1]]
      elseif pos_floor >= N then
        result = data[order[data:size()]]
      else
        local dec = pos - pos_floor
        local a,b = data[order[pos_floor]],data[order[pos_ceil]]
        result = a + dec * (b - a)
      end
      result_tbl[#result_tbl + 1] = result
    end
    return table.unpack(result_tbl)
  end

stats.summary = function(data)
  local data = data:squeeze()
  assert(data:num_dim() == 1, "Needs a data vector")
  local var,mean = stats.var(data)
  local order = data:order()
  local p0,p25,p50,p75,p100 = stats.percentile(data,
                                               0.00, 0.25, 0.50, 0.75, 1.00)
  return setmetatable({
      sd = math.sqrt(var),
      mean = mean,
      p0 = p0,
      p25 = p25,
      p50 = p50,
      p75 = p75,
      p100 = p100,
                      },
    {
      __tostring = function(self)
        local header = iterator{"Mean","SD","Min","Q1","Median","Q3","Max"}:
          map(bind(string.format, "%8s")):concat(" ")
        local data = iterator{self.mean, self.sd,
                              self.p0, self.p25, self.p50,
                              self.p75, self.p100 }:
          map(bind(string.format, "%8.4g")):concat(" ")
        return table.concat({header, data}, "\n")
      end
  })
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

local pearson,pearson_methods = class("stats.running.pearson")
stats.running.pearson = pearson

function pearson:constructor(x,y)
  self.mean_var_x  = stats.running.mean_var()
  self.mean_var_y  = stats.running.mean_var()
  self.xy_sum      = 0
  if x then self:add(x,y) end
end

function pearson_methods:clear()
  self.mean_var_x:clear()
  self.mean_var_y:clear()
  self.xy_sum = 0
  return self
end

function pearson_methods:add(x,y)
  local x,y = x,y
  if not y then
    if type(x) == "table" then x,y = table.unpack(x) end
  end
  assert(x and y, "Needs two values or an array table with two components")
  self.mean_var_x:add(x)
  self.mean_var_y:add(y)
  self.xy_sum = self.xy_sum + x*y
  return self
end

function pearson_methods:compute()
  local N          = self.mean_var_x:size()
  local mu_x,s_x   = self.mean_var_x:compute()
  local mu_y,s_y   = self.mean_var_y:compute()
  local rxy = ( self.xy_sum - N*mu_x*mu_y ) / ( (N-1)*math.sqrt(s_x*s_y) )
  return rxy
end

-------------------------------------------------------------------------------

stats.dist.bernoulli = function(p)
  if class.is_a(p, matrix) then
    return stats.dist.binomial(matrix(1,{1}),p)
  else
    return stats.dist.binomial(1,p)
  end
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

april_set_doc(stats.comb,{
                class = "function", 
                summary = "Computes k-combination",
                params = {
                  "Total number of elements (n)",
                  "How many selected elements (k)",
                },
                outputs = { "A number with (n over k)" },
})

stats.mean_var = make_deprecated_function("stats.mean_var",
                                          "stats.running.mean_var",
                                          stats.running.mean_var)

stats.correlation = {} -- deprecated table
stats.correlation.pearson = make_deprecated_function("stats.correlation.pearson",
                                                     "stats.running.pearson",
                                                     stats.running.pearson)
