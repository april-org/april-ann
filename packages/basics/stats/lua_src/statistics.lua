stats = stats or {} -- global environment

-----------------------------------------------------------------------------

stats.mstats = {}

local mop = matrix.op
local limits_float = mathcore.limits.float

-- arithmetic mean of a matrix
stats.mstats.amean =
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
stats.mstats.gmean =
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
stats.mstats.hmean =
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

local mean_var,mean_var_methods = class("stats.mean_var")
stats.mean_var = mean_var -- global environment

april_set_doc(stats.mean_var, {
		class       = "class",
		summary     = "Class to compute mean and variance",
		description ={
		  "This class is designed to compute mean and variance",
		  "by adding a sequence of data values (or tables)",
		}, })

-----------------------------------------------------------------------------

april_set_doc(stats.mean_var, {
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
		  "a data table or matrix, a function which returns statistics",
                  "of a sample given an iterator.",
		  "A table with the computation of the post-process function",
		  "for every repetition will be returned.",
		},
		params = {
		  data = "A table with the data",
		  R = "Number of repetitions, recommended minimum of 1000",
		  statistic = {
		    "A function witch receives an iterator and computes",
                    "statistics (k>=1 statistics) over all the iterator results.",
                    "The iterator produces a key which is a row in data",
                    "and a value which is the corresponding row.",
                    "If k>1, statistic must return a table with the desired",
                    "k statistics."
		  },
		  verbose = "True or false",
                  ncores = "Number of cores [optional], by default it is 1",
                  seed = "A random seed [optional], by default it is os.time()",
		},
		outputs = {
		  "A table with the k statistics for every repetition."
		},
})

-- self is needed because of __call metamethod, but it will be ignored
local function boot(self,params)
  local params = get_table_fields(
    {
      data        = { mandatory = true },
      R           = { type_match = "number",   mandatory = true },
      statistic   = { type_match = "function", mandatory = true },
      verbose     = { mandatory = false },
      ncores      = { mandatory = false, type_match = "number", default = 1 },
      seed        = { mandatory = false, type_match = "number", default = os.time() },
    },
    params)
  local data        = params.data
  local repetitions = params.R
  local statistic   = params.statistic
  local ncores      = params.ncores
  local seed        = params.seed
  local get_row,N
  -- prepare N and get_row function depending in the type of data parameter
  if type(data) == "table" then
    N = #data
    get_row = function(i) return data[i] end
  elseif class.is_a(data, matrix) or class.is_a(data, matrixInt32) or class.is_a(data, matrixComplex) then
    N = data:dim(1)
    local row
    get_row = function(i) row=data:select(1,i,row) return row end
  else
    errro("Incorrect type, needs a table, a matrix, matrixInt32 or matrixComplex")
  end
  -- returns an iterator of random samples using rnd random object
  local make_iterator = function(rnd)
    local p=0
    return function()
      if p<N then p=p+1 j=rnd:randInt(1,p) return j,get_row(j) end
    end
  end
  -- resample function executed in parallel using parallel_foreach
  local resample = function(i, id)
    collectgarbage("collect")
    local rnd = random(seed + i - 1)
    local r,_ = statistic(make_iterator(rnd))
    assert(not _, "statistic must return one value (it can be a table")
    assert(type(r) == "number" or type(r) == "table",
           "statistic function must return a number or a table")
    if id == 0 and params.verbose and i % 20 == 0 then
      fprintf(io.stderr, "\r%3.0f%%", i/repetitions*100)
      io.stderr:flush()
    end
    if type(r) ~= "table" then r = {r} end
    return r
  end
  local result = parallel_foreach(ncores, repetitions,
                                  resample, util.to_lua_string)
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
    local N = #data
    assert(index > 0 and index <= N)
    local med_conf_size = N*(1.0 - confidence)*0.5
    local a_pos = math.max(1, math.round(med_conf_size))
    local b_pos = math.min(N, math.round(N - med_conf_size))
    local aux = iterator(ipairs(data)):select(2):field(index):table()
    table.sort(aux)
    return aux[a_pos],aux[b_pos]
  end

stats.boot.percentil =
  april_doc{
    class = "function",
    summary = "Returns a percentil value",
    description= {
      "This function returns a percentil value",
      "given the result of stats.boot function and the confidence value.",
      "It could compute the percentil over a slice of the table.",
    },
    params = {
      "The result of stats.boot function.",
      "The percentil [optional], by default it is 0.5. It can be a table of several percentils",
      "The statistic index for which you want compute the percentil [optional], by default it is 1",
    },
    outputs = {
      "The percentil value",
      "Another pecentil value",
      "..."
    },
  } ..
  -- returns the percentil
  function(data, percentil, index)
    local percentil,index  = percentil or 0.95, index or 1
    if type(percentil) ~= "table" then percentil = { percentil } end
    local aux = iterator(ipairs(data)):select(2):field(index):table()
    table.sort(aux)
    local result_tbl = {}
    for _,v in ipairs(percentil) do
      assert(v > 0 and v < 1,
             "Incorrect percentil value, it must be in range (0,1)")
      local N = #data
      assert(index > 0 and index <= N)
      local pos = N*v
      local pos_floor,pos_ceil,result = math.floor(pos),math.ceil(pos)
      local ratio = pos - pos_floor
      local result = aux[pos_floor]*(1-ratio) + aux[pos_ceil]*ratio
      result_tbl[#result_tbl + 1] = result
    end
    return table.unpack(result_tbl)
  end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

local pearson,pearson_methods = class("stats.correlation.pearson")
get_table_from_dotted_string("stats.correlation", true)
stats.correlation.pearson = pearson

function pearson:constructor(x,y)
  self.mean_var_x  = stats.mean_var()
  self.mean_var_y  = stats.mean_var()
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
    return stats.dist.binomial(matrix.col_major(1,{1}),p)
  else
    return stats.dist.binomial(1,p)
  end
end

