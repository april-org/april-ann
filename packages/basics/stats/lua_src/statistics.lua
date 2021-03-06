local alternatives = { right=true, left=true, ["two-sided"] = true }

stats = stats or {} -- global environment
stats.running = stats.running or {}

april_set_doc(stats.running,{
                class = "namespace",
                summary = "Table with running statistics classes",
})

stats.dist = stats.dist or {}

-------------------------------------------------------------------------------

stats.levels = function(m)
  local symbols = {}
  local inv_symbols = {}
  for i=1,#m do
    local x = m[i]
    if not inv_symbols[x] then
      symbols[#symbols+1],inv_symbols[x] = x,true
    end
  end
  table.sort(symbols)
  return symbols
end

stats.hist = function(m, params)
  local params = get_table_fields({
      breaks = { default=13 },
      normalize = { type_match="boolean", default=false },
                                  }, params or {})
  local breaks = params.breaks
  local result = matrix(breaks+1, 5)
  local min    = m:min()
  local max    = m:max()
  local diff   = max - min
  assert(diff > 0, "Unable to compute histogram for given matrix")
  local x      = result:select(2,1):linspace(min, max)
  local result = result[{ {1,breaks}, ':' }]
  local x      = result:select(2,1)
  local dx     = math.abs( x[2] - x[1] )
  local x2     = result:select(2,2):copy(x):scalar_add(0.5*dx)
  local y      = result:select(2,3)
  local z      = result:select(2,4)
  local z2     = result:select(2,5)
  local aux = m:clone():
    scalar_add(-min):
    scal(1.0/diff * breaks):
    floor():
    scalar_add(1.0):
    clamp(-math.huge,breaks):
    flatten()
  local y_aux = iterator.zeros():take(breaks):table()
  for i=1,#aux do local b=aux[i] y_aux[b] = y_aux[b] + 1 end
  y:copy_from_table(y_aux)
  local ratio = 1 / (y:sum() * dx)
  z:copy(y):scal( ratio )
  z2:copy(y):scal(1/m:size())
  local df = data_frame{ data=result, columns={"bin","key","count","density","ratio"} }
  return df
end

stats.ihist = function(m, params)
  local params = get_table_fields({
      symbols = { type_match="table", default={} },
      normalize = { type_match="boolean", default=false },
                                  }, params or {})
  local symbols = params.symbols
  if #symbols == 0 then symbols = stats.levels(m) end
  local inv_symbols = table.invert(symbols)
  assert(#symbols > 0, "Unable to compute histogram for given matrix")
  local x      = matrix(#symbols):linspace()
  local y      = matrix(#symbols)
  local z      = matrix(#symbols)
  local y_aux  = iterator.zeros():take(#symbols):table()
  for i=1,#m do
    local v=m[i]
    local b = inv_symbols[v]
    y_aux[b] = y_aux[b] + 1
  end
  y:copy_from_table(y_aux)
  z:copy(y):scal(1/m:size())
  return data_frame{ data={ bin=x, key=symbols, count=y, ratio=z },
                     columns={ "bin","key","count","ratio" } }
end

-------------------------------------------------------------------------------

do
  local function bisect(dist, x, y, log_p, EPSILON, MAX)
    local i,m = 0,nil
    repeat
      m = 0.5 * (x + y)
      local aux = dist:logcdf(matrix(1,1,{m}))[1]
      if (y-x < EPSILON) or math.abs(aux - log_p) < EPSILON then
        return m
      elseif aux > log_p then
        y = m
      else
        x = m
      end
      i=i+1
    until i == MAX
    fprintf(io.stderr, "Warning!!! bisect maximum number of iterations\n")
    return m
  end

  stats.dist.quantile = function(dist, p, EPSILON, MAX)
    assert(dist:size() == 1, "Only implemented for univariate distributions")
    assert(0 < p and p < 1.0, "The probabilbity should be in range (0,1)")
    local log_p = math.log(p)
    local a = -1.0
    local b =  1.0
    while true do
      local cdfa = dist:logcdf(matrix(1,1,{a}))[1]
      local cdfb = dist:logcdf(matrix(1,1,{b}))[1]
      if cdfa < log_p and cdfb > log_p then break end
      if cdfa > log_p then a = a*10 end
      if cdfb < log_p then b = b*10 end
    end
    return bisect(dist, a, b, log_p, EPSILON or 1e-08, MAX or 10000)
  end
end

local tbl,methods = class("stats.hypothesis_test")
april_set_doc(tbl, {
                class = "class",
                summary = "Result of hypotheses test",
})
stats.hypothesis_test = tbl
tbl.constructor =
  april_doc{
    class = "method",
    summary = "Constructor given a pivot and the H0 stats.dist instance",
  } ..
  function(self, pivot, h0_dist, true_dist)
    assert(pivot and h0_dist and true_dist,
           "Needs a pivot, H0 distribution and true distribution")
    self.P = matrix(1,1,{pivot})
    self.h0_dist   = h0_dist
    self.true_dist = true_dist
  end

methods.pvalue =
  april_doc{
    class = "method",
    summary = "Returns the p-value of the pivot for the H0 distribution",
  } ..
  function(self, alternative)
    local alternative = alternative or "two-sided"
    april_assert(alternatives[alternative],
                 "Unknown alternative value %s", alternative)
    local P = self.P
    local h0_dist = self.h0_dist
    local a,b = 0.0,0.0
    local two_sided = (alternative=="two-sided")
    if two_sided or alternative == "left" then
      a = math.exp( h0_dist:logcdf(-P)[1] )
      if two_sided and a > 0.5 then a = 1.0 - a end
    end
    if two_sided or alternative == "right" then
      b = 1.0 - math.exp( h0_dist:logcdf(P)[1] )
      if two_sided and b > 0.5 then b = 1.0 - b end
    end
    return a + b
  end
methods.pivot =
  april_doc{
    class = "method",
    summary = "Returns the test pivot",
  } ..
function(self) return self.P:get(1,1) end

methods.ci =
  april_doc {
    class = "method",
    summary = "Returns the confidence interval of the true distribution",
  } ..
  function(self, confidence, alternative)
    local alternative = alternative or "two-sided"
    april_assert(alternatives[alternative],
                 "Unknown alternative value %s", alternative)
    local true_dist = self.true_dist
    local alpha = 1.0 - (confidence or 0.95)
    local two_sided = (alternative=="two-sided")
    if two_sided then alpha = alpha * 0.5 end
    local a = -math.huge
    local b =  math.huge
    if two_sided or alternative == "right" then
      a = stats.dist.quantile(true_dist, alpha)
    end
    if two_sided or alternative == "left" then
      b = stats.dist.quantile(true_dist, 1.0 - alpha)
    end
    return a,b
  end

class.extend_metamethod(tbl, "__tostring", function(self)
                          local a,b = self:ci(0.95)
                          local tbl = {
                            "pivot= ", tostring(self:pivot()),
                            "\np-value= ", tostring(self:pvalue()),
                            "\nCI(95%)= [", tostring(a), ", ", tostring(b), "]" }
                          return table.concat(tbl)
end)

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
    if mu:size() == 1 then
      return x - mu, mu
    else
      return matrix.ext.broadcast(bind(x.axpy, nil, -1.0), x, mu), mu
    end
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
    if mu:size() == 1 then
      x = x - mu:get(table.unpack(mu:dim()))
    else
      x = matrix.ext.broadcast(bind(x.axpy, nil, -1.0), x, mu)
    end
    if sigma:size() == 1 then
      x = x / sigma:get(table.unpack(sigma:dim()))
    else
      x = matrix.ext.broadcast(x.cmul, x, 1/sigma)
    end
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

    if type(param1) == 'matrix' then
        return self:addDataMatrix(param1, param2)
    end
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

--[[  function confus_matrix_methods.addDataMatrix(self, m1, m2)
      assert(#m1:dim() == 1, "M1 does not have one dimension")
      assert(#m2:dim() == 1, "M2 does not have dimension")
      assert(m1:dim(1) == m2:dim(1), "Dimensions did not match")
      -- Compute hits
      --
      self.samples = m1:dim(1)
      self.hits = m1:eq(m2):count_ones()

      for gt = 1,self.num_classes do
          for pred = 1, self.num_classes do
            b_m1 = m1:eq(gt)
            b_m2 = m2:eq(pred)
            self.confusion[gt][pred] = b_m1:land(b_m2):count_ones()
          end
      end
  end
]]
  function confus_matrix_methods.addDataMatrix(self, target, pred)
      for i=1,#target do
          local t_c = target[i]
          local p_c = pred[i]
          local row = self.confusion[t_c]
          row[p_c] = row[p_c] + 1
      end
  end


  function confus_matrix_methods.combine(self, conf1)
      assert(conf1.num_classes == self.num_classes)
      self.samples = self.samples + conf1.samples
      self.hits = self.hits + conf1.hits
      for gt = 1,self.num_classes do
          for pred = 1, self.num_classes do
              self.confusion[gt][pred] = self.confusion[gt][pred] + conf1.confusion[gt][pred]
          end
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
          return 1, tp, den
      end
      return tp/den, tp, den
  end

  confus_matrix_methods.getMultiPrecision =
  april_doc{
      class = "method", summary = "Return the precision of joined classes",
      params = {"The indexes of the joined classes"},
      outputs = { "The selected classes Precision." },
  } ..
  function(self, cls)

      local tp = 0
      local den = 0

      -- Moving by columns
      for i=1, self.num_classes do
          local iscorrect = false
          local values = 0
          for t, tipo in pairs(cls) do
              v = self.confusion[i][tipo]
              values = values + v
              if i == tipo then
                  iscorrect = true

              end
          end

          if iscorrect then
              tp = tp + values
          end
          den = den + values
      end     
      if den == 0 then
          return 1, tp, den
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
          return 1, tp, den
      end
      return tp/den, tp, den
  end

  confus_matrix_methods.getMultiRecall =
  april_doc{
      class = "method", summary = "Return the accuracy (hits/total)",
      params = {"Index of the joined class for the Recall"},
      outputs = { "The selected class Recall." },
  } ..
  function(self, cls)

      local tp = 0
      local den = 0
      print ("joining")
      -- Moving by columns
      for j=1, self.num_classes do
          local iscorrect = false
          local values = 0
          for t, tipo in ipairs(cls) do
              local v = self.confusion[tipo][j]
              values = values + v

              if tipo == j then
                  iscorrect = true
              end
          end
          if iscorrect then
              tp = tp + values
          end
          den = den + values
      end
      if den == 0 then
          return 1, tp, den
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

  stats.perm = {}

  local perm =
  function(self,params,...)
      local params = get_table_fields(
      {
          samples    = { mandatory = true, type_match="table" },
          R           = { type_match = "number",   mandatory = true },
          statistic   = { type_match = "function", mandatory = true },
          k           = { type_match = "number", mandatory = false, default = 1 },
          verbose     = { mandatory = false },
          ncores      = { mandatory = false, type_match = "number", default = 1 },
          seed        = { mandatory = false, type_match = "number" },
          random      = { mandatory = false, isa_match  = random },
          paired      = { mandatory = false, type_match = "boolean" },
      },
      params)
      assert(not params.seed or not params.random,
      "Fields 'seed' and 'random' are forbidden together")
      local extra       = table.pack(...)
      local paired      = params.paired
      local samples     = params.samples
      local repetitions = params.R
      local statistic   = params.statistic
      local ncores      = params.ncores
      local seed        = params.seed
      local rnd         = params.random or random(seed)
      local k           = params.k
      local result      = matrix.MMapped(repetitions, k):zeros()
      local joined      = matrix.join(1, samples)
      local joined_size = joined:size()
      local sizes       = iterator(samples):call("size"):table()
      local get_row,N
      -- resample function executed in parallel using parallel_foreach
      local last_i = 0
      local rnd_matrix
      assert(not paired, "Paired option is not fully implemented")
      if paired then
          for i=2,#sizes do assert(sizes[i-1] == sizes[i], "Found different sizes in paired test") end
          local M = sizes[1]
          local sub_indices = matrix.ext.repmat(matrixInt32(M):linspace(), #sizes)
          local indices = {}
          for i=1,#sizes do for j=1,M do indices[#indices+1] = i-1 end end
          rnd_matrix = function()
              local shuf = rnd:shuffle(indices)
              local m = matrixInt32(shuf):scal(M):axpy(1.0, sub_indices)
              return m
          end
      else
          local indices = iterator.range(joined_size):table()
          rnd_matrix = function() return matrixInt32(joined:size(), rnd:shuffle(indices)) end
      end
      local permute = function(i, id)
          collectgarbage("collect")
          -- this loop allows to synchronize the random number generator, allowing to
          -- produce the same exact result independently of ncores value
          for j=last_i+1,i-1 do rnd_matrix() end
          last_i = i
          --
          local new_samples_idx = rnd_matrix()
          local new_samples_joined = joined:index(1,new_samples_idx)
          local new_samples = {}
          local acc = 1
          for i=1,#sizes do
              new_samples[i] = new_samples_joined[{ {acc, acc+sizes[i]-1} }]
              acc = acc + sizes[i]
          end
          local r = table.pack( statistic(multiple_unpack(new_samples, extra)) )
          april_assert(#r == k,
          "Unexpected number of returned values in statistic, expected %d, found %d",
          k, #r)
          result[i]:copy_from_table(r)
          if (not id or id == 1) and params.verbose and i % 20 == 0 then
              fprintf(io.stderr, "\r%3.0f%%", i/repetitions*100)
              io.stderr:flush()
          end
      end
      local ok,msg = xpcall(parallel_foreach, debug.traceback,
      ncores, repetitions, permute)
      result = result:clone()
      if not ok then error(msg) end
      if params.verbose then fprintf(io.stderr, " done\n") end
      return result
  end
  setmetatable(stats.perm, { __call = perm })

  stats.perm.pvalue =
  function(perm_result, observed, idx, alternative)
      assert(not alternative or alternative == "two-sided")
      local ge_observed = perm_result:select(2, idx or 1):lt(observed):count_zeros()
      local pvalue = ge_observed / perm_result:dim(1)
      return (pvalue<0.5) and pvalue or (1.0 - pvalue)
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
      local result      = matrix.MMapped(repetitions, k):zeros()
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
          if (not id or id == 1) and params.verbose and i % 20 == 0 then
              fprintf(io.stderr, "\r%3.0f%%", i/repetitions*100)
              io.stderr:flush()
          end
      end
      local ok,msg = xpcall(parallel_foreach, debug.traceback,
      ncores, repetitions, resample)
      result = result:clone()
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
  function(data, confidence, index, alternative)
      local alternative = alternative or "two-sided"
      april_assert(alternatives[alternative],
      "Unknown alternative value %s", alternative)
      local confidence,index  = confidence or 0.95, index or 1
      assert(confidence > 0 and confidence < 1,
      "Incorrect confidence value, it must be in range (0,1)")
      local Pa = 0.0
      local Pb = 1.0
      if alternative == "two-sided" then
          Pa = (1.0 - confidence)*0.5
          Pb = 1.0 - Pa
      elseif alternative == "left" then
          Pb = confidence
      else
          Pa = 1.0 - confidence
      end
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

  -- An Empirical Investigation of Statistical Significance in NLP.
  -- Taylor Berg-Kirkpatrick David Burkett Dan Klein.
  -- http://www.cs.berkeley.edu/~tberg/papers/emnlp2012.pdf
  stats.boot.pvalue =
  function(data, pivot, index, params)
      local params = get_table_fields(
      {
          h0 = { mandatory = false, default = 0.0, type_match = "number" },
          alternative = { mandatory = false, default = "two-sided", type_match = "string" },
      },
      params or {})
      local h0  = params.h0
      local p50 = stats.boot.percentile(data, 0.50, index)
      local alternative = params.alternative
      april_assert(alternatives[alternative],
      "Unknown alternative value %s", alternative)
      local data = data:select(2, index or 1)
      local a,b  = 0.0,0.0
      local two_sided = (alternative=="two-sided")
      if two_sided or alternative == "left" then
          if two_sided and pivot > 0.0 then
              a = data:lt(-pivot + p50 + h0):count_ones()
          else
              a = data:lt(pivot + p50 + h0):count_ones()
          end
      end
      if two_sided or alternative == "right" then
          if two_sided and pivot < 0.0 then
              b = data:gt(-pivot + p50 + h0):count_ones()
          else
              b = data:gt(pivot + p50 + h0):count_ones()
          end
      end
      local pvalue = (a + b + 1) / (data:size() + 1)
      return pvalue
  end

  ----------------------------------------------------------------------------

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
  do
      local std_norm = stats.dist.normal()
      local check = function(x)
          local tt = type(x)
          if tt == "number" then
              x = matrix{x}
          elseif tt == "table" then
              x = matrix(x)
          end
          x = x:contiguous():rewrap(x:size(), 1)
          return x,tt
      end
      --
      stats.dnorm = function(x, mean, sd)
          local x,tt = check(x)
          if mean then x:axpy(-1.0, mean) end
          if sd then x:scal(1/sd) end
          return std_norm:logpdf(x):exp()
      end
      --
      stats.pnorm = function(x, mean, sd)
          local x,tt = check(x)
          if mean then x:axpy(-1.0, mean) end
          if sd then x:scal(1/sd) end
          return std_norm:logcdf(x):exp()
      end
      --
      stats.qnorm = function(x, mean, sd)
          return stats.dist.quantile(std_norm, x)*(sd or 1) + (mean or 0)
      end
  end

  -------------------------------------------------------------------------------

  class.extend(stats.dist, "quantile", stats.dist.quantile)

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
