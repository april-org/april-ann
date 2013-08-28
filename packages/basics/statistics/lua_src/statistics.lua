class("stats.mean_var")

april_set_doc("stats.mean_var", {
		class       = "class",
		summary     = "Class to compute mean and variance",
		description ={
		  "This class is designed to compute mean and variance",
		  "by adding a sequence of data values (or tables)",
		}, })

-----------------------------------------------------------------------------

april_set_doc("stats.mean_var.__call", {
		class = "method", summary = "Constructor",
		description ={
		  "Constructor of a mean_var object",
		},
		params = {
		  "A number [optional]. If given, the assumed_mean approach",
		  "will be followed.",
		},
		outputs = { "A mean_var object" }, })

function stats.mean_var:__call()
  local obj = {
    old_m = 0,
    old_s = 0,
    new_m = 0,
    new_s = 0,
    N     = 0,
  }
  class_instance(obj, self, true)
  return obj
end

-----------------------------------------------------------------------------

april_set_doc("stats.mean_var.clear", {
		class = "method",
		summary = "Re-initializes the object" })

function stats.mean_var:clear()
  self.old_m = 0
  self.old_s = 0
  self.new_m = 0
  self.new_s = 0
  self.N     = 0
end

-----------------------------------------------------------------------------

april_set_doc("stats.mean_var.add", {
		class = "method", summary = "Adds one value",
		params = {
		  "A number",
		},
		outputs = { "The caller mean_var object (itself)" }, })

april_set_doc("stats.mean_var.add", {
		class = "method", summary = "Adds a sequence of values",
		params = {
		  "A Lua table (as array of numbers)",
		},
		outputs = { "The caller mean_var object (itself)" }, })

april_set_doc("stats.mean_var.add", {
		class = "method",
		summary = "Adds a sequence of values from an iterator function",
		params = {
		  "An iterator function",
		},
		outputs = { "The caller mean_var object (itself)" }, })

function stats.mean_var:add(...)
  local arg = { ... }
  local v = arg[1]
  if type(v) == "table" then
    return self:add(ipairs(v))
  elseif type(v) == "function" then
    for key,value in table.unpack(arg) do
      self:add(value or key)
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

april_set_doc("stats.mean_var.size", {
		class = "method",
		summary = "Return the number of elements added",
		outputs = { "The number of elements added" }, })

function stats.mean_var:size()
  return self.N
end

-----------------------------------------------------------------------------

april_set_doc("stats.mean_var.compute", {
		class = "method",
		summary = "Computes mean and variance of given values",
		outputs = {
		  "A number, the mean of the data",
		  "A number, the variance of the data",
		}, })

function stats.mean_var:compute()
  return self.new_m,self.new_s/(self.N-1)
end

--------------------
-- Confusion Matrix
-- -----------------
class("stats.confusion_matrix")


april_set_doc("stats.confusion_matrix", {
    class       = "class",
    summary     = "class for computing confusion matrix and classification metrics",
    description ={
        "This class is designed to store a confusion matrix and compute main metrics for classification stats",
    },
})

april_set_doc("stats.confusion_matrix.__call", {
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
function stats.confusion_matrix:__call(num_classes, class_dict)

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

    local obj = {
        num_classes = num_classes,
        confusion = confusion,
        hits = 0,
        misses = 0,
        samples = 0,
        -- FIXME: IS NOT POSSIBLE USE MAP DICT AS NIL
        map_dict = map_dict or false
    }
    class_instance(obj, self, true)
    return obj
end

april_set_doc("stats.confusion_matrix.clone", {
    class ="method",
    summary     = "Clone onstructor of confusion matrix.",
    description ={
        "This class is designed to store a confusion matrix and compute main metrics for classification stats",
    },
    params = {
    }
})
function stats.confusion_matrix:clone()
    
    local obj = table.deep_copy(self)

    class_instance(obj, self, true)

    return obj
end
april_set_doc("stats.confusion_matrix.reset", {
		class = "method", summary = "Reset to 0 all the counters",
		})
function stats.confusion_matrix:reset()

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

function stats.confusion_matrix:checkType(clase)
    return type(clase) == "number" and clase >= 1 and clase <= self.num_classes or false
end

---------------------------------------------
function stats.confusion_matrix:addSample(pred, gt)

    if self.map_dict then

        pred = map_dict[pred]
        gt   = map_dict[gt]
    end
    if not self:checkType(pred) or not self:checkType(gt) then
        error("The class is not correct")
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


april_set_doc("stats.confusion_matrix.printConfusionRaw", {
		class = "method", summary = "Print the counters for each class",
		})
function stats.confusion_matrix:printConfusionRaw()

    for i,v in ipairs(self.confusion) do
        print(table.concat(v, "\t"))
    end
end

april_set_doc("stats.confusion_matrix.printConfusion", {
		class = "method", summary = "Print the counters for each class and PR and RC",
    params = { "A num_classes string table [optional] with the tags of each class",}
		})
function stats.confusion_matrix:printConfusion(tags)

    printf("\t|\t Predicted ")
    for i = 1, self.num_classes do
        printf("\t\t")
    end

    printf("|\n")
    printf("______\t|")
    for i = 1, self.num_classes do
        printf("\t___\t")
    end

    printf("\t___\t|\n")
    for i,v in ipairs(self.confusion) do

        local tag = i
        if tags then
            tag = tags[i]
        end
        printf("%s\t|\t", tag)

        printf("%s\t|\t %0.4f\t|\n", table.concat(v, "\t|\t"), self:getRecall(i))
    end
    printf("______\t|")
    for i = 1, self.num_classes do
        printf("\t___\t")
    end

    printf("\t___\t|\n")
    printf("\t|")
    for i = 1, self.num_classes do
        printf("\t%0.4f\t|", self:getPrecision(i))
    end
    printf("\t%0.4f\t|\n", self:getError())
end

function stats.confusion_matrix:printInf()
    
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
april_set_doc("stats.confusion_matrix.addData",
{
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
})
function stats.confusion_matrix:addData(param1, param2)

    local iterator
    if( type(param1) == 'function') then
        iterator = param1
        assert(param2 == nil)
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
april_set_doc("stats.confusion_matrix.getError",
{
    class = "method", summary = "Return the global classification error (misses/total)",
    outputs = { "The global classification error." }, 
})
function stats.confusion_matrix:getError()
    return self.misses/self.samples
end

april_set_doc("stats.confusion_matrix.getAccuracy", {
    class = "method", summary = "Return the accuracy (hits/total)",
    outputs = { "The global accuracy." },
})
function stats.confusion_matrix:getAccuracy()
    return self.hits/self.samples
end

--------------------------------------------------------------
function stats.confusion_matrix:getConfusionTables()
    return self.confusion
end
------------------------------------------------------------
--
april_set_doc("stats.confusion_matrix.getPrecision",
{
    class = "method", summary = "Return the accuracy (hits/total)",
    params = {"The index of the class for computing the Precision"},
    outputs = { "The selected class Precision." },
})
function stats.confusion_matrix:getPrecision(tipo)

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
        return 0
    end
    return tp/den
end

april_set_doc("stats.confusion_matrix.getRecall",
{
    class = "method", summary = "Return the accuracy (hits/total)",
    params = {"The index of the class for computing the Recall"},
    outputs = { "The selected class Recall." },
})
function stats.confusion_matrix:getRecall(tipo)

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
        return 0
    end
    return tp/den
end

april_set_doc("stats.confusion_matrix.getFMeasure",
{
    class = "method", summary = "Return the accuracy (hits/total)",
    params = {"The index of the class for computing the Precision"},
    outputs = { "The selected class Precision." },
})
function stats.confusion_matrix:getFMeasure(tipo, beta)
    local nBeta = beta or 1
    nBeta = nBeta*nBeta
    local PR = self:getRecall(tipo)
    local RC = self:getPrecision(tipo)

    return (1+nBeta)*(PR*RC)/(nBeta*PR+RC)
end



-------------------------------------------------------
april_set_doc("stats.confusion_matrix.clearGTClass",
{
    class = "method", summary = "Clear the counters of one class",
    description= "This function is useful when you don't want to count determinated class.",
    params = {"The index of the class to be clear"},
})
function stats.confusion_matrix:clearGTClass(tipo)

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

april_set_doc("stats.confusion_matrix.clearClass",
{
    class = "method", summary = "Clear the counters of one pair classes",
    description= "This function is useful when you don't want to count determinated pair class.",
    params = {"The index of the Ground Truth class.","The index of the predicted class"},
})
function stats.confusion_matrix:clearClass(gt, pred)

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
april_set_doc("stats.confusion_matrix.clearPredClass",
{
    class = "method", summary = "Clear the counters of one class",
    description= "This function is useful when you don't want to count determinated class.",
    params = {"The index of the class to be clear"},
})
function stats.confusion_matrix:clearPredClass(tipo)

    local n_samples = 0
    -- Moving by Rows
    for i=1, self.num_classes do
        n_samples = n_samples + self.confusion[i][tipo]
        self.confusion[i][tipo] = 0
    end
    
    self.samples = self.samples - n_samples
   
end

