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

function stats.mean_var:__call(assumed_mean)
  local obj = {
    assumed_mean = assumed_mean or 0,
    accum_sum    = 0,
    accum_sum2   = 0,
    N            = 0,
  }
  class_instance(obj, self, true)
  return obj
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
		summary = "Adds a value or values from a function call",
		params = {
		  "A Lua function",
		},
		outputs = { "The caller mean_var object (itself)" }, })

function stats.mean_var:add(v)
  if type(v) == "table" then
    for _,vp in ipairs(v) do return self:add(vp) end
  elseif type(v) == "function" then
    local vp = v()
    return self:add(vp)
  elseif type(v) == "number" then
    local vi = v - self.assumed_mean
    self.accum_sum  = self.accum_sum + vi
    self.accum_sum2 = self.accum_sum2 + vi*vi
    self.N          = self.N + 1
  else
    error("Incorrect type="..type(v)..". Expected number, table or function")
  end
  return self
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
  local mean,var
  local aux_mean = self.accum_sum / self.N
  mean = self.assumed_mean + aux_mean
  var  = (self.accum_sum2 - self.N * aux_mean * aux_mean) / (self.N - 1)
  return mean,var
end

--------------------
-- Confusion Matrix
-- -----------------
class("stats.confusion_matrix")


april_set_doc("stats.confunsion_matrix", {
    class       = "class",
    summary     = "class for computing confusion matrix and classification metrics",
    description ={
        "This class is designed to store a confusion matrix and compute main metrics for classification stats",
    }, })

    ----------------------------------------
    function stats.confusion_matrix:__call(num_classes, map_table)

        local confusion = {}
        for i = 1, num_classes do
            local t = {}
            for j = 1, num_classes do
                table.insert(t, 0)
            end
            table.insert(confusion, t)
        end
        

        if (map_table) then
          assert(#map_table == num_classes, "The map table doesn't have the exact size")
          map_dict = {}

          for i, v in ipairs(map_table) do
            map_dict[v] = i
          end
          
        end

        local obj = {
            num_classes = num_classes,
            confusion = confusion,
            hits = 0,
            misses = 0,
            samples = 0,
            -- FIXME: IS NOT POSSIBLE USE MAP DICT AS NIL
            map_dict = map_dict or -1

        }
        class_instance(obj, self, true)
        return obj
    end

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

        if self.map_dict ~= -1 then
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

    function stats.confusion_matrix:printConfusionRaw()
        
        for i,v in ipairs(self.confusion) do
            print(table.concat(v, "\t"))
        end
    end

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
    function stats.confusion_matrix:getError()
        return self.misses/self.samples
    end

    function stats.confusion_matrix:getAccuracy()
        return self.hits/self.samples
    end

    --------------------------------------------------------------
    function stats.confusion_matrix:getConfusionTables()
        return self.confusion
    end
    ------------------------------------------------------------
    --
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

    function stats.confusion_matrix:getFMeasure(tipo, beta)
        local nBeta = beta or 1
        nBeta = nBeta*nBeta
        local PR = self:getRecall(tipo)
        local RC = self:getPrecision(tipo)

        return (1+nBeta)*(PR*RC)/(nBeta*PR+RC)
    end

