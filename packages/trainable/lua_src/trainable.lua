-------------------------------
-- LOCAL AUXILIARY FUNCTIONS --
-------------------------------

local math = math
local table = table
local string = string
--
local ipairs = ipairs
local pairs = pairs
local assert = assert
--
local type = type
local is_a = class.is_a
local iterator = iterator
local get_table_fields = get_table_fields
local april_assert = april_assert

-----------------------------------------

local MAX_SIZE_WO_COLLECT_GARBAGE=10*1024*1024 -- 10 MB

-----------------------
-- TRAINABLE CLASSES --
-----------------------

-------------------------------------------------------------------------------

--------------------------------
-- DATASET_ITERATOR FUNCTIONS --
--------------------------------

trainable.dataset_pair_iterator =
  april_doc{
    class       = "function",
    summary     = {
      "A function which returns an iterator which",
      "does a dataset pair loop (input_dataset and output_dataset)"
    },
    description ={
      "This class is useful to implement a loop over a dataset.",
      "It implements different loop schemes, in all the cases",
      "returning a bunch_size patterns.",
      "It admits the following traversals: sequential, shuffled,",
      "shuffled with replacement, shuffled with distribution.",
    },
  } ..
  function(t)
    local params = get_table_fields(
      {
        input_dataset  = { mandatory = false, default=nil },
        output_dataset = { mandatory = false, default=nil },
        distribution   = { type_match="table", mandatory = false, default=nil,
                           getter = get_table_fields_ipairs{
                             input_dataset  = { mandatory=true },
                             output_dataset = { mandatory=true },
                             probability    = { type_match="number",
                                                mandatory=true },
                           },
        },
        bunch_size     = { type_match = "number", mandatory = true },
        shuffle        = { isa_match  = random,   mandatory = false, default=nil },
        replacement    = { type_match = "number", mandatory = false, default=nil },
        assert_input_size = { type_match = "number", mandatory = false, default=0 },
        assert_output_size = { type_match = "number", mandatory = false, default=0 },
      }, t)
    -- ERROR CHECKING
    assert(params.input_dataset ~= not params.output_dataset,
           "input_dataset and output_dataset fields are mandatory together")
    assert(not params.input_dataset or not params.distribution,
           "input_dataset/output_dataset fields are forbidden with distribution")
    assert(params.assert_input_size ~= not params.assert_output_size,
           "assert_input_size and assert_output_size are mandatory together")
    if params.input_dataset then
      params.datasets = { params.input_dataset, params.output_dataset }
      params.input_dataset, params.output_dataset = nil
    end
    if params.distribution then
      iterator(ipairs(params.distribution)):
        apply(function(_,data)
	    data.datasets = { data.input_dataset, data.output_dataset }
	    data.input_dataset, data.output_dataset = nil, nil
        end)
    end
    if params.assert_input_size then
      params.assert_pattern_sizes = { params.assert_input_size,
                                      params.assert_output_size }
      params.assert_input_size, params.assert_output_size = nil, nil
    end
    return trainable.dataset_multiple_iterator(params)
  end

----------------------------

trainable.dataset_multiple_iterator =
  april_doc{
    class       = "function",
    summary     = {
      "A function which returns an iterator which",
      "does a dataset loop through multiple datasets"
    },
    description ={
      "This function is useful to implement a loop over a dataset.",
      "It implements different loop schemes, in all the cases",
      "returning a token with bunch_size patterns.",
      "It admits the following traversals: sequential, shuffled,",
      "shuffled with replacement, shuffled with distribution.",
    },
  } ..
  function(t)
    local cgarbage = collectgarbage
    cgarbage("collect")
    local assert = assert
    local table = table
    local iterator = iterator
    local april_assert = april_assert
    --
    local TOO_LARGE_NUMPATTERNS = 1000000
    local params = get_table_fields(
      {
        datasets       = { mandatory = false, default=nil },
        distribution   = { type_match="table", mandatory = false, default=nil,
                           getter = get_table_fields_ipairs{
                             datasets    = { mandatory=true },
                             probability = { type_match="number",
                                             mandatory=true },
                           },
        },
        bunch_size     = { type_match = "number", mandatory = true },
        shuffle        = { isa_match  = random,   mandatory = false, default=nil },
        replacement    = { type_match = "number", mandatory = false, default=nil },
        assert_pattern_sizes = { type_match = "table", mandatory = false,
                                 default={ } },
      }, t)
    -- ERROR CHECKING
    assert(not params.datasets or not params.distribution,
           "datasets field is forbidden with distribution")
    assert(params.datasets or params.distribution,
           "datasets or distribution are needed")
    --
    local bunch_size = params.bunch_size
    --
    local to_dataset_token = function(ds_table)
      local nump
      return iterator(ipairs(ds_table)):
        map(function(_,ds)
            if nump then
              april_assert(nump == ds:numPatterns(),
                           "Incorrect number of patterns, expected %d, found %d",
                           nump, ds:numPatterns())
            else nump = ds:numPatterns()
            end
            return is_a(ds,dataset) and dataset.token.wrapper(ds) or ds
	end):
        table(), nump
    end
    -- TRAINING TABLES
    
    -- for each pattern, index in dataset
    local ds_idx_func
    if params.distribution then
      -- Training with distribution: given a table of datasets the patterns are
      -- sampled following the given apriory probability
      assert(params.shuffle,"shuffle is mandatory with distribution")
      assert(params.replacement,"replacement is mandatory with distribution")
      params.datasets = { }
      local aprioris = {}
      local sizes    = {}
      local sums     = { 0 }
      local num_ds
      --
      params.datasets = iterator(ipairs(params.distribution.datasets)):
      map(function() return dataset.token.union() end):table()
      --
      for i,v in ipairs(params.distribution) do
        local nump
        if num_ds then
          april_assert(num_ds == #v.datasets,
                       "Incorrect number of datasets, expected %d, found %d",
                       num_ds, #v.datasets)
        else num_ds = #v.datasets
        end
        v.datasets = to_dataset_token(v.datasets)
        table.insert(aprioris, v.probability)
        table.insert(sizes, nump)
        table.insert(sums, sums[#sums] + nump)
        --
        iterator(ipairs(params.datasets)):
        apply(function(k,union_ds) union_ds:push_back(v.datasets[k]) end)
      end -- for i,v in ipairs(params.distribution)
      -- generate training tables
      local dice = random.dice(aprioris)
      local i = 1
      ds_idx_func = function()
        if i <= params.replacement then
          i=i+1
          local whichclass=dice:thrown(params.shuffle)
          local idx=params.shuffle:randInt(1,sizes[whichclass])
          return idx+sums[whichclass]
        end
      end
    else -- if params.distribution then ... else
      --
      local num_patterns
      params.datasets,num_patterns = to_dataset_token(params.datasets)
      -- generate training tables depending on training mode (replacement,
      -- shuffled, or sequential)
      if params.replacement then
        assert(params.shuffle,"shuffle is mandatory with replacement")
        local i=1
        ds_idx_func = function()
          if i <= params.replacement then
            i=i+1
            return params.shuffle:randInt(1,num_patterns)
          end
        end
      elseif params.shuffle then
        assert(num_patterns < TOO_LARGE_NUMPATTERNS, 
               "The number of patterns is too large, use a shorter replacement, or non-shuffled training")
        local ds_idx_table = params.shuffle:shuffle(num_patterns)
        local i=0
        ds_idx_func = function()
          i=i+1
          if i<= #ds_idx_table then
            return ds_idx_table[i]
          end
        end
      else
        local i=0
        ds_idx_func = function()
          i=i+1
          if i<= num_patterns then return i end
        end
      end
    end
    -- SANITY CHECK
    iterator(ipairs(params.assert_pattern_sizes)):
      apply(function(k,psize)
	  local ds_psize = params.datasets[k]:patternSize()
	  april_assert(psize == 0 or psize == ds_psize,
		       "Incorrect patternSize at dataset %d, found %d, expected %d",
		       k, ds_psize, psize)
      end)
    --
    -- ITERATOR USING ds_idx_func
    local k=0
    local bunch_indexes = {}
    if #params.datasets > 2 then
      local pattern_size = iterator(ipairs(params.datasets)):select(2):
        call('patternSize'):reduce(math.add, 0)
      local bunch_mb_size = bunch_size * pattern_size * 4
      return function()
        local ds_idx_func = ds_idx_func
        local bunch_indexes = bunch_indexes
        local table = table
        local insert = table.insert
        table.clear(bunch_indexes)
        repeat
          local idx = ds_idx_func()
          table.insert(bunch_indexes, idx)
        until not idx or #bunch_indexes==bunch_size
        -- end condition, return nil
        if #bunch_indexes == 0 then cgarbage("collect") return end
        local data = {}
        for i,v in ipairs(params.datasets) do data[i] = v:getPatternBunch(bunch_indexes) end
        insert(data, bunch_indexes)
        k=k+bunch_mb_size
        if k >= MAX_SIZE_WO_COLLECT_GARBAGE then cgarbage("collect") k=0 end
        return table.unpack(data)
      end
    elseif #params.datasets == 2 then
      local ds1,ds2 = params.datasets[1],params.datasets[2]
      local pattern_size = ds1:patternSize() + ds2:patternSize()
      local bunch_mb_size = bunch_size * pattern_size * 4
      return function()
        local ds_idx_func  = ds_idx_func
        local bunch_indexes = bunch_indexes
        local table = table
        local insert = table.insert
        -- general case
        table.clear(bunch_indexes)
        repeat
          local idx = ds_idx_func()
          insert(bunch_indexes, idx)
        until not idx or #bunch_indexes==bunch_size
        -- end condition, return nil
        if #bunch_indexes == 0 then cgarbage("collect") return end
        local bunch1 = ds1:getPatternBunch(bunch_indexes)
        local bunch2 = ds2:getPatternBunch(bunch_indexes)
        k=k+bunch_mb_size
        if k >= MAX_SIZE_WO_COLLECT_GARBAGE then cgarbage("collect") k=0 end
        return bunch1,bunch2,bunch_indexes
      end
    else -- ( #params.datasets == 1 )
      local ds1 = params.datasets[1]
      local pattern_size = ds1:patternSize()
      local bunch_mb_size = bunch_size * pattern_size * 4
      return function()
        local ds_idx_func  = ds_idx_func
        local bunch_indexes = bunch_indexes
        local table = table
        local insert = table.insert
        -- general case
        table.clear(bunch_indexes)
        repeat
          local idx = ds_idx_func()
          insert(bunch_indexes, idx)
        until not idx or #bunch_indexes==bunch_size
        -- end condition, return nil
        if #bunch_indexes == 0 then cgarbage("collect") return end
        local bunch1 = ds1:getPatternBunch(bunch_indexes)
        k=k+bunch_mb_size
        if k >= MAX_SIZE_WO_COLLECT_GARBAGE then cgarbage("collect") k=0 end
        return bunch1,bunch_indexes
      end
    end
  end

-------------------------------------------------------------------------------

------------------------------------
-- TRAIN_HOLDOUT_VALIDATION CLASS --
------------------------------------

local train_holdout, train_holdout_methods =
  class("trainable.train_holdout_validation", aprilio.lua_serializable)
trainable = trainable or {} -- global environment
trainable.train_holdout_validation = train_holdout -- global environment

april_set_doc(trainable.train_holdout_validation, {
		class       = "class",
		summary     = "Training class using holdout validation",
		description ={
		  "This training class defines a train_func which",
		  "follows a training schedule based on validation error or",
		  "in number of epochs. Method execute receives a function",
		  "which trains one epoch and returns the trainer object,",
		  "the training loss and the validation loss. This method",
		  "returns true in case the training continues, or false if",
		  "the stop criterion is true.",
		}, })

april_set_doc(trainable.train_holdout_validation, {
		class = "method", summary = "Constructor",
		description ={
		  "Constructor of the train_holdout_validation class.",
		},
		params = {
		  epochs_wo_validation = {
		    "Number of epochs from start where the validation is",
		    "ignored [optional], by default it is 0",
		  },
		  min_epochs = "Minimum number of epochs for training [optional]. By default it is 0",
		  max_epochs = "Maximum number of epochs for training",
		  stopping_criterion = {
		    "A predicate function which",
		    "returns true if stopping criterion, false otherwise.",
		    "Some basic criteria are implemented at",
		    "trainable.stopping_criteria table.",
		    "The criterion function is called as",
		    "stopping_criterion({ current_epoch=..., best_epoch=...,",
		    "best_val_error=..., train_error=...,",
		    "validation_error=... }). [optional] By default it is max_epochs criterion.",
		  },
		  tolerance = {
		    "The tolerance>=0 is the minimum relative difference to",
		    "take the current validation loss as the best [optional],",
		    "by default it is 0.",
		  },
		  first_epoch = "The first epoch number [optional]. By default it is 0.",
		},
		outputs = { "Instantiated object" }, })

function train_holdout:constructor(t,saved_state)
  local params = get_table_fields(
    {
      epochs_wo_validation = { mandatory=false, type_match="number", default=0 },
      min_epochs = { mandatory=true, type_match="number", default=0 },
      max_epochs = { mandatory=true, type_match="number" },
      tolerance  = { mandatory=true, type_match="number", default=0 },
      stopping_criterion = { mandatory=true, type_match="function",
			     default = function() return false end },
      first_epoch        = { mandatory=false, type_match="number", default=1 },
    }, t)
  assert(params.tolerance >= 0, "tolerance < 0 is forbidden")
  local saved_state = saved_state or {}
  self.params = params
  self.state  = {
    current_epoch    = saved_state.current_epoch    or 0,
    train_error      = saved_state.train_error      or math.huge,
    validation_error = saved_state.validation_error or math.huge,
    best_epoch       = saved_state.best_epoch       or 0,
    best_val_error   = saved_state.best_val_error   or math.huge,
    best             = saved_state.best             or nil,
    last             = saved_state.last             or nil,
  }
end

train_holdout_methods.execute =
  april_doc{
    class = "method", summary = "Runs one training epoch",
    description ={
      "This method executes one training epoch. It receives an",
      "epoch function which where the user trains the model,",
      "and which returns the trained model, the train loss, and the",
      "validation loss.",
    },
    params = {
      {
        "A function which trains a model and returns the trained model,",
        "the training loss and the validation loss",
      },
      "Variadic list of arguments for the function [optional]",
    },
    outputs = { "True or false, indicating if the training continues or not" },
  } ..
  function(self, epoch_function, ...)
    local params = self.params
    local state  = self.state
    -- check max epochs
    if state.current_epoch >= params.max_epochs then
      return false
    end
    -- check stopping criterion
    if ( state.current_epoch > params.min_epochs and
         params.stopping_criterion(state) ) then
      return false
    end
    -- compute one training step by using epoch_function
    state.current_epoch = state.current_epoch + 1
    state.last, state.train_error, state.validation_error = epoch_function(...)
    assert(state.last and state.train_error and state.validation_error,
           "Needs a function which returns three values: a model, training error and validation error")
    -- update with the best model
    if ( state.validation_error < state.best_val_error or
         state.current_epoch <= params.epochs_wo_validation ) then
      local abs_error = math.abs(state.best_val_error - state.validation_error)
      local rel_error = abs_error / math.abs(state.best_val_error)
      if state.best_val_error == math.huge or rel_error > params.tolerance then
        state.best_epoch     = state.current_epoch
        state.best_val_error = state.validation_error
        state.best           = util.clone( state.last )
      end
    end
    return true
  end

train_holdout_methods.set_param =
  april_doc{
    class = "method",
    summary =
      "Modifies one parameter of which was given at construction",
    params = {
      "The parameter name",
      "The parameter value",
    },
  } ..
  function(self,name,value)
    april_assert(self.params[name], "Param %s not found", name)
    self.params[name] = value
  end

train_holdout_methods.get_param =
  april_doc{
    class = "method",
    summary =
      "Returns the value of a param",
    params = {
      "The parameter name",
    },
    outputs = { "The paremter value" },
  } ..
  function(self,name)
    return self.params[name]
  end

train_holdout_methods.get_state =
  april_doc{
    class = "method",
    summary =
      "Returns the state of the training",
    outputs = {
      "Current epoch",
      "Train loss",
      "Validation loss",
      "Best epoch",
      "Best epoch validation loss",
      "Best trained model",
      "Last trained model",
    },
  } ..
  function(self)
    local state = self.state
    return state.current_epoch, state.train_error, state.validation_error,
    state.best_epoch, state.best_val_error, state.best, state.last
  end

train_holdout_methods.get_state_table =
  april_doc{
    class = "method",
    summary =
      "Returns the state table of the training",
    outputs = {
      current_epoch = "Current epoch",
      train_error = "Train loss",
      validation_error = "Validation loss",
      best_epoch = "Best epoch",
      best_val_error = "Best epoch validation loss",
      best = "Best trained model",
      last = "Last trained model",
    },
  } ..
  function(self)
    return self.state
  end

train_holdout_methods.get_state_string =
  april_doc{
    class = "method",
    summary =
    "Returns the state of the training in string format, for printing",
    outputs = {
      { "A string with the format ('%5d %.6f %.6f    %5d %.6f',",
        "current_epoch,train_error,validation_error,best_epoch,best_val_error)" }
    },
  } ..
  function(self)
    local state = self.state
    return string.format("%5d %.6f %.6f    %5d %.6f",
                         state.current_epoch,
                         state.train_error,
                         state.validation_error,
                         state.best_epoch,
                         state.best_val_error)
  end

train_holdout_methods.is_best =
  april_doc{
    class = "method",
    summary = "Returns if current epoch is the best epoch",
    outputs = {
      "A boolean"
    },
  } ..
  function(self)
    local state = self.state
    return state.current_epoch == state.best_epoch
  end

function train_holdout_methods:ctor_name()
  return "trainable.train_holdout_validation"
end
function train_holdout_methods:ctor_params()
  return self.params, self.state
end

train_holdout_methods.save =
  april_doc{
    class = "method",
    summary = "Saves the training in a filename",
    description = {
      "Saves the training in a filename.",
      "If the filename exists, it is renamed as filename.bak",
    },
    params={
      "The filename",
      { "The format for matrix data ('ascii' or 'binary'),",
        "[optional] by default 'binary'", },
      {
        "Extra data dictionary (a table) [optional].",
        "It is useful to store random objects.",
        "The serialization of this objects is automatic",
        "if they has a 'to_lua_string(format)' method, or",
        "if they are Lua standard types (number, string, table).",
      },
    },
  } ..
  function(self,filename,format,extra)
    assert(format==nil or luatype(format)=="string",
           "Second argument is a string with the format: 'binary' or 'ascii'")
    local f = io.open(filename, "r")
    if f then
      f:close()
      os.execute(string.format("mv -f %s %s.bak", filename, filename))
    end
    local f = io.open(filename, "w") or error("Unable to open " .. filename)
    f:write("return ")
    f:write(self:to_lua_string(format))
    if extra then
      f:write(",\n")
      f:write(table.tostring(extra, format))
    end
    f:write("\n")
    f:close()
  end

trainable.train_holdout_validation.load =
  april_doc{
    class = "function",
    summary = "Loads the training from a filename",
    params={
      "The filename",
    },
    outputs = {
      "A train_holdout_methods instance",
      "A table with extra saved data or nil if not given when saving",
    },
  } ..
  function(filename)
    local f = loadfile(filename) or error("Unable to open " .. filename)
    local obj,extra = f()
    april_assert(obj, "Impossible to load chunk from file %s", filename)
    return obj,extra
  end

-------------------------------------------------------------------------------

------------------------------------
-- TRAIN_HOLDOUT_VALIDATION CLASS --
------------------------------------

local train_wo_validation,train_wo_validation_methods =
  class("trainable.train_wo_validation", aprilio.lua_serializable)
trainable = trainable or {} -- global environment
trainable.train_wo_validation = train_wo_validation -- global environment

april_set_doc(trainable.train_wo_validation, {
		class       = "class",
		summary     = "Training class without validation",
		description ={
		  "This training class defines a train_func which",
		  "follows a training schedule based on training error or",
		  "in number of epochs. Method execute receives a function",
		  "which trains one epoch and returns the trainer object and",
		  "the training loss. This method",
		  "returns true in case the training continues, or false if",
		  "the stop criterion is true.",
		}, })

april_set_doc(trainable.train_wo_validation, {
		class = "method", summary = "Constructor",
		description ={
		  "Constructor of the train_wo_validation class.",
		},
		params = {
		  min_epochs = "Minimum number of epochs for training [optional]. By default it is 0,",
		  max_epochs = "Maximum number of epochs for training",
		  percentage_stopping_criterion = "A number [optional]. By default it is 0.01",
		  first_epoch = "The first epoch number [optional]. By default it is 1",
		},
		outputs = { "Instantiated object" }, })

function train_wo_validation:constructor(t,saved_state)
  local params = get_table_fields(
    {
      min_epochs = { mandatory=true, type_match="number", default=0 },
      max_epochs = { mandatory=true, type_match="number" },
      percentage_stopping_criterion = { mandatory=true, type_match="number",
					default = 0.01 },
      first_epoch        = { mandatory=false, type_match="number", default=1 },
    }, t)
  local saved_state = saved_state or {}
  self.params = params
  self.state  = {
    current_epoch     = saved_state.current_epoch     or 0,
    train_error       = saved_state.train_error       or math.huge,
    train_improvement = saved_state.train_improvement or math.huge,
    last              = saved_state.last              or nil,
  }
end

train_wo_validation_methods.execute =
  april_doc{
    class = "method", summary = "Runs one training epoch",
    description ={
      "This method executes one training epoch. It receives an",
      "epoch function which where the user trains the model,",
      "and which returns the trained model and the train loss.",
    },
    params = {
      {
        "A function which trains a model and returns the trained model",
        "and the training loss",
      },
      "Variadic list of arguments for the function [optional]",
    },
    outputs = { "True or false, indicating if the training continues or not" },
  } ..
  function(self, epoch_function, ...)
    local params = self.params
    local state  = self.state
    -- stopping criterion
    if (state.current_epoch > params.min_epochs and
        state.train_improvement < params.percentage_stopping_criterion) then
      return false
    end
    -- check max epochs
    if state.current_epoch >= params.max_epochs then
      return false
    end
    -- compute one training step by using epoch_function
    state.current_epoch = state.current_epoch + 1
    local model,tr_err = epoch_function(...)
    assert(model and tr_err,
           "Needs a function which returns two values: a model and training error")
    --
    local prev_tr_err    = state.train_error
    local tr_improvement
    if state.current_epoch > 1 then
      tr_improvement = (prev_tr_err - tr_err)/prev_tr_err
    else
      tr_improvement = math.huge
    end
    
    state.train_error       = tr_err
    state.train_improvement = tr_improvement
    state.last              = model
    
    return true
  end

train_wo_validation_methods.set_param = 
  april_doc{
    class = "method",
    summary =
      "Modifies one parameter of which was given at construction",
    params = {
      "The parameter name",
      "The parameter value",
    },
  } ..
  function(self,name,value)
    april_assert(self.params[name], "Param %s not found", name)
    self.params[name] = value
  end

train_wo_validation_methods.get_param =
  april_doc{
    class = "method",
    summary =
      "Returns the value of a param",
    params = {
      "The parameter name",
    },
    outputs = { "The paremter value" },
  } ..
  function(self,name)
    return self.params[name]
  end

train_wo_validation_methods.get_state =
  april_doc{
    class = "method",
    summary =
      "Returns the state of the training",
    outputs = {
      "Current epoch",
      "Train loss",
      "Train relative improvement",
      "Last trained model",
    },
  } ..
  function(self)
    local state = self.state
    return state.current_epoch, state.train_error,
    state.train_improvement,state.last
  end

train_wo_validation_methods.get_state_table =
  april_doc{
    class = "method",
    summary =
      "Returns the state table of the training",
    outputs = {
      current_epoch = "Current epoch",
      train_error = "Train loss",
      train_improvement = "Train relative improvement",
      last = "Last trained model",
    },
  } ..
  function(self)
    return self.state
  end

train_wo_validation_methods.get_state_string =
  april_doc{
    class = "method",
    summary =
    "Returns the state of the training in string format, for printing",
    outputs = {
      { "A string with the format ('%5d %.6f    %.6f',",
        "current_epoch,train_error,train_improvement)" }
    },
  } ..
  function(self)
    local state = self.state
    return string.format("%5d %.6f    %.6f",
                         state.current_epoch,
                         state.train_error,
                         state.train_improvement)
  end

function train_wo_validation_methods:ctor_name()
  return "trainable.train_wo_validation"
end
function train_wo_validation_methods:ctor_params()
  return self.params, self.state
end

train_wo_validation_methods.save =
  april_doc{
    class = "method",
    summary = "Saves the training in a filename",
    description = {
      "Saves the training in a filename.",
      "If the filename exists, it is renamed as filename.bak",
    },
    params={
      "The filename",
      { "The format for matrix data ('ascii' or 'binary'),",
        "[optional] by default 'binary'", },
      {
        "Extra data dictionary (a table) [optional].",
        "It is useful to store random objects.",
        "The serialization of this objects is automatic",
        "if they has a 'to_lua_string(format)' method, or",
        "if they are Lua standard types (number, string, table).",
      },
    }
  } ..
  function(...)
    return train_holdout_methods.save(...)
  end

trainable.train_wo_validation.load = 
  april_doc{
    class = "function",
    summary = "Loads the training from a filename",
    params={
      "The filename",
    },
    outputs = {
      "A train_wo_validation_methods instance",
      "A table with extra saved data or nil if not given when saving",
    },
  } ..
  function(...)
    return trainable.train_holdout_validation.load(...)
  end

-------------------------------------------------------------------------------

-------------------------
-- STOPPING CRITERIA --
-------------------------

trainable.stopping_criteria = trainable.stopping_criteria or {}

april_set_doc(trainable.stopping_criteria, {
		class       = "namespace",
		summary     = "Table with built-in stopping criteria", })


--------------------------------------------------------------------------

trainable.stopping_criteria.make_max_epochs_wo_imp_absolute =
  april_doc{
    class       = "function",
    summary     = "Returns a stopping criterion based on absolute loss.",
    description = 
      {
        "This function returns a stopping criterion function",
        "which returns is true if current_epoch - best_epoch >= abs_max."
      }, 
    params = { "Absolute maximum difference (abs_max)" },
    outputs = { "A stopping criterion function" },
  } ..
  function(abs_max)
    local f = function(params)
      return (params.current_epoch - params.best_epoch) >= abs_max
    end
    return f
  end

--------------------------------------------------------------------------

trainable.stopping_criteria.make_max_epochs_wo_imp_relative =
  april_doc{
    class       = "function",
    summary     = "Returns a stopping criterion based on relative loss.",
    description = 
      {
        "This function returns a stopping criterion function",
        "which returns is true if not",
        "current_epoch/best_epoch < rel_max."
      }, 
    params = { "Relative maximum difference (rel_max)" },
    outputs = { "A stopping criterion function" },
  } ..
  function(rel_max)
    local f = function(params)
      return not (params.current_epoch/params.best_epoch < rel_max)
    end
    return f
  end
