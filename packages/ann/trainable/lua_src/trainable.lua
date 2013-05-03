-------------------------------
-- LOCAL AUXILIARY FUNCTIONS --
-------------------------------

local function check_dataset_sizes(ds1, ds2)
  assert(ds1:numPatterns() == ds2:numPatterns(),
	 string.format("Different input/output datasets "..
			 "numPatterns found: "..
			 "%d != %d",
		       ds1:numPatterns(),
		       ds2:numPatterns()))
end

-----------------------
-- TRAINABLE CLASSES --
-----------------------

-- CLASS supervised_trainer
april_set_doc("trainable.supervised_trainer",
	      "Lua class for supervised machine learning models training")
class("trainable.supervised_trainer")

april_set_doc("trainable.supervised_trainer", "Methods")
function trainable.supervised_trainer:__call(ann_component,
					     loss_function,
					     bunch_size)
  local obj = {
    ann_component    = assert(ann_component,"Needs an ANN component object"),
    loss_function    = loss_function or false,
    weights_table    = {},
    components_table = {},
    weights_order    = {},
    components_order = {},
    bunch_size       = bunch_size or false,
  }
  obj = class_instance(obj, self, true)
  if ann_component:get_is_built() then obj:build() end
  return obj
end

function trainable.supervised_trainer:set_loss_function(loss_function)
  self.loss_function = loss_function
end

function trainable.supervised_trainer:component(str)
  if #self.weights_order == 0 then
    error("Needs execution of build method")
  end
  return self.components_table[str] or error("Incorrect component name " .. str)
end

function trainable.supervised_trainer:weights(str)
  if #self.weights_order == 0 then
    error("Needs execution of build method")
  end
  return self.weights_table[str] or error("Incorrect weights name " .. str)
end

function trainable.supervised_trainer:randomize_weights(t)
  local params = get_table_fields(
    {
      random  = { isa_match = random,  mandatory = true },
      inf     = { type_match="number", mandatory = true },
      sup     = { type_match="number", mandatory = true },
      use_fanin  = { type_match="boolean", mandatory = false, default = false },
      use_fanout = { type_match="boolean", mandatory = false, default = false },
    }, t)
  assert(#self.weights_order > 0,
	 "Execute build method before randomize_weights")
  for i,wname in ipairs(self.weights_order) do
    local current_inf = params.inf
    local current_sup = params.sup
    local constant    = 0
    local connection  = self.weights_table[wname]
    if params.use_fanin then
      constant = constant + connection:get_input_size()
    end
    if params.use_fanout then
      constant = constant + connection:get_output_size()
    end
    if constant > 0 then
      current_inf = current_inf / math.sqrt(constant)
      current_sup = current_sup / math.sqrt(constant)
    end
    connection:randomize_weights{ random = params.random,
				  inf    = current_inf,
				  sup    = current_sup }
  end
end

function trainable.supervised_trainer:build(t)
  local params = get_table_fields(
    {
      weights = { type_match="table",  mandatory = false, default=nil },
      input   = { type_match="number", mandatory = false, default=nil },
      output  = { type_match="number", mandatory = false, default=nil },
    }, t or {})
  self.weights_table = params.weights_table or {}
  self.weights_table,
  self.components_table = self.ann_component:build{
    input   = params.input,
    output  = params.output,
    weights = self.weights_table, }
  self.weights_order = {}
  for name,_ in pairs(self.weights_table) do
    table.insert(self.weights_order, name)
  end
  table.sort(self.weights_order)
  self.components_order = {}
  for name,_ in pairs(self.components_table) do
    table.insert(self.components_order, name)
  end
  table.sort(self.components_order)
  return self.weights_table,self.components_table
end

april_set_doc("trainable.supervised_trainer",
	      "\ttrain_step(input,target) => performs one training step "..
		"(reset, forward, loss, gradient, backprop, and update)")
function trainable.supervised_trainer:train_step(input, target)
  if type(input)  == "table" then input  = tokens.memblock(input)  end
  if type(target) == "table" then target = tokens.memblock(target) end
  self.ann_component:reset()
  local output   = self.ann_component:forward(input)
  local tr_loss  = self.loss_function:loss(output, target)
  local gradient = self.loss_function:gradient(output, target)
  self.ann_component:backprop(gradient)
  self.ann_component:update()
  return tr_loss,gradient
end

function trainable.supervised_trainer:validate_step(input, target)
  if type(input)  == "table" then input  = tokens.memblock(input)  end
  if type(target) == "table" then target = tokens.memblock(target) end
  self.ann_component:reset()
  local output   = self.ann_component:forward(input)
  local tr_loss  = self.loss_function:loss(output, target)
  return tr_loss
end

function trainable.supervised_trainer:calculate(input)
  if type(input) == "table" then input = tokens.memblock(input) end
  return self.ann_component:forward(input):convert_to_memblock():to_table()
end

april_set_doc("trainable.supervised_trainer",
	      "\ttrain_step(t) => performs one training epoch with a given "..
		" table with datasets. Arguments:")
april_set_doc("trainable.supervised_trainer",
	      "\t                 [t.bunch_size]  mini batch size (bunch)")
april_set_doc("trainable.supervised_trainer",
	      "\t                 [t.input_dataset]  dataset with input patterns")
april_set_doc("trainable.supervised_trainer",
	      "\t                 [t.output_dataset]  dataset with output patterns")
april_set_doc("trainable.supervised_trainer",
	      "\t                 [t.distribution]")
april_set_doc("trainable.supervised_trainer",
	      "\t                 [t.shuffle]  optional random object")
april_set_doc("trainable.supervised_trainer",
	      "\t                 [t.replacement]  optional replacement size")
function trainable.supervised_trainer:train_dataset(t)
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
      bunch_size     = { type_match = "number",
			 mandatory = (self.bunch_size == false),
			 default=self.bunch_size },
      shuffle        = { isa_match  = random,   mandatory = false, default=nil },
      replacement    = { type_match = "number", mandatory = false, default=nil },
    }, t)
  -- ERROR CHECKING
  assert(params.input_dataset ~= not params.output_dataset,
	 "input_dataset and output_dataset fields are mandatory together")
  assert(not params.input_dataset or not params.distribution,
	 "input_dataset/output_dataset fields are forbidden with distribution")
  --
  
  -- TRAINING TABLES
  
  -- for each pattern, index in dataset
  local ds_idx_table = {}
  -- set to ZERO the accumulated of loss
  self.loss_function:reset()
  if params.distribution then
    -- Training with distribution: given a table of datasets the patterns are
    -- sampled following the given apriory probability
    assert(params.shuffle,"shuffle is mandatory with distribution")
    assert(params.replacement,"replacement is mandatory with distribution")
    params.input_dataset  = dataset.token.union()
    params.output_dataset = dataset.token.union()
    local aprioris = {}
    local sizes    = {}
    local sums     = { 0 }
    for i,v in ipairs(params.distribution) do
      if isa(v.input_dataset, dataset) then
	v.input_dataset  = dataset.token.wrapper(v.input_dataset)
	v.output_dataset = dataset.token.wrapper(v.output_dataset)
      end
      check_dataset_sizes(v.input_dataset, v.output_dataset)
      table.insert(aprioris, v.probability)
      table.insert(sizes, v.input_dataset:numPatterns())
      table.insert(sums, sums[#sums] + sizes[#sizes])
      params.input_dataset:push_back(v.input_dataset)
      params.output_dataset:push_back(v.output_dataset)
    end
    -- generate training tables
    local dice = random.dice(aprioris)
    for i=1,params.replacement do
      local whichclass=dice:thrown(params.shuffle)
      local idx=params.shuffle:randInt(1,sizes[whichclass])
      table.insert(ds_idx_table, idx + sums[whichclass])
    end
  else
    if isa(params.input_dataset, dataset) then
      params.input_dataset  = dataset.token.wrapper(params.input_dataset)
      params.output_dataset = dataset.token.wrapper(params.output_dataset)
    end
    check_dataset_sizes(params.input_dataset, params.output_dataset)
    local num_patterns = params.input_dataset:numPatterns()
    -- generate training tables depending on training mode (replacement,
    -- shuffled, or sequential)
    if params.replacement then
      assert(params.shuffle,"shuffle is mandatory with replacement")
      for i=1,params.replacement do
	table.insert(ds_idx_table, params.shuffle:randInt(1,num_patterns))
      end
    elseif params.shuffle then
      ds_idx_table = params.shuffle:shuffle(num_patterns)
    else
      for i=1,num_patterns do table.insert(ds_idx_table, i) end
    end
  end
  -- TRAIN USING ds_idx_table
  local bunch_indexes = {}
  for i=1,#ds_idx_table do
    local idx = ds_idx_table[i]
    table.insert(bunch_indexes, idx - 1) -- OJITO restamos 1
    if i==#ds_idx_table or #bunch_indexes == bunch_size then
      local input_bunch  = params.input_dataset:getPatternBunch(bunch_indexes)
      local output_bunch = params.output_dataset:getPatternBunch(bunch_indexes)
      self:train_step(input_bunch, output_bunch)
      bunch_indexes = {}
    end
  end
  ds_idx_table = nil
  return self.loss_function:get_accum_loss()
end

function trainable.supervised_trainer:validate_dataset(t)
  local params = get_table_fields(
    {
      input_dataset  = { mandatory = true },
      output_dataset = { mandatory = true },
      bunch_size     = { type_match = "number",
			 mandatory = (self.bunch_size == false),
			 default=self.bunch_size },
      shuffle        = { isa_match  = random, mandatory = false, default=nil },
      replacement    = { type_match = "number", mandatory = false, default=nil },
    }, t)
  -- ERROR CHECKING
  assert(params.input_dataset ~= not params.output_dataset,
	 "input_dataset and output_dataset fields are mandatory together")
  assert(not params.input_dataset or not params.distribution,
	 "input_dataset/output_dataset fields are forbidden with distribution")
  -- TRAINING TABLES
  
  -- for each pattern, index in corresponding datasets
  local ds_idx_table = {}
  self.loss_function:reset()
  if isa(params.input_dataset, dataset) then
    params.input_dataset  = dataset.token.wrapper(params.input_dataset)
    params.output_dataset = dataset.token.wrapper(params.output_dataset)
  end
  check_dataset_sizes(params.input_dataset, params.output_dataset)
  local num_patterns = params.input_dataset:numPatterns()
  -- generate training tables depending on training mode (replacement,
  -- shuffled, or sequential)
  if params.replacement then
    assert(params.shuffle,"shuffle is mandatory with replacement")
    for i=1,params.replacement do
      table.insert(ds_idx_table, param.shuffle:randInt(1,num_patterns))
    end
  elseif params.shuffle then
    ds_idx_table = params.shuffle:shuffle(num_patterns)
  else
    for i=1,num_patterns do table.insert(ds_idx_table, i) end
  end
  -- TRAIN USING ds_idx_table
  local bunch_indexes = {}
  for i=1,#ds_idx_table do
    local idx = ds_idx_table[i]
    table.insert(bunch_indexes,  idx - 1) -- OJITO - 1
    if i==#ds_idx_table or #bunch_indexes == bunch_size then
      local input_bunch  = params.input_dataset:getPatternBunch(bunch_indexes)
      local output_bunch = params.output_dataset:getPatternBunch(bunch_indexes)
      self:validate_step(input_bunch, output_bunch)
      bunch_indexes = {}
    end
  end
  ds_idx_table = nil
  return self.loss_function:get_accum_loss()
end

function trainable.supervised_trainer:use_dataset(t)
  local params = get_table_fields(
    {
      input_dataset  = { mandatory = true },
      output_dataset = { mandatory = false, default=nil },
      bunch_size     = { type_match = "number",
			 mandatory = (self.bunch_size == false),
			 default=self.bunch_size },
    }, t)
  if isa(params.input_dataset, dataset) then
    params.input_dataset = dataset.token.wrapper(params.input_dataset)
    if params.output_dataset then
      params.output_dataset = dataset.token.wrapper(params.output_dataset)
    else
      local nump    = params.input_dataset:numPatterns()
      local outsize = self.ann_component:get_output_size()
      params.output_dataset = dataset.matrix(matrix(nump, outsize))
      t.output_dataset      = params.output_dataset
      params.output_dataset = dataset.token.wrapper(params.output_dataset)
    end
  elseif not params.output_dataset then
    params.output_dataset = dataset.token.vector(params.input_dataset:numPatterns())
    t.output_dataset = params.output_dataset
  end
  local bunch_indexes = {}
  local nump = params.input_dataset:numPatterns()
  for i=1,nump do
    table.insert(bunch_indexes, i)
    if #bunch_indexes==params.bunch_size or i==nump then
      local input  = params.input_dataset:getPatternBunch(bunch_indexes)
      local output = self.ann_component:forward(input)
      params.output_dataset:putPatternBunch(bunch_indexes,output)
      bunch_indexes = {}
    end
  end
  return t.output_dataset
end

function trainable.supervised_trainer:show_weights()
  for _,wname in pairs(self.weights_order) do
    local w = self.weights_table[wname]:weights():toTable()
    print(wname, table.concat(w, " "))
  end
end

function trainable.supervised_trainer:clone()
  local obj = trainable.supervised_trainer(self.ann_component:clone(),
					   nil,
					   self.bunch_size)
  if self.loss_function then
    obj:set_loss_function(self.loss_function:clone())
  end
  if #self.weights_order > 0 then
    for wname,connections in pairs(self.weights_table) do
      obj.weights_table[wname] = connections:clone()
    end
    obj:build{ weights = obj.weights_table,
	       input   = self.ann_component:get_input_size(),
	       output  = self.ann_component:get_output_size(), }
  end
  return obj
end

-- params is a table like this:
-- { training_table   = { input_dataset = ...., output_dataset = ....., .....},
--   validation_table = { input_dataset = ...., output_dataset = ....., .....},
--   validation_function = function( thenet, validation_table ) .... end
--   min_epochs = NUMBER,
--   max_epochs = NUMBER,
--   -- train_params is this table ;)
--   stopping_criterion = function{ current_epoch, best_epoch, best_val_error, train_error, validation_error, train_params } .... return true or false end
--   update_function = function{ current_epoch, best_epoch, best_val_error, 
--   first_epoch = 1
-- }
--
-- and returns this:
--  return { best_net         = best_net,
--	     best_epoch       = best_epoch,
--	     best_val_error   = best_val_error,
--	     num_epochs       = epoch,
--	     last_train_error = last_train_error,
--	     last_val_error   = last_val_error }
function trainable.supervised_trainer:train_holdout_validation(t)
  local params = get_table_fields(
    {
      training_table   = { mandatory=true, type_match="table" },
      validation_table = { mandatory=true, type_match="table" },
      validation_function = { mandatory=false, type_match="function",
			      default=function(thenet,t)
				return thenet:validate_dataset(t)
			      end },
      best_function = { mandatory=false, type_match="function",
			default=function(thenet,t) end },
      min_epochs = { mandatory=true, type_match="number" },
      max_epochs = { mandatory=true, type_match="number" },
      stopping_criterion = { mandatory=true, type_match="function" },
      update_function    = { mandatory=false, type_match="function",
			     default=function(t) return end },
      first_epoch        = { mandatory=false, type_match="number", default=1 },
    }, t)
  local best_epoch       = params.first_epoch
  local best             = self:clone()
  local best_val_error   = params.validation_function(self,
						      params.validation_table)
  local last_val_error   = best_val_error
  local last_train_error = 0
  local last_epoch       = 0
  for epoch=params.first_epoch,params.max_epochs do
    collectgarbage("collect")
    local clock = util.stopwatch()
    clock:go()
    local tr_error  = self:train_dataset(params.training_table)
    local val_error = params.validation_function(self, params.validation_table)
    last_train_error,last_val_error,last_epoch = tr_error,val_error,epoch
    clock:stop()
    cpu,wall = clock:read()
    if val_error < best_val_error then
      best_epoch     = epoch
      best_val_error = val_error
      best           = self:clone()
      params.best_function(best)
    end
    params.update_function({ current_epoch    = epoch,
			     best_epoch       = best_epoch,
			     best_val_error   = best_val_error,
			     train_error      = tr_error,
			     validation_error = val_error,
			     cpu              = cpu,
			     wall             = wall,
			     train_params     = params })
    if (epoch > params.min_epochs and
	  params.stopping_criterion({ current_epoch    = epoch,
				      best_epoch       = best_epoch,
				      best_val_error   = best_val_error,
				      train_error      = tr_error,
				      validation_error = val_error,
				      train_params     = params })) then
      break						  
    end
  end
  return { best             = best,
	   best_val_error   = best_val_error,
	   best_epoch       = best_epoch,
	   last_epoch       = last_epoch,
	   last_train_error = last_train_error,
	   last_val_error   = last_val_error }
end

-- This function trains without validation, it is trained until a maximum of
-- epochs or until the improvement in training error is less than given
-- percentage
--
-- params is a table like this:
-- { training_table   = { input_dataset = ...., output_dataset = ....., .....},
--   min_epochs = NUMBER,
--   max_epochs = NUMBER,
--   update_function = function{ current_epoch, best_epoch, best_val_error,
--   percentage_stopping_criterion = NUMBER (normally 0.01 or 0.001)
-- }
--
-- returns the trained object
function trainable.supervised_trainer:train_wo_validation(t)
  local params = get_table_fields(
    {
      training_table = { mandatory=true },
      min_epochs = { mandatory=true, type_match="number" },
      max_epochs = { mandatory=true, type_match="number" },
      update_function = { mandatory=false, type_match="function",
			  default=function(t) return end },
      percentage_stopping_criterion = { mandatory=false, type_match="number",
					default=0.01 },
    }, t)
  local prev_tr_err = 1111111111111
  local best        = self:clone()
  for epoch=1,params.max_epochs do
    local tr_table = params.training_table
    if type(tr_table) == "function" then tr_table = tr_table() end
    collectgarbage("collect")
    local tr_err         = self:train_dataset(tr_table)
    local tr_improvement = (prev_tr_err - tr_err)/prev_tr_err
    if (epoch > params.min_epochs and
	tr_improvement < params.percentage_stopping_criterion) then
      break
    end
    best = self:clone()
    params.update_function{ current_epoch     = epoch,
			    train_error       = tr_err,
			    train_improvement = tr_improvement,
			    train_params      = params }
    prev_tr_err = tr_err
  end
  return best
end

-------------------------
-- STOPPING CRITERIONS --
-------------------------
trainable.stopping_criterions = trainable.stopping_criterions or {}

function trainable.stopping_criterions.make_max_epochs_wo_imp_absolute(abs_max)
  local f = function(params)
    return (params.current_epoch - params.best_epoch) >= abs_max
  end
  return f
end

function trainable.stopping_criterions.make_max_epochs_wo_imp_relative(rel_max)
  local f = function(params)
    return not (params.current_epoch/params.best_epoch < rel_max)
  end
  return f
end
