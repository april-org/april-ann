-------------------------------
-- LOCAL AUXILIARY FUNCTIONS --
-------------------------------

local function check_dataset_sizes(ds1, ds2)
  if ds1:numPatterns() ~= ds2:numPatterns() then
    error(string.format("Different input/output datasets "..
			  "numPatterns found: "..
			  "%d != %d",
			ds1:numPatterns(),
			ds2:numPatterns()))
  end
  return true
end

-----------------------
-- TRAINABLE CLASSES --
-----------------------

-- CLASS supervised_trainer
april_set_doc("trainable.supervised_trainer",
	      "Lua class for supervised machine learning models training")
class("trainable.supervised_trainer")

april_set_doc("trainable.supervised_trainer", "Methods")
function trainable.supervised_trainer:__call(ann_component, loss_function)
  local obj = {
    ann_component    = ann_component,
    loss_function    = loss_function,
    weights_table    = {},
    components_table = {},
    weights_order    = {},
    components_order = {}
  }
  obj = class_instance(obj, self, true)
  return obj
end

function trainable.supervised_trainer:set_loss_function(loss_function)
  self.loss_function = loss_function
end

function trainable.supervised_trainer:randomize_weights(t)
  local params = get_table_fields(
    {
      random  = { mandatory = true },
      inf     = { mandatory = true },
      sup     = { mandatory = true },
      use_fanin  = { mandatory = false, default = false },
      use_fanout = { mandatory = false, default = false },
    }, t)
  if #self.weights_order == 0 then
    error("Execute build method before randomize_weights")
  end
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

function trainable.supervised_trainer:build(weights_table)
  self.weights_table = weights_table or {}
  self.weights_table,
  self.components_table = self.ann_component:build(self.weights_table)
  self.weights_order = {}
  for name,_ in pairs(self.weights_table) do table.insert(self.weights_order,
							  name) end
  table.sort(self.weights_order)
  self.components_order = {}
  for name,_ in pairs(self.components_table) do table.insert(self.components_order,
							     name) end
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
	      "\t                 t.bunch_size  mini batch size (bunch)")
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
      distribution   = { mandatory = false, default=nil,
			 getter = get_table_fields_ipairs{
			   input_dataset  = { mandatory=true },
			   output_dataset = { mandatory=true },
			   probability    = { type_match="number",
					      mandatory=true },
			 },
      },
      bunch_size     = { type_match = "number", mandatory = true },
      shuffle        = { type_match = "random", mandatory = false, default=nil },
      replacement    = { type_match = "number", mandatory = false, default=nil },
    }, t)
  -- ERROR CHECKING
  if params.input_dataset == not params.output_dataset then
    error("input_dataset and output_dataset fields are mandatory together")
  end
  if params.input_dataset and params.distribution then
    error("input_dataset/output_dataset fields are forbidden with distribution")
  end
  --
  
  if params.distribution then error("Distribution is not correctly implemented") end
  
  -- params.ann  = self.ann_component
  -- params.loss = self.loss_function
  -- trainable.supervised_trainer_static_functions.train_dataset(params)

  -- TRAINING TABLES
  
  -- for each pattern, a pair of input/output datasets (or nil if not
  -- distribution)
  local ds_pat_table = {}
  -- for each pattern, index in corresponding datasets
  local ds_idx_table = {}
  self.loss_function:reset()
  if params.distribution then
    -- Training with distribution: given a table of datasets the patterns are
    -- sampled following the given apriory probability
    local _=params.shuffle or error("shuffle is mandatory with distribution")
    local _=params.replacement or error("replacement is mandatory with distribution")

    local sizes = {}
    for i,v in ipairs(params.distribution) do
      if isa(v.input_dataset, dataset) then
	v.input_dataset  = dataset.token.wrapper(v.input_dataset)
	v.output_dataset = dataset.token.wrapper(v.output_dataset)
      end
      check_dataset_sizes(v.input_dataset, v.output_dataset)
      table.insert(aprioris, v.probability)
      table.insert(sizes, v.input_dataset:numPatterns())
    end
    -- generate training tables
    local dice = random.dice(aprioris)
    for i=1,#params.replacement do
      local whichclass=dice:thrown(params.shuffle)
      local ds_table=params.distribution[whichclass]
      local idx=shuffle:randInt(1,sizes[whichclass])
      table.insert(ds_pat_table, params.distribution[whichclass])
      table.insert(ds_idx_table, idx)
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
      local _=params.shuffle or error("shuffle is mandatory with replacement")
      for i=1,params.replacement do
	table.insert(ds_idx_table, param.shuffle:randInt(1,num_patterns))
      end
    elseif params.shuffle then
      ds_idx_table = params.shuffle:shuffle(num_patterns)
    else
      for i=1,num_patterns do table.insert(ds_idx_table, i) end
    end
  end
  -- TRAIN USING ds_idx_table and ds_pat_table
  local bunch_indexes = {}
  for i=1,#ds_idx_table do
    local idx = ds_idx_table[i]
    table.insert(bunch_indexes, idx - 1) -- OJITO restamos 1
    if i==#ds_idx_table or #bunch_indexes == bunch_size then
      local input_bunch  = params.input_dataset:getPatternBunch(bunch_indexes)
      local output_bunch = params.output_dataset:getPatternBunch(bunch_indexes)
      trainer:train_step(input_bunch, output_bunch)
      bunch_indexes = {}
    end
  end
  ds_pat_table = nil
  ds_idx_table = nil
  return self.loss_function:get_accum_loss()
end

function trainable.supervised_trainer:validate_dataset(t)
  -- t.ann  = self.ann_component
  -- t.loss = self.loss_function
  -- trainable.supervised_trainer_static_functions.validate_dataset(t)

  local params = get_table_fields(
    {
      input_dataset  = { mandatory = true, default=nil },
      output_dataset = { mandatory = true, default=nil },
      bunch_size     = { type_match = "number", mandatory = true },
      shuffle        = { type_match = "random", mandatory = false, default=nil },
      replacement    = { type_match = "number", mandatory = false, default=nil },
    }, t)
  -- ERROR CHECKING
  if params.input_dataset == not params.output_dataset then
    error("input_dataset and output_dataset fields are mandatory together")
  end
  if params.input_dataset and params.distribution then
    error("input_dataset/output_dataset fields are forbidden with distribution")
  end
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
    local _=params.shuffle or error("shuffle is mandatory with replacement")
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
      trainer:validate_step(input_bunch, output_bunch)
      bunch_indexes = {}
    end
  end
  ds_pat_table = nil
  ds_idx_table = nil
  return self.loss_function:get_accum_loss()
end

function trainable.supervised_trainer:use_dataset(t)
  -- t.ann  = self.ann_component
  -- trainable.supervised_trainer_static_functions.use_dataset(t)
end
