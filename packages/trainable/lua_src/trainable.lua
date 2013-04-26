trainable = trainable or {}
trainable.supervised_trainer = trainable.supervised_trainer or {}

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
class(trainable.supervised_trainer)

april_set_doc("trainable.supervised_trainer",
	      "Methods")
function trainable.supervised_trainer:__call(ann_component, loss_function)
  local obj = { ann_component = ann_component, loss_function = loss_function }
  setmetatable(obj, self)
  return obj
end

april_set_doc("trainable.supervised_trainer",
	      "\ttrain_step(input,target) => performs one training step "..
		"(reset, forward, loss, gradient, backprop, and update)")
function trainable.supervised_trainer:train_step(input, target)
  if type("input")  == "table" then input  = tokens.memblock(input)  end
  if type("target") == "table" then target = tokens.memblock(target) end
  self.ann_component:reset()
  local output   = self.ann_component:forward(input)
  local tr_loss  = self.loss_function:loss(output, target)
  local gradient = self.loss_function:gradient(output, target)
  self.ann_component:backprop(gradient)
  self.ann_component:update()
  return tr_loss,gradient
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
      shuffle        = { type_match = "random", mandatroy = false, default=nil },
      replacement    = { type_match = "number", mandatroy = false, default=nil },
    }, t)
  -- ERROR CHECKING
  if params.input_dataset == not params.output_dataset then
    error("input_dataset and output_dataset fields are mandatory together")
  end
  if params.input_dataset and params.distribution then
    error("input_dataset/output_dataset fields are forbidden with distribution")
  end
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
  local input_bunch,output_dataset = {},{}
  collectgarbage("collect")
  for i=1,#ds_idx_table do
    local idx    = ds_idx_table[i]
    local input  = (ds_pat_table[i] or params).input_dataset:getPattern(idx)
    local output = (ds_pat_table[i] or params).output_dataset:getPattern(idx)
    table.insert(input_bunch,  ds_table.input_dataset:getPattern(index))
    table.insert(output_bunch, ds_table.output_dataset:getPattern(index))
    if i==#ds_idx_table or #input_bunch == bunch_size then
      trainer:train_step(input_bunch,output_bunch)
      input_bunch,output_bunch = {},{}
    end
  end
  collectgarbage("collect")
  return self.loss_function:get_accum_loss()
end
