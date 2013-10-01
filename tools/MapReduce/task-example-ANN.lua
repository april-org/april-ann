-- arg is available as a vararg argument when the script is loaded by loadfile
-- or load functions. Please, use arg to parse arguments with cmdOpt.
-- local arg = { ... }

require "common"
-- NOTE that common, worker, and master are tables used by MapReduce paradigm,
-- also note that map and reduce functions are defined by April-ANN, so use
-- other names

local bunch_size     = 32
local replacement    = 192
local weights_random = random(1234)
local description    = "256 inputs 256 tanh 128 tanh 10 log_softmax"
local inf            = -1
local sup            =  1
local shuffle_random = random() -- TOTALLY RANDOM FOR EACH WORKER
local learning_rate  = 0.04
local momentum       = 0.02
local weight_decay   = 1e-04
local max_epochs     = 400
local LOSS_STR       = "LOSS"
local epoch          = 0
local best_epoch     = 0
local best_val_loss  = 1111111111111
local tr_loss
local val_loss
local min_epochs     = 20

local thenet = ann.mlp.all_all.generate(description)
local trainer = trainable.supervised_trainer(thenet,
					     ann.loss.multi_class_cross_entropy(10),
					     bunch_size)
local optimizer = trainer:get_optimizer()

trainer:build()
trainer:randomize_weights{
  random      = weights_random,
  inf         = inf,
  sup         = sup,
  use_fanin   = true,
}

trainer:set_option("learning_rate", learning_rate)
trainer:set_option("momentum",      momentum)
trainer:set_option("weight_decay",  weight_decay)
-- it is better to avoid BIAS regularization 
trainer:set_layerwise_option("b.", "weight_decay", 0)

-- the data size is mandatory. Please, use absolute paths, instead of relative,
-- and remember that in any case the data is loaded in the workers.
dir = os.getenv("APRIL_TOOLS_DIR") .. "/MapReduce/data/"
local data = {
  -- the training are the first 800 digits (80 x 10 classes)
  { dir.."digits.png", 80 },
}

function load_dataset_from_offset_and_steps(m, offset, numSteps)
  local in_ds = dataset.matrix(m,
			       {
				 patternSize = {16,16},
				 offset      = offset,
				 numSteps    = numSteps,
				 stepSize    = {16,16},
				 orderStep   = {1,0}
			       })
  local m2 = matrix(10,{1,0,0,0,0,0,0,0,0,0})
  local out_ds = dataset.matrix(m2,
				{
				  patternSize = {10},
				  offset      = {0},
				  numSteps    = {numSteps[1]*numSteps[2]},
				  stepSize    = {-1},
				  circular    = {true}
				})
  return dataset.token.wrapper(in_ds), dataset.token.wrapper(out_ds)
end

function load_dataset_from_value(value)
  local data,first,last = load(value)()
  local m = common.cache(data,
			 function()
			   return ImageIO.read(data):
			   to_grayscale():
			   invert_colors():
			   matrix()
			 end)
  local offset   = { (first-1) * 16, 0 }
  local numSteps = { (last - first + 1), 10 }
  return table.pack( load_dataset_from_offset_and_steps(m, offset, numSteps) )
end

-- Loads the data if it was necessary, so the master executes decoding, and
-- workers receives the decoded data. The decoded_data could be a Lua string
-- which has the ability to load the data, or directly a data value. The
-- returned value will be passed to the split function in order to split it.
local function decode(encoded_data,data_size)
  return encoded_data
end

-- This function receives a decoded data, its size, and splits it by the given
-- first,last pair of values, and returns the data value split and the size of
-- the split (it is possible to be different from the indicated last-first+1
-- size). The returned values are automatically converted using tostring()
-- function. So, it is possible to return a string, a table with strings, or a
-- value convertible to string, or a table with values convertible to string.
local function split(decoded_data,data_size,first,last)
  return string.format("return %q,%d,%d",decoded_data,first,last), last-first+1
end

-- Receives a key,value pair, and produces an array of key,value string (or able
-- to be string-converted by Lua) pairs. In Machine Learning problems, decoded
-- values could be cached by the given key, avoiding to load it every time,
-- improving the performance of the application. The common.cache function is
-- useful for this purpose. Please, be careful because all cached values will be
-- keep at memory of the machine where the task was executed.
local function mmap(key,value)
  util.omp_set_num_threads(1)
  local in_ds,out_ds = table.unpack(common.cache(key,
						 function()
						   return load_dataset_from_value(value)
						 end))
  --
  local bunch = iterator(range(1,bunch_size)):
  map(function() return shuffle_random:randInt(1,in_ds:numPatterns()) end):
  table()
  --
  local input  = in_ds:getPatternBunch(bunch)
  local target = out_ds:getPatternBunch(bunch)
  local weight_grads,loss_matrix = trainer:compute_gradients_step(input,target)
  -- the weights
  local result = iterator( pairs(weight_grads) ):
  map(function(name,mat)
	local mat_str = mat:to_lua_string()
	return {name, { "return " .. mat_str,
			-- get the number of times this weights are being shared
			-- between different components
			trainer:weights(name):get_shared_count() } }
      end):
  table()
  -- the loss
  table.insert(result, { LOSS_STR,  "return " .. loss_matrix:to_lua_string() })
  return result
end

-- receive a key and an array of values, and produces a pair of strings
-- key,value (or able to be string-converted by Lua) pairs
local function mreduce(key,values)
  util.omp_set_num_threads(1)
  if key == LOSS_STR then
    -- the loss
    local loss_matrix = load(values[1])()
    for i=2,#values do loss_matrix:axpy(1.0, load(values[i])()) end
    loss_matrix:scal(1/#values)
    return key, "return " .. loss_matrix:to_lua_string()
  else
    -- the gradients
    local N = values[1][2] -- accumulate here the shared count
    local g = load(values[1][1])()
    for i=2,#values do
      local v = load(values[i][1])()
      g:axpy(1.0, v)
      N = N + values[i][2]
    end
    return key, { "return " .. g:to_lua_string(), N }
  end
end

-- receives a dictionary of [key]=>value, produces a value which is shared
-- between all workers, and shows the result on user screen
local function sequential(list)
  util.omp_set_num_threads(4)
  epoch = epoch + 1
  local m = common.cache("VALIDATION_MATRIX",
			 function()
			   return ImageIO.read(data[1][1]):
			   to_grayscale():
			   invert_colors():
			   matrix()
			 end)
  local in_ds,out_ds = table.unpack(common.cache("VALIDATION_DATASET",
						 function()
						   return
						     table.pack(load_dataset_from_offset_and_steps(m, {1280,0}, {20,10}))
						 end))
  local loss_matrix  = load(list[LOSS_STR])()
  local weights_grad = iterator(pairs(list)):
  filter(function(key,value) return key ~= LOSS_STR end):
  map(function(key,value)
	-- set the number of times this weights are being shared between
	-- different components
	trainer:weights(key):set_shared_count(value[2])
	return key,load(value[1])()
      end):
  table()
  --
  optimizer:execute(function() return weights_grad, bunch_size, loss_matrix end,
		    trainer:get_weights_table())
  --
  -- trainer:save(string.format("net-%04d.lua", epoch), "ascii")
  --
  -- validation
  val_loss = trainer:validate_dataset{
    input_dataset  = in_ds,
    output_dataset = out_ds
  }
  if val_loss < best_val_loss then
    best_val_loss,best_epoch = val_loss,epoch
  end
  -- print training detail
  tr_loss = loss_matrix:sum()/loss_matrix:dim(1)
  print(epoch, tr_loss, val_loss)
  -- returns the weights list, which will be loaded at share function
  return common.share_trainer_weights(trainer)
end

-- this function receives the shared list returned by sequential function
local function share(list)
  common.load_trainer_weights(trainer, list)
end

-- Check for running, return true for continue, false for stop
local stop_criterion =
  trainable.stopping_criteria.make_max_epochs_wo_imp_relative(2)
local function loop()
  return epoch < min_epochs or
    not stop_criterion({
			 current_epoch  = epoch,
			 best_epoch     = best_epoch,
			 best_val_error = best_val_loss,
			 train_error    = tr_loss,
			 validate_error = val_loss,
		       })
end

return {
  name="EXAMPLE",
  data=data,
  decode=decode,
  split=split,
  map=mmap,
  reduce=mreduce,
  sequential=sequential,
  share=share,
  loop=loop,
}
