-- arg is available as a vararg argument when the script is loaded by loadfile
-- or load functions. Please, use arg to parse arguments with cmdOpt.
-- local arg = { ... }

require "common"
-- NOTE that common, worker, and master are tables used by MapReduce paradigm,
-- also note that map and reduce functions are defined by April-ANN, so use
-- other names

util.omp_set_num_threads(1)

local bunch_size     = 12
local semilla        = 1234
local weights_random = random(semilla)
local description    = "256 inputs 256 tanh 128 tanh 10 log_softmax"
local inf            = -1
local sup            =  1
local shuffle_random = random(5678)
local learning_rate  = 0.08
local momentum       = 0.01
local weight_decay   = 1e-05
local max_epochs     = 100
local LOSS_STR       = "LOSS"

local thenet = ann.mlp.all_all.generate(description)
thenet:set_option("learning_rate", learning_rate)
thenet:set_option("momentum",      momentum)
thenet:set_option("weight_decay",  weight_decay)
local trainer = trainable.supervised_trainer(thenet,
					     ann.loss.multi_class_cross_entropy(10),
					     bunch_size)
trainer:build()
trainer:randomize_weights{
  random      = weights_random,
  inf         = inf,
  sup         = sup,
  use_fanin   = true,
}

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
  return in_ds, out_ds
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
  local in_ds,out_ds = table.unpack(common.cache(key,
						 function()
						   return load_dataset_from_value(value)
						 end))
  
  local data_loss  = trainer:train_dataset{
    input_dataset  = in_ds,
    output_dataset = out_ds,
    shuffle        = shuffle_random,
  }
  -- the weights
  local result = common.map_trainer_weights(trainer)
  -- the loss
  table.insert(result, { LOSS_STR, data_loss })
  return result
end

-- receive a key and an array of values, and produces a pair of strings
-- key,value (or able to be string-converted by Lua) pairs
local function mreduce(key,values)
  if key == LOSS_STR then
    -- the loss
    return key, iterator(ipairs(values)):select(2):reduce(math.add(),0)
  else
    -- the weights
    return key, common.reduce_trainer_weights(values)
  end
end

local count = 0
-- receives a dictionary of [key]=>value, produces a value which is shared
-- between all workers, and shows the result on user screen
local function sequential(list)
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
  common.load_trainer_weights(trainer, list)
  
  -- validation
  local val_loss = trainer:validate_dataset{
    input_dataset  = in_ds,
    output_dataset = out_ds
  }
  count = count + 1
  -- print training detail
  print(count, list[LOSS_STR], val_loss)
  -- returns the weights list, which will be loaded at share function
  return list
end

-- this function receives the shared list returned by sequential function
local function shared(list)
  common.load_trainer_weights(trainer, list)
end

-- Check for running, return true for continue, false for stop
local function loop()
  return count < max_epochs
end

return {
  name="EXAMPLE",
  data=data,
  decode=decode,
  split=split,
  map=mmap,
  reduce=mreduce,
  sequential=sequential,
  shared=shared,
  loop=loop,
}
