digits_image = ImageIO.read(string.get_path(arg[0]).."../../../EXAMPLES/digits.png")
m1           = digits_image:to_grayscale():invert_colors():matrix()

-- TRAINING --
-- a simple matrix for the desired output
train_input  = dataset.matrix(m1,
			      {
				patternSize = {16,16},
				offset      = {0,0},
				numSteps    = {80,10},
				stepSize    = {16,16},
				orderStep   = {1,0}
			      })
m2 = matrix(10,{1,0,0,0,0,0,0,0,0,0})
-- a circular dataset which advances with step -1
train_output = dataset.matrix(m2,
			      {
				patternSize = {10},
				offset      = {0},
				numSteps    = {800},
				stepSize    = {-1},
				circular    = {true}
			      })
-- VALIDATION --
val_input = dataset.matrix(m1,
			   {
			     patternSize = {16,16},
			     offset      = {1280,0},
			     numSteps    = {20,10},
			     stepSize    = {16,16},
			     orderStep   = {1,0}
			   })
val_output   = dataset.matrix(m2,
			      {
				patternSize = {10},
				offset      = {0},
				numSteps    = {200},
				stepSize    = {-1},
				circular    = {true}
			      })

bunch_size = 32
thenet = ann.mlp.all_all.generate("256 inputs 128 tanh 10 log_softmax")
if util.is_cuda_available() then thenet:set_use_cuda(true) end
trainer = trainable.supervised_trainer(thenet,
				       ann.loss.multi_class_cross_entropy(),
				       bunch_size,
                                       ann.optimizer.adadelta(),
                                       true, -- smooth parameter
                                       15)    -- max gradients norm parameter
trainer:build()
--
trainer:set_option("weight_decay", 1e-04)
-- we avoid weight_decay in bias
trainer:set_layerwise_option("b.", "weight_decay", 0)

trainer:randomize_weights{
  random      = random(52324),
  use_fanin   = true,
  use_fanout  = true,
  inf         = -1,
  sup         =  1,
}

training_data = {
  input_dataset  = train_input,
  output_dataset = train_output,
  shuffle        = random(25234),
}

validation_data = {
  input_dataset  = val_input,
  output_dataset = val_output,
}

clock = util.stopwatch()
clock:go()
--print("# Epoch Training  Validation")
-- print("Epoch Training  Validation")
train_func = trainable.train_holdout_validation{
  min_epochs=10,
  max_epochs = 20,
  stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(1.1),
  tolerance = 0.1,
}
-- training loop
tmpname = os.tmpname()
while train_func:execute(function()
			   local tr = trainer:train_dataset(training_data)
			   local va = trainer:validate_dataset(validation_data)
                           print(tr,va)
			   return trainer,tr,va
			 end) do
  util.serialize({ train_func, training_data.shuffle }, tmpname)
  train_func,training_data.shuffle = table.unpack((util.deserialize(tmpname)))
end
util.serialize(train_func, tmpname)
train_func = util.deserialize(tmpname)
os.remove(tmpname)
os.remove(tmpname..".bak")
clock:stop()
cpu,wall = clock:read()
num_epochs = train_func:get_state_table().current_epoch
clock:stop()
cpu,wall   = clock:read()
best       = train_func:get_state_table().best
local val_error,val_variance=best:validate_dataset{
  input_dataset  = val_input,
  output_dataset = val_output,
  loss           = ann.loss.zero_one(10)
}
local out_ds = best:use_dataset{ input_dataset = val_input }
best:use_dataset{ input_dataset = val_input, output_dataset = out_ds }
for input,bunch_indexes in trainable.dataset_multiple_iterator{
  datasets = { val_input }, bunch_size = 1 } do
  local out = best:calculate(input)
  local target = matrix(out:dim(1),out:dim(2),
			out_ds:getPattern(bunch_indexes[1]))
  assert(out:equals(target))
end
--printf("# Wall total time: %.3f    per epoch: %.3f\n", wall, wall/num_epochs)
--printf("# CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/num_epochs)
--printf("# Validation error: %f  +-  %f\n", val_error, val_variance)



local img = ann.connections.input_filters_image(best:weights("w1"), {16,16})
ImageIO.write(img, "/tmp/filters.png")
util.serialize(trainer, "wop.lua")
