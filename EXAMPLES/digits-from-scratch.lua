digits_image = ImageIO.read(string.get_path(arg[0]).."digits.png")
m1           = digits_image:to_grayscale():invert_colors():matrix()
--
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
--
bunch_size = 32
thenet         = ann.mlp.all_all.generate("256 inputs 128 tanh 10 log_softmax")
connections    = thenet:build()
weights_random = random(52324)
cnn_array      = iterator(pairs(connections)):enumerate():table()
table.sort(cnn_array, function(a,b) return a[1]<b[1] end) -- sort by name
iterator(ipairs(cnn_array)):select(2):field(2):
apply(function(cnn)
	local sqrt_fan = math.sqrt(cnn:get_input_size() + cnn:get_output_size())
	cnn:randomize_weights{
	  random = weights_random,
	  inf = -1/sqrt_fan,
	  sup =  1/sqrt_fan,
	}
      end)
--
shuffle = random(25234)
training_data = {
  input_dataset  = train_input,
  output_dataset = train_output,
  shuffle        = shuffle,
  bunch_size     = bunch_size,
}
--
validation_data = {
  input_dataset  = val_input,
  output_dataset = val_output,
  bunch_size     = bunch_size,
}
--
loss = ann.loss.multi_class_cross_entropy()
opt  = ann.optimizer.sgd()
opt:set_option("learning_rate", 0.01)
opt:set_option("momentum",      0.01)
opt:set_option("weight_decay",  1e-04)
opt:set_option("L1_norm",       1e-05)
--
opt:set_layerwise_option("b1", "weight_decay", 0)
opt:set_layerwise_option("b2", "weight_decay", 0)
opt:set_layerwise_option("b1", "L1_norm",      0)
opt:set_layerwise_option("b2", "L1_norm",      0)
--
local weight_grads -- upvalue of train
local train = function(thenet,data,loss,opt)
  weight_grads = weight_grads or { } -- upvalue
  loss:reset()
  for input,target in trainable.dataset_pair_iterator(data) do
    local tr_loss,_,_,tr_matrix =
      opt:execute(function(it)
		    thenet:reset(it)
		    local out = thenet:forward(input)
		    local tr_loss,tr_matrix = loss:compute_loss(out,target)
		    if not tr_loss then return nil end
		    thenet:backprop(loss:gradient(out,target))
		    for _,w in pairs(weight_grads) do w:zeros() end
		    weight_grads = thenet:compute_gradients(weight_grads)
		    return tr_loss,weight_grads,tr_matrix:dim(1),tr_matrix
		  end,
		  connections)
    loss:accum_loss(tr_loss,tr_matrix)
  end
  local tr,_=loss:get_accum_loss()
  return tr
end
--
local validate = function(thenet,data,loss)
  loss:reset()
  for input,target in trainable.dataset_pair_iterator(data) do
    thenet:reset()
    local out = thenet:forward(input)
    loss:accum_loss(loss:compute_loss(out,target))
  end
  local va,_=loss:get_accum_loss()
  return va
end
--
clock = util.stopwatch()
clock:go()
print("# Epoch Training  Validation")
stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(2)
train_func = trainable.train_holdout_validation{
  min_epochs = 100,
  max_epochs = 1000,
  stopping_criterion = stopping_criterion
}
while train_func:execute(function()
			   local tr_loss = train(thenet,training_data,loss,opt)
			   local va_loss = validate(thenet,validation_data,loss)
			   return thenet,tr_loss,va_loss
			 end) do
  print(train_func:get_state_string())
  local state = train_func:get_state_table()
  if state.current_epoch == state.best_epoch then
    train_func:save("jjj.lua", "binary",
		    { optimizer = opt, loss = loss,
		      weights = connections, shuffle = shuffle })
  end
end
clock:stop()
cpu,wall   = clock:read()
num_epochs = train_func:get_state_table().current_epoch
printf("# Wall total time: %.3f    per epoch: %.3f\n", wall, wall/num_epochs)
printf("# CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/num_epochs)
