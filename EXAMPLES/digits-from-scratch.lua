digits_image = ImageIO.read(string.get_path(arg[0]).."digits.png")
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
				       bunch_size)
trainer:build()
--
trainer:set_option("learning_rate", 0.001)
trainer:set_option("momentum", 0.01)
trainer:set_option("weight_decay", 1e-04)
trainer:set_option("L1_norm", 1e-05)
-- we avoid weight_decay in bias
trainer:set_layerwise_option("b.", "weight_decay", 0)
trainer:set_layerwise_option("b.", "L1_norm", 0)

trainer:randomize_weights{
  random      = random(52324),
  use_fanin   = true,
  use_fanout  = true,
  inf         = -1,
  sup         =  1,
}

trainer:save("jarl.net", "binary")

training_data = {
  input_dataset  = train_input,
  output_dataset = train_output,
  shuffle        = random(25234),
  bunch_size     = bunch_size,
}

validation_data = {
  input_dataset  = val_input,
  output_dataset = val_output,
  bunch_size     = bunch_size,
}


local weight_grads = {}
loss = ann.loss.multi_class_cross_entropy()
opt  = ann.optimizer.sgd()
opt:set_option("learning_rate", 0.001)
opt:set_option("momentum", 0.01)
opt:set_option("weight_decay", 1e-04)
opt:set_option("L1_norm", 1e-05)
opt:set_layerwise_option("b1", "weight_decay", 0)
opt:set_layerwise_option("b2", "weight_decay", 0)
opt:set_layerwise_option("b1", "L1_norm", 0)
opt:set_layerwise_option("b2", "L1_norm", 0)
local train = function(data)
  loss:reset()
  for input,target in trainable.dataset_pair_iterator(data) do
    local tr_loss,_,_,tr_matrix =
      opt:execute(function(it)
		    thenet:reset(it)
		    local out = thenet:forward(input)
		    local tr_loss,tr_matrix = loss:compute_loss(out,target)
		    thenet:backprop(loss:gradient(out,target))
		    iterator(pairs(weight_grads)):select(2):call('zeros')
		    weight_grads = thenet:compute_gradients(weight_grads)
		    return tr_loss,weight_grads,tr_matrix:dim(1),tr_matrix
		  end, thenet:copy_weights())
    loss:accum_loss(tr_loss,tr_matrix)
  end
  return loss:get_accum_loss()
end
local validate = function(data)
  loss:reset()
  for input,target in trainable.dataset_pair_iterator(data) do
    local out = thenet:forward(input)
    loss:accum_loss(loss:compute_loss(out,target))
  end
  return loss:get_accum_loss()  
end
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
			   local tr_loss,_ = train(training_data)
			   local va_loss,_ = validate(validation_data)
			   return thenet,tr_loss,va_loss
			 end) do
  print(train_func:get_state_string())
end
clock:stop()
cpu,wall   = clock:read()
num_epochs = train_func:get_state_table().current_epoch
printf("# Wall total time: %.3f    per epoch: %.3f\n", wall, wall/num_epochs)
printf("# CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/num_epochs)
