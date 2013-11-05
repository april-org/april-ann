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

bunch_size = 64
thenet = ann.mlp.all_all.generate("256 inputs 128 tanh 10 log_softmax")
if util.is_cuda_available() then thenet:set_use_cuda(true) end
trainer = trainable.supervised_trainer(thenet,
				       ann.loss.multi_class_cross_entropy(10),
				       bunch_size,
				       ann.optimizer.rprop())
trainer:build()
trainer:randomize_weights{
  random      = random(52324),
  use_fanin   = true,
  use_fanout  = true,
  inf         = -1,
  sup         =  1,
}
trainer:set_option("learning_rate", 0.1)
trainer:set_option("momentum",      0.01)
trainer:set_option("weight_decay",  1e-05)
-- bias has weight_decay of ZERO
trainer:set_layerwise_option("b.", "weight_decay", 0)

trainer:save("jarl.net", "binary")

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
print("# Epoch Training  Validation")
stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_relative(2)
result = trainer:train_holdout_validation{ training_table     = training_data,
					   validation_table   = validation_data,
					   min_epochs         = 20,
					   max_epochs         = 1000,
					   stopping_criterion = stopping_criterion,
					   update_function    =
					     function(t) printf("%4d %.6f %.6f (%4d %.6f)\n",
								t.current_epoch,
								t.train_error,
								t.validation_error,
								t.best_epoch,
								t.best_val_error) end }
clock:stop()
cpu,wall   = clock:read()
num_epochs = result.last_epoch
local val_error,val_variance=result.best:validate_dataset{
  input_dataset  = val_input,
  output_dataset = val_output,
  loss           = ann.loss.zero_one(10)
}
printf("# Wall total time: %.3f    per epoch: %.3f\n", wall, wall/num_epochs)
printf("# CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/num_epochs)
printf("# Validation error: %f  +-  %f\n", val_error, val_variance)
