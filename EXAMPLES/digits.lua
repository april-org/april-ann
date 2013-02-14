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

thenet = ann.mlp.all_all.generate{
  bunch_size  = 32,
  topology    = "256 inputs 128 tanh 10 softmax",
  random      = random(52324),
  use_fanin   = true,
  inf         = -1,
  sup         =  1,
}
thenet:set_option("learning_rate", 0.01)
thenet:set_option("momentum",      0.01)
thenet:set_option("weight_decay",  1e-05)
thenet:set_error_function(ann.error_functions.logistic_cross_entropy())

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
stopping_criterion = ann.stopping_criterions.make_max_epochs_wo_imp_relative(2)
result = ann.train_crossvalidation{ ann = thenet,
				    training_table     = training_data,
				    validation_table   = validation_data,
				    max_epochs         = 1000,
				    stopping_criterion = stopping_criterion,
				    update_function    =
				      function(t) printf("%4d %.6f %.6f (%4d %.6f)\n",
							 t.current_epoch,
							 t.train_error,
							 t.validation_error,
							 t.best_epoch,
							 t.best_val_error) end }
-- validation_function = 
--   function(thenet, t)
--     return thenet:validate_dataset(t)
--   end
clock:stop()
cpu,wall   = clock:read()
num_epochs = result.last_epoch
printf("# Wall total time: %.3f    per epoch: %.3f\n", wall, wall/num_epochs)
printf("# CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/num_epochs)
printf("# Validation error: %f\n", result.best_val_error)
