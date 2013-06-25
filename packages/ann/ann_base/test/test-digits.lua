-- un generador de valores aleatorios... y otros parametros
bunch_size     = tonumber(arg[1]) or 64
semilla        = 1234
weights_random = random(semilla)
description    = "256 inputs 256 tanh 128 tanh 10 log_softmax"
inf            = -1
sup            =  1
shuffle_random = random(5678)
learning_rate  = 0.08
momentum       = 0.01
weight_decay   = 1e-05
max_epochs     = 10

-- training and validation
errors = {
  {2.2699234, 2.0364990},
  {1.6727996, 1.2317111},
  {0.9137920, 0.5743981},
  {0.5094233, 0.3587638},
  {0.3098969, 0.2858882},
  {0.2155668, 0.1911811},
  {0.1612766, 0.1525298},
  {0.1280116, 0.1290455},
  {0.1094717, 0.1413695},
  {0.0995687, 0.1391480}
}
epsilon = 1e-04

--------------------------------------------------------------

m1 = ImageIO.read("test/digits.png"):to_grayscale():invert_colors():matrix()
train_input = dataset.matrix(m1,
			     {
			       patternSize = {16,16},
			       offset      = {0,0},
			       numSteps    = {80,10},
			       stepSize    = {16,16},
			       orderStep   = {1,0}
			     })

val_input  = dataset.matrix(m1,
			    {
			      patternSize = {16,16},
			      offset      = {1280,0},
			      numSteps    = {20,10},
			      stepSize    = {16,16},
			      orderStep   = {1,0}
			    })
-- una matriz pequenya la podemos cargar directamente
m2 = matrix(10,{1,0,0,0,0,0,0,0,0,0})

-- ojito con este dataset, fijaros que usa una matriz de dim 1 y talla
-- 10 PERO avanza con valor -1 y la considera CIRCULAR en su unica
-- dimension

train_output = dataset.matrix(m2,
			      {
				patternSize = {10},
				offset      = {0},
				numSteps    = {800},
				stepSize    = {-1},
				circular    = {true}
			      })

val_output   = dataset.matrix(m2,
			      {
				patternSize = {10},
				offset      = {0},
				numSteps    = {200},
				stepSize    = {-1},
				circular    = {true}
			      })


thenet = ann.mlp.all_all.generate(description)
thenet:set_option("learning_rate", learning_rate)
thenet:set_option("momentum",      momentum)
thenet:set_option("weight_decay",  weight_decay)
trainer = trainable.supervised_trainer(thenet,
				       ann.loss.multi_class_cross_entropy(10),
				       bunch_size)
trainer:build()
trainer:randomize_weights{
  random      = weights_random,
  inf         = inf,
  sup         = sup,
  use_fanin   = true,
}

-- datos para entrenar
datosentrenar = {
  input_dataset  = train_input,
  output_dataset = train_output,
  shuffle        = shuffle_random,
}

datosvalidar = {
  input_dataset  = val_input,
  output_dataset = val_output,
}

totalepocas = 0

errorval = trainer:validate_dataset(datosvalidar)
-- print("# Initial validation error:", errorval)

clock = util.stopwatch()
clock:go()

-- print("Epoch Training  Validation")
for epoch = 1,max_epochs do
  collectgarbage("collect")
  totalepocas = totalepocas+1
  errortrain  = trainer:train_dataset(datosentrenar)
  errorval    = trainer:validate_dataset(datosvalidar)
  --  printf("%4d  %.7f %.7f\n",
  --	 totalepocas,errortrain,errorval)
  if math.abs(errortrain - errors[epoch][1]) > epsilon then
    error(string.format("Training error %g is not equal enough to "..
			  "reference error %g",
			errortrain, errors[epoch][1]))
  end
  if math.abs(errorval - errors[epoch][2]) > epsilon then
    error(string.format("Validation error %g is not equal enough to "..
			  "reference error %g",
			errorval, errors[epoch][2]))
  end
end

clock:stop()
cpu,wall = clock:read()
-- printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
-- printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
-- print("Test passed! OK!")
