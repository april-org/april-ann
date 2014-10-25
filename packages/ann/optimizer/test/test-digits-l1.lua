mathcore.set_use_cuda_default(util.is_cuda_available())
--
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
L1_norm        = 0.001
max_epochs     = 10

-- training and validation
errors = {
  {2.3165729, 2.2201636},
  {2.1821389, 2.0622385},
  {1.9339662, 1.8144050},
  {1.8861914, 1.7708337},
  {1.8548096, 1.9131246},
  {1.7150941, 1.7762133},
  {1.7553701, 1.7308848},
  {1.6740500, 1.4985144},
  {1.3345400, 1.2828860},
  {1.4751233, 1.1786772},
}
epsilon = 1e-04

--------------------------------------------------------------

m1 = ImageIO.read(string.get_path(arg[0]) .. "../../ann/test/digits.png"):to_grayscale():invert_colors():matrix()
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
trainer = trainable.supervised_trainer(thenet,
				       ann.loss.multi_class_cross_entropy(10),
				       bunch_size)
trainer:build()

trainer:set_option("learning_rate", learning_rate)
trainer:set_option("momentum",      momentum)
trainer:set_option("L1_norm",       L1_norm)
-- bias has weight_decay of ZERO
trainer:set_layerwise_option("b.", "L1_norm", 0)

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
  errortrain,vartrain  = trainer:train_dataset(datosentrenar)
  errorval,varval      = trainer:validate_dataset(datosvalidar)
  printf("%4d  %.7f %.7f :: %.7f %.7f\n",
  	 totalepocas,errortrain,errorval,vartrain,varval)
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
--printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
--printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
-- print("Test passed! OK!")

for wname,w in trainer:iterate_weights("w.*") do
  local v = w:clone():abs():min()
  assert(v == 0 or v > L1_norm)
  -- print(w:clone():eq(0.0):sum(), w:size())
end
