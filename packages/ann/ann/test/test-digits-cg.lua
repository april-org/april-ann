-- un generador de valores aleatorios... y otros parametros
bunch_size     = tonumber(arg[1]) or 512
semilla        = 1234
weights_random = random(semilla)
description    = "256 inputs 256 tanh 128 tanh 10 log_softmax"
inf            = -1
sup            =  1
shuffle_random = random(5678)
rho            = 0.001
sig            = 0.8
weight_decay   = 1e-05
max_epochs     = 10

-- training and validation
errors = {
  {0.1754176, 0.2544766},
  {0.0198613, 0.1189429},
  {0.0038684, 0.0996554},
  {0.0014355, 0.0770654},
  {0.0004041, 0.0775090},
  {0.0002285, 0.0777847},
  {0.0000280, 0.0713880},
  {0.0000100, 0.0884530},
  {0.0000044, 0.0916537},
  {0.0000033, 0.0902332},
}
epsilon = 1e-03

--------------------------------------------------------------

m1 = ImageIO.read(string.get_path(arg[0]) .. "digits.png"):to_grayscale():invert_colors():matrix()
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
if util.is_cuda_available() then thenet:set_use_cuda(true) end
trainer = trainable.supervised_trainer(thenet,
				       ann.loss.multi_class_cross_entropy(10),
				       bunch_size,
				       ann.optimizer.cg())
trainer:build()

trainer:set_option("rho", rho)
trainer:set_option("sig", sig)
trainer:set_option("weight_decay",  weight_decay)
-- bias has weight_decay of ZERO
trainer:set_layerwise_option("b.", "weight_decay", 0)

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
