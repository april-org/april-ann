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
errors = matrix.fromString[[10 2
ascii
2.7256057 18.605270
14.769267 5.830131
5.7847500 4.2319937
3.9067090 3.0507331
2.7962668 1.0699219
1.0011752 0.8399307
0.6373432 0.5586927
0.3391171 0.3530262
0.2058428 0.2627542
0.1379610 0.2229893
]]
epsilon = 1e-02

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
if util.is_cuda_available() then thenet:set_use_cuda(true) end
trainer = trainable.supervised_trainer(thenet,
				       ann.loss.multi_class_cross_entropy(10),
				       bunch_size,
				       ann.optimizer.rprop())
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
  errortrain,vartrain  = trainer:train_dataset(datosentrenar)
  errorval,varval      = trainer:validate_dataset(datosvalidar)
  printf("%4d  %.7f %.7f :: %.7f %.7f\n",
  	 totalepocas,errortrain,errorval,vartrain,varval)
  if math.abs(errortrain - errors:get(epoch,1)) > epsilon then
    error(string.format("Training error %g is not equal enough to "..
  			  "reference error %g",
  			errortrain, errors:get(epoch,1)))
  end
  if math.abs(errorval - errors:get(epoch,2)) > epsilon then
    error(string.format("Validation error %g is not equal enough to "..
  			  "reference error %g",
  			errorval, errors:get(epoch,2)))
  end
end

clock:stop()
cpu,wall = clock:read()
--printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
--printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
-- print("Test passed! OK!")
