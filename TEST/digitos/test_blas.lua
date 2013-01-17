-- un generador de valores aleatorios... y otros parametros
bunch_size    = tonumber(arg[1]) or 64
semilla       = 1234
aleat         = random(semilla)
description   = "256 inputs 1024 tanh 10 softmax"
inf           = -0.1
sup           =  0.1
otrorand      = random(5678)
learning_rate = 0.01
momentum      = 0.0
weight_decay  = 0.0
max_epochs    = 10

--------------------------------------------------------------

m1 = matrix.loadImage("digits.png", "gray")
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


lared = ann.mlp.all_all.generate{
  bunch_size  = bunch_size,
  topology    = description,
  random      = aleat,
  inf         = inf,
  sup         = sup,
}
--lared = ann.mlp.all_all.load("prueba.net", bunch_size)

-- datos para entrenar
datosentrenar = {
  input_dataset  = train_input,
  output_dataset = train_output,
  shuffle        = otrorand,
}

datosvalidar = {
  input_dataset  = val_input,
  output_dataset = val_output,
}

lared:set_option("learning_rate", learning_rate)
lared:set_option("momentum",      momentum)
lared:set_option("weight_decay",  weight_decay)

lared:set_error_function(ann.error_functions.mse())

totalepocas = 0

clock = util.stopwatch()
clock:go()

print("Epoch Training  Validation")
for epoch = 1,max_epochs do
  totalepocas = totalepocas+1
  errortrain  = lared:train_dataset(datosentrenar)
  errorval    = lared:validate_dataset(datosvalidar)
  printf("%4d  %.7f %.7f\n",
	 totalepocas,errortrain,errorval)
end

clock:stop()
cpu,wall = clock:read()
printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
ann.mlp.all_all.save(lared, "red_blas.net", "ascii", "old")
