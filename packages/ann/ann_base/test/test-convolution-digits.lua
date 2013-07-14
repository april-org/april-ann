-- un generador de valores aleatorios... y otros parametros
bunch_size     = tonumber(arg[1]) or 64
semilla        = 1234
weights_random = random(semilla)
inf            = -0.6
sup            =  0.6
shuffle_random = random(5678)
learning_rate  = 0.1
momentum       = 0.1
weight_decay   = 0 --1e-05
max_epochs     = 100
max_norm_penalty = 2

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


thenet = ann.components.stack():
push(ann.components.rewrap{ size={ 1, 16, 16 } }):
push(ann.components.convolution{ name="t", weights="w", kernel={1, 3, 3}, n=20 }):
-- push(ann.components.max_pooling{ kernel={1,7,7} }):
push(ann.components.actf.hardtanh()):
push(ann.components.convolution{ kernel={20, 7, 7}, n=40 }):
-- push(ann.components.max_pooling{ kernel={1,3,3} }):
push(ann.components.actf.hardtanh{ name="actf1" }):
push(ann.components.flatten()):
push(ann.components.hyperplane{ input=((16-3-7+2)^2)*40, output= 1000 }):
push(ann.components.actf.tanh{ name="actf2" }):
push(ann.components.hyperplane{ input=1000, output= 10 }):
push(ann.components.actf.log_softmax())

thenet:set_option("learning_rate", learning_rate)
thenet:set_option("momentum",      momentum)
thenet:set_option("weight_decay",  weight_decay)
thenet:set_option("max_norm_penalty",  max_norm_penalty)
trainer = trainable.supervised_trainer(thenet,
				       ann.loss.multi_class_cross_entropy(10),
				       bunch_size)
trainer:build()
trainer:randomize_weights{
  random      = weights_random,
  inf         = inf,
  sup         = sup,
  use_fanin   = true,
  use_fanout  = true,
}
trainer:component("actf2"):set_option("dropout_factor", 0.5)

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


trainer:for_each_pattern{
  input_dataset = datosvalidar.input_dataset,
  bunch_size    = 1,
  func = function(idxs, trainer)
    local c = trainer:component("t")
    local o = c:get_output():get_matrix()
    local d = o:dim()
    local k = 0
    for w in o:sliding_window{ size={1,1,d[3],d[4]}, step={1,1,1,1},
			       numSteps={d[1], d[2], 1, 1} }:iterate() do
      local img = w:clone():rewrap(d[3]*d[4]):clone("row_major"):rewrap(d[3],d[4])
      matrix.saveImage(img:adjust_range(0,1), "/tmp/WW-".. idxs[1] .. "-"..k..".pnm")
      k=k+1
    end
  end
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
  printf("%4d  %.7f %.7f\n",
  	 totalepocas,errortrain,errorval)
  thenet:set_option("learning_rate",
		    thenet:get_option("learning_rate")*0.95)
end

trainer:for_each_pattern{
  input_dataset = datosvalidar.input_dataset,
  bunch_size    = 1,
  func = function(idxs, trainer)
    local c = trainer:component("t")
    local o = c:get_output():get_matrix()
    local d = o:dim()
    local k = 0
    for w in o:sliding_window{ size={1,1,d[3],d[4]}, step={1,1,1,1},
			       numSteps={d[1], d[2], 1, 1} }:iterate() do
      local img = w:clone():rewrap(d[3]*d[4]):clone("row_major"):rewrap(d[3],d[4])
      matrix.saveImage(img:adjust_range(0,1), "/tmp/jajaja-".. idxs[1] .. "-"..k..".pnm")
      k=k+1
    end
  end
}

w = trainer:weights("w"):matrix()
local k=0
for waux in w:sliding_window():iterate() do
  matrix.saveImage(waux:scal(1/waux:norm2()):clone("row_major"):rewrap(3,3):adjust_range(0,1),
		   "KK-"..k..".pnm")
  k=k+1
end




clock:stop()
cpu,wall = clock:read()
printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
