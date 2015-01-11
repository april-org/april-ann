mathcore.set_use_cuda_default(util.is_cuda_available())
--
local bunch_size       = tonumber(arg[1]) or 256
local semilla          = 1234
local weights_random   = random(semilla)
local inf              = -2.4
local sup              =  2.4
local shuffle_random   = random(5678)
local weight_decay     = 0.0001
local max_epochs       = 400
local hidden_size      = 1024
local rho              = 0.01
local beta             = 3

--------------------------------------------------------------

local m1 = ImageIO.read(string.get_path(arg[0]) .. "digits.png"):to_grayscale():invert_colors():matrix()
local train_input = dataset.matrix(m1,
                                   {
                                     patternSize = {16,16},
                                     offset      = {0,0},
                                     numSteps    = {80,10},
                                     stepSize    = {16,16},
                                     orderStep   = {1,0}
})

local val_input  = dataset.matrix(m1,
                                  {
                                    patternSize = {16,16},
                                    offset      = {1280,0},
                                    numSteps    = {20,10},
                                    stepSize    = {16,16},
                                    orderStep   = {1,0}
})
-- una matriz pequenya la podemos cargar directamente
local m2 = matrix(10,{1,0,0,0,0,0,0,0,0,0})

-- ojito con este dataset, fijaros que usa una matriz de dim 1 y talla
-- 10 PERO avanza con valor -1 y la considera CIRCULAR en su unica
-- dimension

local train_output = dataset.matrix(m2,
                                    {
                                      patternSize = {10},
                                      offset      = {0},
                                      numSteps    = {800},
                                      stepSize    = {-1},
                                      circular    = {true}
})

local val_output   = dataset.matrix(m2,
                                    {
                                      patternSize = {10},
                                      offset      = {0},
                                      numSteps    = {200},
                                      stepSize    = {-1},
                                      circular    = {true}
})

local thenet = ann.components.stack():
  push( ann.components.hyperplane{ input=train_input:patternSize(),
                                   output=hidden_size,
                                   bias_weights="b1",
                                   -- shared weights
                                   dot_product_weights="w" } ):
  push( ann.components.actf.sparse_logistic{ rho=rho, beta=beta } ):
  push( ann.components.hyperplane{ input=hidden_size,
                                   output=train_input:patternSize(),
                                   bias_weights="b2",
                                   -- shared weights
                                   dot_product_weights="w",
                                   transpose = true} ):
  push( ann.components.actf.log_logistic() )

local trainer = trainable.supervised_trainer(thenet,
                                             ann.loss.cross_entropy(),
                                             bunch_size,
                                             ann.optimizer.adadelta())
trainer:build()

trainer:set_option("weight_decay", weight_decay)
trainer:set_layerwise_option("b.", "weight_decay", 0.0)

trainer:randomize_weights{
  random      = weights_random,
  inf         = inf,
  sup         = sup,
  use_fanin   = true,
  use_fanout  = true,
}

-- datos para entrenar
local datosentrenar = {
  input_dataset  = train_input,
  output_dataset = train_input,
  shuffle        = shuffle_random,
}

local datosvalidar = {
  input_dataset  = val_input,
  output_dataset = val_input,
}

local totalepocas = 0
local errorval = trainer:validate_dataset(datosvalidar)
print("# Initial validation error:", errorval)

local clock = util.stopwatch()
clock:go()

-- print("Epoch Training  Validation")
for epoch = 1,max_epochs do
  collectgarbage("collect")
  totalepocas = totalepocas+1
  local errortrain  = trainer:train_dataset(datosentrenar)
  local errorval    = trainer:validate_dataset(datosvalidar)
  --
  printf("%4d  %.7f %.7f\n", totalepocas, errortrain, errorval)
end

local img = ann.connections.input_filters_image(trainer:weights("w"), {16,16})
ImageIO.write(img,"/tmp/filters.sparse.png")

clock:stop()
local cpu,wall = clock:read()
printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
