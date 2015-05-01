mathcore.set_use_cuda_default(util.is_cuda_available())
--
local bunch_size       = tonumber(arg[1]) or 256
local semilla          = 1234
local weights_random   = random(semilla)
local inf              = -2.4
local sup              =  2.4
local shuffle_random   = random(5678)
local learning_rate    = 1.0
local momentum         = 0.9
local weight_decay     = 0.01
local max_norm_penalty = 4
local max_epochs       = 1000
local check_grandients = false
local check_tokens     = false

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

local thenet = ann.mlp.all_all.
  generate(table.concat{"%d inputs 64 prelu$args ",
                        "64 prelu$args dropout{ prob=0.5, random=#1 } ",
                        "64 prelu$args dropout{ prob=0.5, random=#1 } ",
                        "64 prelu$args dropout{ prob=0.5, random=#1 } ",
                        "64 prelu$args dropout{ prob=0.5, random=#1 } ",
                        "10 log_softmax"}%{ train_input:patternSize(),
                                            args = "{ shared=true }"},
           { random(2392548) })

local trainer = trainable.supervised_trainer(thenet,
                                             ann.loss.multi_class_cross_entropy(10),
                                             bunch_size,
                                             ann.optimizer.adadelta())
trainer:build()

trainer:set_option("learning_rate",     learning_rate)
trainer:set_option("momentum",          momentum)
-- regularization
trainer:set_layerwise_option("w.*", "weight_decay",      weight_decay)
trainer:set_layerwise_option("w.*", "max_norm_penalty",  max_norm_penalty)

trainer:randomize_weights{
  name_match  = "w.*",
  random      = weights_random,
  inf         = inf,
  sup         = sup,
  use_fanin   = true,
  use_fanout  = true,
}
iterator(trainer:iterate_weights("b.*")):select(2):
  apply(matrix.."zeros")
iterator(trainer:iterate_weights("a.*")):select(2):
  apply(bind(matrix.."fill", nil, 0.25))

-- for _,c in trainer:iterate_components("conv-b*") do
--   c:set_option("learning_rate", 0.0001)
-- end
-- trainer:set_component_option("actf2", "dropout_factor", 0.5)

-- datos para entrenar
local datosentrenar = {
  input_dataset  = train_input,
  output_dataset = train_output,
  shuffle        = shuffle_random,
}

local datosvalidar = {
  input_dataset  = val_input,
  output_dataset = val_output,
  --  bunch_size = 128,
}

if check_grandients then
  trainer:grad_check_dataset({
			       input_dataset  = dataset.slice(val_input, 1, 10),
			       output_dataset = dataset.slice(val_output, 1, 10),
			       bunch_size = 10,
			       verbose = false,
			     })
end

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
  local norm2_w = trainer:norm2(".*w.*")
  local norm2_b = trainer:norm2(".*b.*")
  --
  printf("%4d  %.7f %.7f      %.7f %.7f\n",
  	 totalepocas,errortrain,errorval,norm2_w,norm2_b)
end

clock:stop()
local cpu,wall = clock:read()
printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)

local val_error,val_variance=trainer:validate_dataset{
  input_dataset  = val_input,
  output_dataset = val_output,
  loss           = ann.loss.zero_one(),
}
printf("# Validation error: %f  +-  %f\n", val_error, val_variance)

print(trainer:weights("a1"))
