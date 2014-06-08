-- un generador de valores aleatorios... y otros parametros
local bunch_size     = 1024
local weights_random = random(1234)
local description    = "256 inputs 128 tanh 10 log_softmax"
local inf            = -0.1
local sup            =  0.1
local shuffle_random = random(5678)

local rnd = random(123824)
local burnin = 2000
local max_epochs = 1000
local nchains = 1

--------------------------------------------------------------

local m1 = ImageIO.read(string.get_path(arg[0]) .. "../../../TEST/digitos/digits.png"):to_grayscale():invert_colors():matrix()
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

local thenet = ann.mlp.all_all.generate(description)
local trainer = trainable.supervised_trainer(thenet,
                                             ann.loss.multi_class_cross_entropy(),
                                             bunch_size,
                                             bayesian.optimizer.hmc())
trainer:build()
trainer:set_option("nsteps",      80)
trainer:set_option("var",         60)
trainer:set_option("seed",     74967)
trainer:set_option("epsilon",  0.002)

local trainers = {}
for i=1,nchains do
  local tr = trainer:clone()
  tr:randomize_weights{
    random      = weights_random,
    inf         = inf,
    sup         = sup,
    use_fanin   = true,
  }
  table.insert(trainers, tr)
end

-- datos para entrenar
local datosentrenar = {
  input_dataset  = train_input,
  output_dataset = train_output,
  shuffle        = shuffle_random,
  replacement    = bunch_size,
}

local datosvalidar = {
  input_dataset  = val_input,
  output_dataset = val_output,
  bunch_size = 32,
}

local totalepocas = 0

-- local errorval = trainer:validate_dataset(datosvalidar)
-- print("# Initial validation error:", errorval)

local clock = util.stopwatch()
clock:go()

for j=1,#trainers do
  trainers[j]:get_optimizer():start_burnin()
end
for i=1,burnin do
  for j=1,#trainers do
    trainers[j]:train_dataset(datosentrenar)
    print(trainers[j]:get_optimizer():get_state_string())
  end
end
for j=1,#trainers do
  trainers[j]:get_optimizer():finish_burnin()
end

for i=1,max_epochs do
  for j=1,#trainers do
    trainers[j]:train_dataset(datosentrenar)
    print(trainers[j]:get_optimizer():get_state_string())
  end
end

for j=1,#trainers do
  local thenet = trainers[j]:get_component()
  local hmc = trainers[j]:get_optimizer()
  local bayesian_model = bayesian.build_bayes_comb{
    forward = function(weights, input)
      thenet:build{ weights = weights }
      return thenet:forward(input):get_matrix()
    end,
    N=100,
    shuffle=rnd,
    samples=hmc:get_samples(),
  }
  local bayesian_trainer = trainable.supervised_trainer(bayesian_model, 
                                                        ann.loss.zero_one(),
                                                        32)
  bayesian_trainer:build()
  print("TR", bayesian_trainer:validate_dataset({
                                                  input_dataset=datosentrenar.input_dataset,
                                                  output_dataset=datosentrenar.output_dataset,
                                                }))
  print("VA", bayesian_trainer:validate_dataset(datosvalidar))
end

clock:stop()
local cpu,wall = clock:read()
printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
-- print("Test passed! OK!")
