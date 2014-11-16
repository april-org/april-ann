mathcore.set_use_cuda_default(util.is_cuda_available())
--
local check = utest.check
local T = utest.test
--
T("SGDConvexTest", function()
    local opt = ann.optimizer.adadelta()
    -- optimize quadractic function: f(x) = 3*x^2 - 2*x + 10
    local function f(x) return (3*x^2 - 2*x + 10):sum() end
    local function df_dx(x) return 6*x - 2 end
    -- df(x)/dx = 6*x - 2
    -- the minimum is in x=0.333
    local x = matrix(1,1,{-100})
    for i=1,20000 do
      opt:execute(function() return f(x),{df_dx(x)} end, {x})
    end
    check.eq(x, matrix(1,1,{0.333}))
end)

T("SGDTestDigits", function()
    -- un generador de valores aleatorios... y otros parametros
    bunch_size     = tonumber(arg[1]) or 64
    semilla        = 1234
    weights_random = random(semilla)
    description    = "256 inputs 256 tanh 128 tanh 10 log_softmax"
    inf            = -0.1
    sup            =  0.1
    shuffle_random = random(5678)
    weight_decay   = 0.001
    max_epochs     = 10

    -- training and validation
    errors = {
      {2.3053319, 2.3015521},
      {2.1570034, 1.8272703},
      {1.7826253, 1.9765174},
      {1.4031528, 1.0635141},
      {1.2507044, 0.9257209},
      {0.7426600, 0.7414427},
      {0.7360206, 0.5097144},
      {0.6020166, 0.4537252},
      {0.3055784, 0.3659463},
      {0.2493757, 0.3132612},
    }
    epsilon = 0.01 -- 1% relative difference

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
                                           bunch_size,
                                           ann.optimizer.adadelta())
    trainer:build()

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
    local weights = trainer.weights_table
    -- print("Epoch Training  Validation")
    local tmp = os.tmpname()
    for epoch = 1,max_epochs do
      collectgarbage("collect")
      totalepocas = totalepocas+1
      errortrain,vartrain  = trainer:train_dataset(datosentrenar)
      errorval,varval      = trainer:validate_dataset(datosvalidar)
      trainer:save(tmp)
      trainer = trainable.supervised_trainer.load(tmp)
      printf("%4d  %.7f %.7f :: %.7f %.7f\n",
             totalepocas,errortrain,errorval,vartrain,varval)
      -- check.number_eq(errortrain, errors[epoch][1], epsilon,
      --                 string.format("Training error %g is not equal enough to "..
      --                                 "reference error %g",
      --                               errortrain, errors[epoch][1]))
      -- check.number_eq(errorval, errors[epoch][2], epsilon,
      --                 string.format("Validation error %g is not equal enough to "..
      --                                 "reference error %g",
      --                              errorval, errors[epoch][2]))
    end
    os.remove(tmp)
    clock:stop()
    cpu,wall = clock:read()
    --printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
    --printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
    -- print("Test passed! OK!")
end)
