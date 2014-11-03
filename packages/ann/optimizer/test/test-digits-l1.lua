mathcore.set_use_cuda_default(util.is_cuda_available())
--
local check = utest.check
local T = utest.test
--
T("L1TestDigits", function()
    -- un generador de valores aleatorios... y otros parametros
    bunch_size     = tonumber(arg[1]) or 64
    semilla        = 1234
    weights_random = random(semilla)
    description    = "256 inputs 256 tanh 128 tanh 10 log_softmax"
    inf            = -1
    sup            =  1
    shuffle_random = random(5678)
    learning_rate  = 0.08
    momentum       = 0.0
    L1_norm        = 0.001
    max_epochs     = 10

    -- training and validation
    errors = {
      {2.2798486, 2.0456107},
      {1.7129538, 1.2954185},
      {0.9751059, 0.6590891},
      {0.5633602, 0.4138165},
      {0.3464335, 0.3450162},
      {0.2428290, 0.2518864},
      {0.1867137, 0.1979152},
      {0.1466725, 0.1708217},
      {0.1282059, 0.1904573},
      {0.1170338, 0.1766910},
    }
    epsilon = 0.05 -- 5%

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
      check.number_eq(errortrain, errors[epoch][1], epsilon,
                      string.format("Training error %g is not equal enough to "..
                                      "reference error %g",
                                    errortrain, errors[epoch][1]))
      check.number_eq(errorval, errors[epoch][2], epsilon,
                      string.format("Validation error %g is not equal enough to "..
                                      "reference error %g",
                                    errorval, errors[epoch][2]))
    end

    clock:stop()
    cpu,wall = clock:read()
    --printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
    --printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
    -- print("Test passed! OK!")

    for wname,w in trainer:iterate_weights("w.*") do
      local v = w:clone():abs():min()
      check.TRUE(v == 0 or v > L1_norm)
      check.gt(w:eq(0.0):to_float():sum(), 0)
      -- print(w:eq(0.0):to_float():sum())
    end
end)
