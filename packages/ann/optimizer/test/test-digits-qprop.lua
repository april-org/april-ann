mathcore.set_use_cuda_default(util.is_cuda_available())
--
local check = utest.check
local T = utest.test
--
T("QuickPropTest", function()
    -- un generador de valores aleatorios... y otros parametros
    bunch_size     = tonumber(arg[1]) or 512
    semilla        = 1234
    weights_random = random(semilla)
    description    = "256 inputs 256 tanh 128 tanh 10 log_softmax"
    inf            = -1
    sup            =  1
    shuffle_random = random(5678)
    weight_decay   = 1e-05
    max_epochs     = 10

    -- training and validation
    errors = matrix.fromString[[10 2
                                ascii
                                2.4257019 2.2666397
                                2.2404771 2.0808170
                                2.0071127 1.5948118
                                1.4796402 1.0622932
                                0.9099334 0.6604984
                                0.5221485 0.4503360
                                0.3342607 0.3491776
                                0.2382729 0.2637241
                                0.1805640 0.2074246
                                0.1439983 0.1729773
                              ]]
    epsilon = 0.01

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
                                           ann.optimizer.quickprop())
    trainer:build()
    trainer:set_option("learning_rate", 0.1)

    trainer:randomize_weights{
      random      = weights_random,
      inf         = inf,
      sup         = sup,
      use_fanin   = true,
    }

    trainer:set_option("weight_decay", weight_decay)
    trainer:set_layerwise_option("b.*", "weight_decay", 0.0)

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
      check.number_eq(errortrain, errors:get(epoch,1), epsilon,
                      string.format("Training error %g is not equal enough to "..
                                      "reference error %g",
                                    errortrain, errors:get(epoch,1)))
      check.number_eq(errorval, errors:get(epoch,2), epsilon,
                      string.format("Validation error %g is not equal enough to "..
                                      "reference error %g",
                                    errorval, errors:get(epoch,2)))
    end
    os.remove(tmp)
    clock:stop()
    cpu,wall = clock:read()
    --printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
    --printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
    -- print("Test passed! OK!")
end)
