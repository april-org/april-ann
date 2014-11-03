mathcore.set_use_cuda_default(util.is_cuda_available())
--
local check = utest.check
local T = utest.test
T("ConjugateGradientTest", function()
    --
    -- un generador de valores aleatorios... y otros parametros
    bunch_size     = tonumber(arg[1]) or 512
    semilla        = 1234
    weights_random = random(semilla)
    description    = "256 inputs 256 tanh 128 tanh 10 log_softmax"
    inf            = -1
    sup            =  1
    shuffle_random = random(5678)
    rho            = 0.01
    sig            = 0.8
    weight_decay   = 1e-05
    max_epochs     = 10

    -- training and validation
    errors = matrix.fromString[[10 2
ascii
0.1364878 0.2552190
0.0078906 0.0778204
0.0018425 0.0892997
0.0005873 0.0647947
0.0001588 0.1008427
0.0000475 0.0784141
0.0000055 0.0974825
0.0000017 0.1003913
0.0000015 0.0980982
0.0000013 0.0992237
]]
    epsilon = 0.4 -- 40% error

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
    trainer:save(tmp)
    trainer = trainable.supervised_trainer.load(tmp)
    for epoch = 1,max_epochs do
      collectgarbage("collect")
      totalepocas = totalepocas+1
      errortrain,vartrain  = trainer:train_dataset(datosentrenar)
      errorval,varval      = trainer:validate_dataset(datosvalidar)
      trainer:save(tmp)
      trainer = trainable.supervised_trainer.load(tmp)
      printf("%4d  %.7f %.7f :: %.7f %.7f :: %f\n",
             totalepocas,errortrain,errorval,vartrain,varval,trainer:norm2("w.*"))
      --check.number_eq(errortrain, errors:get(epoch,1), epsilon,
      --string.format("Training error %g is not equal enough to "..
      --"reference error %g",
      --errortrain, errors:get(epoch,1)))
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
