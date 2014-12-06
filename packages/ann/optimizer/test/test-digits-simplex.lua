mathcore.set_use_cuda_default(util.is_cuda_available())
--
local check = utest.check
local T = utest.test
--
T("SimplexConvexTest", function()
    local opt = ann.optimizer.simplex()
    opt:set_option("tol", 1e-10)
    -- optimize quadractic function: f(x) = 3*x^2 - 2*x + 10
    local function f(x) return (3*x^2 - 2*x + 10):sum() end
    -- df(x)/dx = 6*x - 2
    -- the minimum is in x=0.333
    local x = matrix(1,1,{-100})
    opt:execute(function(x) return f(x[1]) end, {x})
    check.eq(x, matrix(1,1,{0.333}))
end)

T("SimplexTestDigits", function()
    -- un generador de valores aleatorios... y otros parametros
    bunch_size     = 1024
    semilla        = 1234
    weights_random = random(semilla)
    description    = "256 inputs 10 softmax"
    inf            = -0.01
    sup            =  0.01
    shuffle_random = random(5678)
    
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
                                           ann.loss.zero_one(), --multi_class_cross_entropy(10),
                                           bunch_size,
                                           ann.optimizer.simplex())
    trainer:build()

    trainer:set_option("rand", shuffle_random)
    trainer:set_option("beta", 0.01)
    trainer:set_option("max_iter", 100)
    trainer:set_option("weight_decay", 0.01)
    trainer:set_option("verbose", true)
    
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
      replacement    = bunch_size,
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
    for totalepocas=1,6 do
      collectgarbage("collect")
      errortrain,vartrain  = trainer:train_dataset(datosentrenar)
      trainer:save(tmp)
      trainer = trainable.supervised_trainer.load(tmp)
      errorval,varval      = trainer:validate_dataset(datosvalidar)
      fprintf(io.stderr, "%4d  %.7f %.7f :: %.7f %.7f\n",
              totalepocas,errortrain,errorval,vartrain,varval)
    end
    os.remove(tmp)
    clock:stop()
    cpu,wall = clock:read()
    --printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
    --printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
    -- print("Test passed! OK!")
end)
