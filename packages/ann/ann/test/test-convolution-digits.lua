mathcore.set_use_cuda_default(util.is_cuda_available())
--
local bunch_size       = tonumber(arg[1]) or 64
local semilla          = 1234
local weights_random   = random(semilla)
local inf              = -0.6
local sup              =  0.6
local shuffle_random   = random(5678)
local learning_rate    = 0.1
local momentum         = 0.2
local weight_decay     = 1e-04
local L1_norm          = 0.00001
local max_norm_penalty = 4
local max_epochs       = 100
local check_grandients = false
local check_tokens     = false

local conv1 = {1, 3, 3} nconv1=10
local maxp1 = {1, 2, 2}
local conv2 = {nconv1, 2, 2,} nconv2=20
local maxp2 = {1, 2, 2}
local hidden = 100

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

local thenet = ann.components.stack{ name="stack" }:
  push( ann.components.rewrap{ size={ 1, 16, 16 },
                               name="rewrap" } ):
  push( ann.components.convolution{ kernel=conv1, n=nconv1,
                                    name="conv-w1", weights="w1" } ):
  push( ann.components.convolution_bias{ n=nconv1, ndims=#conv1,
                                         name="conv-b1", weights="b1" } ):
  push( ann.components.actf.relu{ name="actf-1" } ):
  push( ann.components.max_pooling{ kernel=maxp1,
                                    name="pool-1" } ):
  push( ann.components.convolution{ kernel=conv2, n=nconv2,
                                    name="conv-w2", weights="w2" } ):
  push( ann.components.convolution_bias{ n=nconv2, ndims=#conv2,
                                         name="conv-b2", weights="b2" } ):
  push( ann.components.actf.relu{ name="actf-2" } ):
  push( ann.components.max_pooling{ kernel=maxp2,
                                    name="pool-2" } ):
  push( ann.components.flatten{ name="flatten" } )

local convolution_output_size = thenet:precompute_output_size()[1]

thenet:
  push( ann.components.hyperplane{ input=convolution_output_size,
                                   output=hidden,
                                   name="hyp-1",
                                   bias_name="b3",
                                   dot_product_name="w3",
                                   bias_weights="b3",
                                   dot_product_weights="w3" } ):
  push( ann.components.actf.relu{ name="actf-3" } ):
  push( ann.components.hyperplane{ input=hidden, output= 10,
                                   name="hyp-2",
                                   bias_name="b4",
                                   dot_product_name="w4",
                                   bias_weights="b4",
                                   dot_product_weights="w4" } ):
  push( ann.components.actf.log_softmax{ name="actf-4" } )


local trainer = trainable.supervised_trainer(thenet,
                                             ann.loss.multi_class_cross_entropy(10),
                                             bunch_size)
trainer:build()

trainer:set_option("learning_rate",     learning_rate)
trainer:set_option("momentum",          momentum)
trainer:set_option("weight_decay",      weight_decay)
trainer:set_option("L1_norm",           L1_norm)
trainer:set_option("max_norm_penalty",  max_norm_penalty)
--
trainer:set_layerwise_option("b.", "weight_decay", 0.0)
trainer:set_layerwise_option("b.", "max_norm_penalty", 0.0)
trainer:set_layerwise_option("b.", "L1_norm", 0.0)

trainer:randomize_weights{
  random      = weights_random,
  inf         = inf,
  sup         = sup,
  use_fanin   = true,
  use_fanout  = true,
}
trainer:randomize_weights{
  name_match  = "b.",
  random      = weights_random,
  inf         = 0,
  sup         = 0.2,
  use_fanin   = true,
  use_fanout  = true,
}

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
			       bunch_size = 1,
			       verbose = false,
			     })
end

for input,idxs in trainable.dataset_multiple_iterator{
  datasets   = { datosvalidar.input_dataset },
  bunch_size = 1, } do
  trainer:calculate(input)
  local c = trainer:component("pool-1")
  local o = c:get_output():get_matrix()
  local d = o:dim()
  local k = 0
  for w in o:sliding_window{ size={1,1,d[3],d[4]}, step={1,1,1,1},
			     numSteps={d[1], d[2], 1, 1} }:iterate() do
    local img = w:clone():rewrap(d[3],d[4])
    matrix.saveImage(img:adjust_range(0,1), "/tmp/WW-".. idxs[1] .. "-"..k..".pnm")
    k=k+1
  end
end

local totalepocas = 0

local errorval = trainer:validate_dataset(datosvalidar)
print("# Initial validation error:", errorval)

local clock = util.stopwatch()
clock:go()

if check_tokens then
  function trainer:train_step(input, output)
    trainable.supervised_trainer.train_step(self,input,output)
    for name,c in self:iterate_components() do -- ("w[34]") do
      print("\n================== "..name.." ==================")
      print("\n++++++++++ input ++++++++++")
      print(c:get_input():get_matrix())
      print("\n++++++++++ output ++++++++++")
      print(c:get_output():get_matrix())
      print("\n++++++++++ error input ++++++++++")
      print(c:get_error_input():get_matrix())
      print("\n++++++++++ error output ++++++++++")
      print(c:get_error_output():get_matrix())
      print("\n======================================")
    end
  end
end

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
  if false then
    local inp  = trainer:component("conv"):get_input():get_matrix()
    local outp = trainer:component("conv"):get_output():get_matrix()
    local err = trainer:component("conv"):get_error_input():get_matrix()
    print("DIM", table.concat(err:dim(), " "))
    for i=1,err:dim()[2] do
    --   for j=1,err:dim()[1] do
    -- 	print(inp:select(2,i):select(1,j):transpose())
    -- 	print()
    -- 	print(outp:select(2,i):select(1,j):transpose())
    -- 	print()
    -- 	print(err:select(2,i):select(1,j):transpose())
    -- 	print()
    --   end
      local sum = err:select(2,i):sum()
      print(i, sum,
	    learning_rate/math.sqrt(bunch_size*err:dim()[3]*err:dim()[4]))
      if sum > 40 then
	for j=1,err:dim()[1] do
	  print(outp:select(2,i):select(1,j):transpose())
	  print()
	  print(err:select(2,i):select(1,j):transpose())
	  print()
	end
      end
    end
  end
  printf("%4d  %.7f %.7f      %.7f %.7f\n",
  	 totalepocas,errortrain,errorval,norm2_w,norm2_b)
  trainer:set_option("learning_rate",
		     trainer:get_option("learning_rate")*0.95)
end

-- trainer:for_each_pattern{
--   input_dataset = datosvalidar.input_dataset,
--   bunch_size    = 1,
--   func = function(idxs, trainer)
--     local c = trainer:component("pool-1")
--     local o = c:get_output():get_matrix()
--     local d = o:dim()
--     local k = 0
--     for w in o:sliding_window{ size={1,1,d[3],d[4]}, step={1,1,1,1},
-- 			       numSteps={d[1], d[2], 1, 1} }:iterate() do
--       local img = w:clone():rewrap(d[3],d[4])
--       matrix.saveImage(img:adjust_range(0,1), "/tmp/jajaja-".. idxs[1] .. "-"..k..".pnm")
--       k=k+1
--     end
--   end
-- }


local img = ann.connections.input_filters_image(trainer:weights("w1"), {3,3})
ImageIO.write(img,"/tmp/filters.png")

clock:stop()
local cpu,wall = clock:read()
printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
