mathcore.set_use_cuda_default(util.is_cuda_available())
--
local bunch_size       = tonumber(arg[1]) or 200
local semilla          = 1234
local weights_random   = random(semilla)
local inf              = -2.4
local sup              =  2.4
local shuffle_random   = random(5678)
local weight_decay     = 0.001
local max_epochs       = 1600
local hidden_size      = 1024
local rho              = 0.01 -- desired sparsity
local beta             = 20   -- weight for sparsity penalty

--------------------------------------------------------------

local m1 = ImageIO.read(string.get_path(arg[0]) .. "../../../../TEST/digitos/digits.png"):to_grayscale():invert_colors():matrix()
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

local input_size = train_input:patternSize()

local AD = autodiff
local M  = AD.matrix
local op = AD.op
local s  = AD.ann.logistic
local T  = op.transpose
local x,w,b1,b2 = M('x w b1 b2')

b1:set_broadcast(false, true)
b2:set_broadcast(false, true)

local shared = {
  w = matrix(hidden_size, train_input:patternSize()),
  b1 = matrix(hidden_size, 1),
  b2 = matrix(train_input:patternSize(), 1),
}

-- local CE = op.sum(op.cmul(x,op.log(hat_x)) + op.cmul(1-x,op.log(1-hat_x)))
-- local SP = op.sum( rho * op.log( rho * hat_rho^-1 ) + (1 - rho) * op.log( (1 - rho) * (1 - hat_rho)^-1  ) )

local h = s( w * x + b1 )        -- hidden layer activation
local hat_x = s( T(w) * h + b2 ) -- reconstruction
local hat_rho = op.sum(h, 2) * (1/bunch_size) -- activation average
local MSE = op.sum( op.sum( 0.5 * (hat_x - x)^2, 2 ) ) / bunch_size -- loss
local L2 = op.sum(w^2)                           -- regularization
local SP = op.sum( 0.5 * ( hat_rho - rho )^2 )   -- sparsity penalty
local J  = MSE + weight_decay * L2  + beta * SP  -- loss + reg + sparsity
local dJ_dw = table.pack( J, AD.diff(J, {w, b1, b2}) ) -- J + derivatives
-- compiled functions
local sparsity = AD.func(hat_rho, { x }, shared)  -- returns hid layer sparsity
local forward = AD.func(hat_x, { x }, shared)     -- returns reconstruction
local forward_mse = AD.func(MSE, { x }, shared)   -- returns MSE loss
local backprop = AD.func(dJ_dw, { x }, shared)    -- returns J + derivatives

local weights_list = { "b1", "b2", "w" }

for wname in iterator(weights_list) do
  ann.connections.randomize_weights( shared[wname],
                                     { inf=inf/math.sqrt(hidden_size + input_size),
                                       sup=sup/math.sqrt(hidden_size + input_size),
                                       random=weights_random })
end

-- datos para entrenar
local datosentrenar = {
  input_dataset  = train_input,
  output_dataset = train_input,
  shuffle        = shuffle_random,
  bunch_size     = bunch_size,
  replacement    = bunch_size,
}

local datosvalidar = {
  input_dataset  = val_input,
  output_dataset = val_input,
}

local totalepocas = 0
local valmat = datosvalidar.input_dataset:toMatrix()
local errorval = forward_mse(valmat:t())
print("# Initial validation error:", errorval)

local clock = util.stopwatch()
clock:go()

local opt = ann.optimizer.adadelta()

local function train_dataset(data)
  local mv = stats.running.mean_var()
  for input in trainable.dataset_pair_iterator(data) do
    local tr_loss = opt:execute(
      function()
        local tr_loss,w,b1,b2 = backprop(input:t())
        return tr_loss, { w=w, b1=b1, b2=b2 }
      end,
      shared)
    mv:add(tr_loss)
  end
  return mv:compute()
end

-- print("Epoch Training  Validation")
for epoch = 1,max_epochs do
  collectgarbage("collect")
  totalepocas = totalepocas+1
  local errortrain  = train_dataset(datosentrenar)
  local errorval    = forward_mse(valmat:t())
  --
  printf("%4d  %.7f %.7f\n", totalepocas, errortrain, errorval)
end

local img = ann.connections.input_filters_image(shared.w, {16,16})
ImageIO.write(img,"/tmp/filters.sparse.png")

clock:stop()
local cpu,wall = clock:read()
printf("Wall total time: %.3f    per epoch: %.3f\n", wall, wall/max_epochs)
printf("CPU  total time: %.3f    per epoch: %.3f\n", cpu, cpu/max_epochs)
