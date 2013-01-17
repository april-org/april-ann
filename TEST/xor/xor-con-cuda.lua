learning_rate  = 0.4
momentum       = 0.1
weight_decay   = 1e-05
semilla        = 1234
aleat          = random(semilla)
num_weights    = 9
bunch_size     = tonumber(arg[1]) or 64
m = matrix.fromString[[
    9
    ascii
      -0.5 -1.2 1.0
      -2.0 4.0 -4.0
      -1.0 2.0 2.0
]]

function show_weights(lared)
  w = lared:weights()
  pesos = dataset.matrix(w,
			 {
			   patternSize = {1},
			   offset      = {0},
			   numSteps    = {num_weights},
			   stepSize    = {1}
			 })
  
  print()
  for i = 1,ds_input:numPatterns() do
    if ann then
      value = lared:calculate(ds_input:getPattern(i))[1]
    else
      value = lared:use(ds_input:getPattern(i))[1]
    end
    printf("%s\t %s\n",
	   table.concat(ds_input:getPattern(i),","),
	   value)
  end
  print()

  for i=1,num_weights do
    printf ("%d\t %1.14f\n", i, pesos:getPattern(i)[1])
  end
end

-----------------------------------------------------------

if ann then
  lared=ann.mlp.all_all.generate{
    topology   = "2 inputs 2 tanh 1 tanh",
    w          = m,
    oldw       = m,
    bunch_size = bunch_size,
  }
  lared:set_option("learning_rate", learning_rate)
  lared:set_option("momentum",      momentum)
  lared:set_option("weight_decay",  weight_decay)
else
  if bunch_size > 1 then
    lared = mlp.generate_with_bunch{
      topology   = "2 inputs 2 tanh 1 tanh",
      w          = m,
      oldw       = m,
      bunch_size = bunch_size,
				   }
  else
    lared = mlp.generate("2 inputs 2 tanh 1 tanh", m, m)
  end
end

auxnet=lared:clone()

if ann then
  ann.mlp.all_all.save(lared, "wop.net", "ascii", "old")
else
  mlp.save(lared, "wop.net", "ascii", "old")
end

m_xor = matrix.fromString[[
    4 3
    ascii
    0 0 0
    0 1 1
    1 0 1
    1 1 0
]]

ds_input  = dataset.matrix(m_xor,{patternSize={1,2}})
ds_output = dataset.matrix(m_xor,{offset={0,2},patternSize={1,1}})

ds_input1  = dataset.matrix(m_xor,{
			      patternSize={1,2},
			      numSteps={1,1}})
ds_output1 = dataset.matrix(m_xor,{
			      offset={0,2},
			      patternSize={1,1},
			      numSteps={1,1}})

ds_input2  = dataset.matrix(m_xor,{
			      offset={1,0},
			      patternSize={1,2},
			      numSteps={1,1}})
ds_output2 = dataset.matrix(m_xor,{
			      offset={1,2},
			      patternSize={1,1},
			      numSteps={1,1}})

-----------------------
-- Valores iniciales --
-----------------------

print ("----------------------------------------------")
print ("----------------------------------------------")
print ("Valores iniciales")

lared = auxnet:clone()
if not ann then
  if bunch_size > 1 then
    lared = mlp.generate_with_bunch{
      topology   = "2 inputs 2 tanh 1 tanh",
      w          = m,
      oldw       = m,
      bunch_size = bunch_size,
				   }
  else
    lared = mlp.generate("2 inputs 2 tanh 1 tanh", m, m)
  end
end
show_weights(lared)

-- ---------------------
-- -- BP Con momentum --
-- ---------------------

-- print ("----------------------------------------------")
-- print ("----------------------------------------------")
-- print ("Test para el BP con momentum 0.4, learning rate 0.4, weight decay 0.0, y sin shuffle")

-- data={
--   input_dataset  = ds_input,
--   output_dataset = ds_output,
--   shuffle        = aleat
-- }

-- for i=1,2 do
--   if ann then
--     lared:train_dataset(data)
--   else
--     data.learning_rate = learning_rate
--     data.momentum      = momentum
--     data.weight_decay  = weight_decay
--     lared:train(data)
--   end
-- end
-- print ("\nDespués de 2 épocas:\n")
-- show_weights(lared)

-- for i=3,300 do
--   if ann then
--     lared:train_dataset(data)
--   else
--     data.learning_rate = learning_rate
--     data.momentum      = momentum
--     data.weight_decay  = weight_decay
--     lared:train(data)
--   end
-- end
-- print ("\nDespués de 300 épocas:\n")
-- show_weights(lared)

-- -----------------------------------
-- -- BP sin momentum y sin shuffle --
-- -----------------------------------

-- print ("----------------------------------------------")
-- print ("----------------------------------------------")
-- --print ("Test para el BP con momentum sin shuffle")


-- lared=auxnet:clone()
-- if not ann then
--   lared = mlp.generate_with_bunch{
--     topology   = "2 inputs 2 tanh 1 tanh",
--     w          = m,
--     oldw       = m,
--     bunch_size = bunch_size,
--   }
-- end

print ("----------------------------------------------")
print ("--------------------------------------------")
print ("Después de mostrar (0,0)")

data={
  input_dataset  = ds_input1,
  output_dataset = ds_output1,
  shuffle        = aleat
}
if ann then
  lared:train_dataset(data)
else
  data.learning_rate = learning_rate
  data.momentum      = momentum
  data.weight_decay  = weight_decay
  lared:train(data)
end
show_weights(lared)

print ("\nDespués de mostrar (0,1)")
data={
  input_dataset  = ds_input2,
  output_dataset = ds_output2,
  shuffle        = aleat
}

if ann then
  lared:train_dataset(data)
else
  data.learning_rate = learning_rate
  data.momentum      = momentum
  data.weight_decay  = weight_decay
  lared:train(data)
end
show_weights(lared)

lared=auxnet:clone()
if not ann then
  if bunch_size > 1 then
    lared = mlp.generate_with_bunch{
      topology   = "2 inputs 2 tanh 1 tanh",
      w          = m,
      oldw       = m,
      bunch_size = bunch_size,
				   }
  else
    lared = mlp.generate("2 inputs 2 tanh 1 tanh", m, m)
  end
end
data={
  input_dataset  = ds_input,
  output_dataset = ds_output,
  shuffle        = aleat
}

lared:set_use_cuda(true, true)

print ("\nDespués de una época")
if ann then
  lared:train_dataset(data)
else
  data.learning_rate = learning_rate
  data.momentum      = momentum
  data.weight_decay  = weight_decay
  lared:train(data)
end
show_weights(lared)

--if ann then ann.mlp.all_all.save(lared, "jeje.net", "ascii", "old") end

print ("\nDespués de dos épocas")
if ann then
  lared:train_dataset(data)
else
  data.learning_rate = learning_rate
  data.momentum      = momentum
  data.weight_decay  = weight_decay
  lared:train(data)
end
show_weights(lared)

lared:set_error_function(ann.error_functions.cross_entropy())

print ("\nDespués de 30000 épocas")
for i=3,30000 do
  if ann then
   print(i, lared:train_dataset(data))
  else
    data.learning_rate = learning_rate
    data.momentum      = momentum
    data.weight_decay  = weight_decay
    lared:train(data)
  end
end
show_weights(lared)
