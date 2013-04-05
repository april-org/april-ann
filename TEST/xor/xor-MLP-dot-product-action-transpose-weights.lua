learning_rate  = 0.4
momentum       = 0.1
weight_decay   = 1e-05
semilla        = 1234
aleat          = random(semilla)
bunch_size     = tonumber(arg[1]) or 64
delta          = 1e-05

initial_result = {
  0.18666581809521,
  0.16942968964577,
  0.17109067738056,
  0.14423334598541
}

final_result = {
  0.0135539099,
  0.9875200987,
  0.9875199199,
  0.0143862367
}

m = matrix.fromString[[
    19
    ascii
       1.0 0.1 0.2
      -1.0 0.3 0.4
      -0.5 -1.2 1.0
      -2.0 4.0 -4.0
       0.1 1.1 -1.5
      -1.0 2.0 2.0 -1.0
]]

bias0_m = matrix.fromString[[
    2
    ascii
      1.0
     -1.0
]]

bias1_m = matrix.fromString[[
    3
    ascii
      -0.5
      -2.0
       0.1
]]
bias2_m = matrix.fromString[[
    1
    ascii
      -1.0
]]
w0_m = matrix.fromString[[
    4
    ascii
      0.1 0.2
      0.3 0.4
]]
w1t_m = matrix.fromString[[
    6
    ascii
      -1.2 4.0 1.1
       1.0 -4.0 -1.5
]]
w2_m = matrix.fromString[[
    3
    ascii
      2.0 2.0 -1.0
]]

function show_weights(lared)
  lared:show_weights()
  print()
  local outds = dataset.matrix(matrix(4))
  lared:use_dataset{ input_dataset = ds_input, output_dataset = outds }
  for i = 1,ds_input:numPatterns() do
    value = lared:calculate(ds_input:getPattern(i))[1]
    printf("%s\t %s\t %s\n",
	   table.concat(ds_input:getPattern(i),","),
	   value,
	   outds:getPattern(i)[1])
  end
  print()
end

function check_result(lared, result)
  local outds = dataset.matrix(matrix(4))
  lared:use_dataset{ input_dataset = ds_input, output_dataset = outds }
  for i = 1,ds_input:numPatterns() do
    value = lared:calculate(ds_input:getPattern(i))[1]
    if math.abs(value - result[i]) > delta then
      error("Incorrect result using bunch_size=1!!! expected " ..
	    value .. " was " .. result[i])
    end
    if math.abs(outds:getPattern(i)[1] - result[i]) > delta then
      error("Incorrect result using bunch_size=" .. bunch_size .."!!!")
    end
  end
end

-----------------------------------------------------------

lared = ann.mlp{ bunch_size = bunch_size }
-- neuron layers
i  = ann.units.real_cod{ size = 2, ann = lared, type = "inputs" }
h0 = ann.units.real_cod{ size = 2, ann = lared, type = "hidden" }
h1 = ann.units.real_cod{ size = 3, ann = lared, type = "hidden" }
o  = ann.units.real_cod{ size = 1, ann = lared, type = "outputs" }

-- connection layers
b0 = ann.connections.bias{ size = h0:num_neurons(), ann = lared, w = bias0_m }
c0 = ann.connections.all_all{ input_size = i:num_neurons(),
			      output_size = h0:num_neurons(), ann = lared,
			      w = w0_m}
b1 = ann.connections.bias{ size = h1:num_neurons(), ann = lared,
			   w = bias1_m }
c1 = ann.connections.all_all{ input_size = h1:num_neurons(),
			      output_size = h0:num_neurons(), ann = lared,
			      w = w1t_m }
b2 = ann.connections.bias{ size = o:num_neurons(), ann = lared,
			   w = bias2_m }
c2 = ann.connections.all_all{ input_size = h1:num_neurons(),
			      output_size = o:num_neurons(), ann = lared,
			      w = w2_m }

-- first layer actions
lared:push_back_all_all_layer{
  input   = i,
  output  = h0,
  bias    = b0,
  weights = c0,
  actfunc = ann.activations.tanh() }
-- second layer actions
lared:push_back_all_all_layer{
  input   = h0,
  output  = h1,
  bias    = b1,
  weights = c1,
  actfunc = ann.activations.tanh(),
  transpose = true }
-- third layer actions
lared:push_back_all_all_layer{
  input   = h1,
  output  = o,
  bias    = b2,
  weights = c2,
  actfunc = ann.activations.logistic() }

lared:set_option("learning_rate", learning_rate)
lared:set_option("momentum",      momentum)
lared:set_option("weight_decay",  weight_decay)

lared2=ann.mlp.all_all.generate{
  topology   = "2 inputs 2 tanh 3 tanh 1 logistic",
  w          = m,
  oldw       = m,
  bunch_size = bunch_size,
			       }
lared2:set_option("learning_rate", learning_rate)
lared2:set_option("momentum",      momentum)
lared2:set_option("weight_decay",  weight_decay)

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

--------
-- GO --
--------

check_result(lared,  initial_result)
check_result(lared2, initial_result)

data = {
  input_dataset  = ds_input,
  output_dataset = ds_output,
  shuffle        = random(1234)
}
data2 = {
  input_dataset  = ds_input,
  output_dataset = ds_output,
  shuffle        = random(1234)
}

for i=1,30000 do
  lared:train_dataset(data)
  lared2:train_dataset(data2)
end

check_result(lared,  final_result)
check_result(lared2, final_result)

print("TEST PASSED!")
