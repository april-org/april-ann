red = ann.mlp{ bunch_size = 4 }
a1 = ann.units.real{ size = 2, ann = red, type = "inputs" }
a2 = ann.units.real{ size = 2, ann = red, type = "hidden",  actf = ann.activations.tanh() }
a3 = ann.units.real{ size = 1, ann = red, type = "outputs", actf = ann.activations.logistic() }
cnn1 = ann.connections.all_all{ input_size = 2, output_size = 2 , ann = red }
cnn2 = ann.connections.all_all{ input_size = 2, output_size = 1 , ann = red }
red:register_action( ann.actions.backprop{ input = a1, output = a2, connections = cnn1 })
red:register_action( ann.actions.backprop{ input = a2, output = a3, connections = cnn2 })
red:register_input(a1)
red:register_output(a3)
red:randomize_weights{ random=random(1234) }

red:set_option("learning_rate", 0.1)
red:set_option("momentum",      0.2)
red:set_option("weight_decay",  1e-6)

m = matrix.fromString[[
    4 3
    ascii
    0 0 0
    0 1 1
    1 0 1
    1 1 0
]]

ds_input  = dataset.matrix(m,{patternSize={1,2}})
ds_output = dataset.matrix(m,{offset={0,2},patternSize={1,1}})

rnd = random(1234)

for i=1,100000 do
  train_error = red:train_dataset{
    input_dataset  = ds_input,
    output_dataset = ds_output,
    shuffle        = rnd,
  }
  val_error = red:validate_dataset{
    input_dataset  = ds_input,
    output_dataset = ds_output,
  }
  print (i, train_error, val_error)
end

for _,pat in ds_input:patterns() do
  out = red:calculate(pat)
  printf("%s  %s\n", table.concat(pat, " "), table.concat(out, " "))
end
