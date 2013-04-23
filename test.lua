m = matrix.fromString[[
    9
    ascii
      -0.5 -1.2 1.0
      -2.0 4.0 -4.0
      -1.0 2.0 2.0
]]
thenet = ann.components.stack{ name="stack" }
thenet:push( ann.components.hyperplane{ input=2, output=2,
					bias_weights="b1",
					dot_product_weights="w1",
					name="hyperplane1",
					bias_name="b1",
					dot_product_name="w1" } )
thenet:push( ann.components.logistic{ name="h1" } )
thenet:push( ann.components.hyperplane{ input=2, output=1,
					bias_weights="b2",
					dot_product_weights="w2",
					name="hyperplane2",
					bias_name="b2",
					dot_product_name="w2" } )
thenet:push( ann.components.logistic{ name="output" } )
weights_table,components_table = thenet:build()

weights_table["b1"]:load{ w=m, first_pos=0, column_size=3 }
weights_table["w1"]:load{ w=m, first_pos=1, column_size=3 }
weights_table["b2"]:load{ w=m, first_pos=6, column_size=3 }
weights_table["w2"]:load{ w=m, first_pos=7, column_size=3 }

function print_token(token)
  if token then
    local t = token:convert_to_memblock():to_table()
    for i=1,#t do t[i] = string.format("%.4f", t[i]) end
    print(table.concat(t, " "))
  else
    print(token)
  end
end

function show_gradients()
  for _,name in ipairs({"output", "w2", "h1", "w1"}) do
    c = components_table[name]
    local errors = c:get_error_input()
    print(name)
    print_token(errors)
  end
  for _,name in ipairs({"b1","w1","b2","w2"}) do
    local w = weights_table[name]:weights()
    print("WEIGHTS", name, table.concat(w:toTable(), " "))
  end
end

thenet:set_option("learning_rate", 0.01)
input_batch = tokens.memblock(tokens.table.bunch{ {0,0}, {0,1}, {1,0}, {1,1} })
target_batch = tokens.memblock(tokens.table.bunch{ {0}, {1}, {1}, {0} })
lossfunc = ann.loss.mse(thenet:get_output_size())

print_token(thenet:forward(input_batch))

for i=1,100000 do
  lossfunc:reset()
  thenet:reset()
  local output  = thenet:forward(input_batch)
  -- print_token(output)
  local tr_loss = lossfunc:loss(output, target_batch)
  print(i, tr_loss)
  local gradient = lossfunc:gradient(output, target_batch)
  thenet:backprop(gradient)
  -- show_gradients()
  thenet:update()
end
