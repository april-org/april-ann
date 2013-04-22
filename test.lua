m = matrix.fromString[[
    9
    ascii
      -0.5 -1.2 1.0
      -2.0 4.0 -4.0
      -1.0 2.0 2.0
]]
thenet = ann.components.stack()
thenet:push( ann.components.hyperplane{ input=2, output=2,
					bias_weights="b1",
					dot_product_weights="w1",
					bias_name="b1",
					dot_product_name="w1" } )
thenet:push( ann.components.logistic() )
thenet:push( ann.components.hyperplane{ input=2, output=1,
					bias_weights="b2",
					dot_product_weights="w2",
					bias_name="b2",
					dot_product_name="w2" } )
thenet:push( ann.components.logistic() )
weights_table,components_table = thenet:build()

weights_table["b1"]:load{ w=m, first_pos=0, column_size=3 }
weights_table["w1"]:load{ w=m, first_pos=1, column_size=3 }
weights_table["b2"]:load{ w=m, first_pos=6, column_size=3 }
weights_table["w2"]:load{ w=m, first_pos=7, column_size=3 }

function print_token(token)
  if token then
    print(table.concat(token:convert_to_memblock():to_table(), " "))
  else
    print(token)
  end
end



function doforward(input_tbl)
  local output = thenet:forward(tokens.memblock(input_tbl))
  print_token(components_table["w1"]:get_input())
  print_token(components_table["w1"]:get_output())
  print_token(components_table["b1"]:get_input())
  print_token(components_table["b1"]:get_output())
  print_token(components_table["w2"]:get_input())
  print_token(components_table["w2"]:get_output())
  print_token(components_table["b2"]:get_input())
  print_token(components_table["b2"]:get_output())
  print_token(output)
end

-- executing with a bunch of patterns
doforward(tokens.table.bunch{ {0,0}, {0,1}, {1,0}, {1,1} })

for _,v in ipairs({ {0,0}, {0,1}, {1,0}, {1,1} }) do
  print("##########")
  doforward(v)
end
