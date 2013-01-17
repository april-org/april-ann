ann = ann or {}
ann.mlp = ann.mlp or {}
ann.mlp.all_all = ann.mlp.all_all or {}

function ann.mlp.all_all.save(model, filename, mode, old)
  if type(model) ~= "ann.mlp.all_all" then
    error ("Incorrect ANN mode!!!")
  end
  local w,oldw = model:weights()
  local f = io.open(filename,"w")
  f:write("return {\n\""..model:description().."\",\nmatrix.fromString[["..
	  w:toString(mode).."]],")
  if old == "old" then 
    f:write("\nmatrix.fromString[[\n"..
	    oldw:toString(mode).."]],\n")
  end
  f:write("}\n")
  f:close()
end

function ann.mlp.all_all.load(filename, bunch_size)
  local c = loadfile(filename)
  local data = c()
  return c and ann.mlp.all_all.generate{
    topology    = data[1],
    w           = data[2],
    oldw        = data[3],
    bunch_size  = bunch_size,
  }
end

-- layer_num must be > 0 and <= than num_layers
function ann.mlp.all_all.prune_up_to_layer(model, layer_num, bunch_size)
  if type(model) ~= "ann.mlp.all_all" then
    error ("Incorrect ANN mode!!!")
  end
  local layers = {}
  local size = 0
  for i=1,layer_num do
    table.insert(layers, model:get_layer_connections(i))
    size = size + layers[i]:size()
  end
  local w    = matrix( size )
  local oldw = matrix( size )
  local next_pos = 0
  for i=1,#layers do
    next_pos = layers[i]:weights{
      w         = w,
      oldw      = oldw,
      first_pos = next_pos
    }
  end
  local desc_tokens = string.tokenize(model:description(), " ")
  local new_description = table.concat(desc_tokens, " ", 1, layer_num*2)
  local new_model = ann.mlp.all_all.generate{
    topology   = new_description,
    bunch_size = bunch_size,
    w          = w,
    oldw       = oldw
  }
  return new_model
end
