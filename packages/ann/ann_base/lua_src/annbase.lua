ann.mlp         = ann.mlp or {}
ann.mlp.all_all = ann.mlp.all_all or {}
function ann.mlp.all_all.generate(topology)
  local thenet = ann.components.stack()
  local name   = "layer"
  local count  = 1
  local t      = string.tokenize(topology)
  local prev_size = tonumber(t[1])
  for i=3,#t,2 do
    local size = tonumber(t[i])
    local actf = t[i+1]
    thenet:push( ann.components.hyperplane{
		   input=prev_size, output=size,
		   bias_weights="b" .. count,
		   dot_product_weights="w" .. count,
		   name="layer" .. count,
		   bias_name="b" .. count,
		   dot_product_name="w" .. count } )
    if not ann.components[actf] then
      error("Incorrect activation function: " .. actf)
    end
    thenet:push( ann.components[actf]{ name = actf .. count } )
    count = count + 1
    prev_size = size
  end
  return thenet
end
