class.extend(matrix, "t", matrixChar.."transpose")

-- serialization
matrix.__generic__.__make_all_serialization_methods__(matrixChar, "ascii")

matrixChar.meta_instance.__tostring = function(self)
  local dims   = self:dim()
  local coords = {}
  local out    = {}
  local row    = {}
  local so_large = false
  for i=1,#dims do
    coords[i]=1
    if coords[i] > 20 then so_large = true end
  end
  if not so_large then
    for i=1,self:size() do
      table.insert(out, string.format("[%s] = %s",
				      table.concat(coords, ","),
				      self:get(table.unpack(coords))))
      local j=#dims+1
      repeat
	j=j-1
	coords[j] = (coords[j] % dims[j]) + 1
      until j==1 or coords[j] ~= 1
    end
  else
    table.insert(out, "Large matrix, not printed to display")
  end
  table.insert(out, string.format("# MatrixChar of size [%s] [%s]",
				  table.concat(dims, ","),
				  self:get_reference_string()))
  return table.concat(out, "\n")
end

function matrixChar.loadfile()
  error("Deprecated, use fromFilename method")
end

function matrixChar.savefile()
  error("Deprecated, use toFilename method")
end
