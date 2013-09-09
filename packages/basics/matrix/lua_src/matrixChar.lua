-- OVERWRITTING TOSTRING FUNCTION
class_extension(matrixChar, "to_lua_string",
                function(self)
                  return string.format("matrixChar.fromString[[%s]]",
                                       self:toString())
                end)

matrixChar.meta_instance.__tostring = function(self)
  local dims   = self:dim()
  local coords = {}
  local out    = {}
  local row    = {}
  for i=1,#dims do coords[i]=1 end
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
