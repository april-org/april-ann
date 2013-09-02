-- OVERWRITTING TOSTRING FUNCTION
matrixDouble.meta_instance.__tostring = function(self)
  local dims   = self:dim()
  local coords = {}
  local out    = {}
  local row    = {}
  for i=1,#dims do coords[i]=1 end
  for i=1,self:size() do
    if #dims > 2 and coords[#dims] == 1 and coords[#dims-1] == 1 then
      table.insert(out,
		   string.format("\n# pos [%s]",
				 table.concat(coords, ",")))
    end
    table.insert(row, string.format("% -11.6g", self:get(table.unpack(coords))))
    local j=#dims+1
    repeat
      j=j-1
      coords[j] = coords[j] + 1
      if coords[j] > dims[j] then coords[j] = 1 end
    until j==1 or coords[j] ~= 1
    if coords[#coords] == 1 then
      table.insert(out, table.concat(row, " ")) row={}
    end
  end
  table.insert(out, string.format("# MatrixDouble of size [%s] [%s]",
				  table.concat(dims, ","),
				  self:get_reference_string()))
  return table.concat(out, "\n")
end

function matrixDouble.loadfile()
  error("Deprecated, use fromFilename method")
end

function matrixDouble.savefile()
  error("Deprecated, use toFilename method")
end
