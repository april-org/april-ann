-- OVERWRITTING TOSTRING FUNCTION
matrix.meta_instance.__tostring = function(self)
  local t      = self:toTable()
  local dims   = self:dim()
  local major  = self:get_major_order()
  local coords = {}
  local out    = {}
  local row    = {}
  for i=1,#dims do coords[i]=0 end
  for i=1,#t do
    if #dims > 2 and coords[#dims-2] == 0 then
      table.insert(out,
		   string.format("\n# pos [%s]",
				 table.concat(coords, ",", 1,#dims-2)))
    end
    local j=#dims+1
    repeat
      j=j-1
      coords[j] = coords[j] + 1
      if coords[j] >= dims[j] then coords[j] = 0 end
    until j==1 or coords[j] ~= 0
    table.insert(row, string.format("%g", t[i]))
    if coords[#coords] == 0 then
      table.insert(out, table.concat(row, " ")) row={}
    end
  end
  table.insert(out, string.format("# Matrix of size [%s] in %s",
				  table.concat(dims, ","), major))
  return table.concat(out, "\n")
end

matrix.meta_instance.__add = function(op1, op2)
  if not isa(op1,matrix) then op1,op2=op2,op1 end
  return op1:add(op2)
end

matrix.meta_instance.__sub = function(op1, op2)
  if not isa(op1,matrix) then op1,op2=op2,op1 end
  return op1:sub(op2)
end

matrix.meta_instance.__mul = function(op1, op2)
  if not isa(op1,matrix) then op1,op2=op2,op1 end
  if type(op2) == "number" then return op1:mul_scalar(op2)
  else return op1:mul(op2)
  end
end

matrix.meta_instance.__unm = function(op)
  return op1:mul_scalar(op, -1)
end

-- IMAGE

function matrix.loadImage(filename,format)
  fprintf(io.stderr, "WARNING: matrix.loadImage is deprecated\n")
  local f
  -- si la imagen esta en formato netpbm no hace falta convert:
  if string.match(filename,".*%.p[gnp]m$") then
    f = io.open(filename,"r")
  else
    local dest_format = "pnm"
    if format == "gray" then dest_format = "pgm" end
    f = io.popen(string.format("convert %s %s:-",filename,dest_format))
  end
  if not f then
    error(string.format("Error loading image %s'", filename))
  end
  local b = f:read("*a")
  f:close()
  return matrix.fromPNM(b,format)
end

function matrix.saveImage(matrix,filename)
  local f = io.open(filename,"w")
  f:write(matrix:toPNM())
  f:close()
end

function matrix.loadfile(filename)
  --local f = io.open(filename,"r")
  --local b = f:read("*a")
  --f:close()
  return matrix.fromFilename(filename)
end

function matrix.savefile(matrix,filename,format)
  --local f = io.open(filename,"w")
  --f:write(matrix:toString(format))
  --f:close()
  matrix:toFilename(filename, format)
end

-- RAW (ASCII)

-- TODO: MEter esta funcion como parametro format de matrix:toString
function matrix.saveRAW(matrix,filename)
  local f = io.open(filename,"w")
  local t = matrix:toTable()
  for i=1,table.getn(t) do
     f:write(t[i] .. "\n")
  end
  f:close()
end
