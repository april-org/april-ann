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
