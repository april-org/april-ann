function Image.meta_instance:__tostring()
  local w,h,x,y = self:geometry()
  local mat  = self:matrix()
  local str = string.format("# Image %dx%d%+d%+d [%s]",
			    w,h,x,y,
			    mat:get_reference_string())
  return str
end

function ImageRGB.meta_instance:__tostring()
  local w,h,x,y = self:geometry()
  local mat  = self:matrix()
  local str = string.format("# ImageRGB %dx%d%+d%+d [%s]",
			    w,h,x,y,
			    mat:get_reference_string())
  return str
end

function Image.empty(xsize, ysize)
        local m=matrix(ysize, xsize) -- OJO: ysize == numfilas, etc
	return Image(m)
end

function Image.load(filename)
	fprintf(io.stderr, "WARNING: Image.load() is deprecated!\n"..
	                   "Use ImageIO instead.\n")
	local m=matrix.loadImage(filename, "gray")
	return Image(m)
end

function Image.load_pgm_gz(filename)
	fprintf(io.stderr, "WARNING: Image.load_pgm_gz() is deprecated!\n"..
	                   "Use ImageIO instead.\n")
	local f=io.popen("zcat "..filename)
	local s=f:read("*a")
	f:close()
	return Image(matrix.fromPNM(s))
end


function Image.save(img, filename)
	fprintf(io.stderr, "WARNING: Image.save() is deprecated!\n"..
	                   "Use ImageIO instead.\n")
	-- Clonamos la imagen para salvar a fichero solo el trozo indicado
	-- por el crop de la propia imagen, en lugar de TODA la matriz.
	local i=img:clone() 
	local m=i:info()
	matrix.saveImage(m, filename)
end

