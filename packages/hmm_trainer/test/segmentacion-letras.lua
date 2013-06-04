function generate_segmentation_image(imgfile, emission_sequence)
	-- Genera una imagen con la palabra en text_image con sus letras
	-- separadas segun la secuencia de emisiones para el alfabeto dado.
	
	local text_image=Image.load_pgm_gz(imgfile)
	
	local result=text_image:clone()
	local e = dataset.matrix(emission_sequence)
	local width, height = text_image:geometry() -- width no se usa
	local cur_emission
	local prev_emission
	local ncambios=0
	local factor=1
	for col, pat in e:patterns() do
		cur_emission = pat[1]
		
		-- si la letra actual es distinta a la anterior, pintamos una
		-- franja en negativo
		if prev_emission then
			if cur_emission ~= prev_emission then
				ncambios=ncambios+1
				if ncambios==3 then
					if factor==1 then factor = 0.9
					else factor = 1 end
					ncambios=0
				end
			end
		end
		
		for i=0,height-1 do
			color = 1-result:getpixel(col-1,i)
			
			if color < 0 then color = 0 end
			if color > 1 then color = 1 end
			result:putpixel(col-1, i, 
				1-factor*color)
		end	

		prev_emission = cur_emission
	end

	return result
end

---- Programa principal

IMG_DIR="../test"

tabla=dofile(arg[1]) -- Toma como parametro un fichero alignment*.lua
DEST_DIR=arg[2]
suffix=arg[3]

for _, t in ipairs(tabla) do
	img = generate_segmentation_image(IMG_DIR.."/"..t[2], t[3])
	Image.save(img, DEST_DIR.."/"..t[2].."_"..suffix..".pgm")
end


