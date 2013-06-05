-- recibe dos numeros, el numero de estados de la segmentacion
-- inicial, y el numero de estados de cada HMM que queremos que tenga
orig_states = tonumber(arg[1])
orig_dir    = arg[2]
dest_states = tonumber(arg[3])
if not dest_states then
  dest_states = {}
  local i=0
  for line in io.lines(arg[3]) do
    dest_states[i] = tonumber(line)
    i = i+1
  end
else
  local n     = dest_states
  dest_states = {}
  for i=0,100 do
    dest_states[i] = n
  end
end
dest_dir    = arg[4]

if orig_dir == dest_dir then
  error ("NO PUEDES SOBREESCRIBIR EL ORIGEN Y EL DESTINO")
end

semilla     = 1234
rnd         = random(semilla)


-- recorre cada fichero del directorio
for i,file_name in ipairs(glob(orig_dir.."/*")) do
  local file_basename  = string.basename(file_name)
  local m              = matrix.loadfile(file_name)
  local msize          = m:dim()[1]
  local t              = {}
  local ini_pos        = 1
  local ant_ch         = -1
  local ant_emiss      = -1
  local ch             = -1
  local num_emisiones_perdidas = 0
  collectgarbage("collect")
  -- para cada patron de la matriz original
  local ds = dataset.matrix(m)
  for j=1,msize do
    local pat = ds:getPattern(j)
    -- sacamos el indice de caracter que representa la emision
    ch = math.floor((pat[1]-1)/orig_states)
    -- si hemos cambiado de caracter o hemos llegado al final de la
    -- matriz
    -- print(j, "ant_ch: "..ant_ch, "ant_emiss: "..ant_emiss, "ch: "..ch, "emiss: "..pat[1])
    if ch ~= ant_ch or pat[1] < ant_emiss or j == msize then
      -- si no estamos en la primera iteracion
      if j > 1 then
	local size      = j - ini_pos
	local ini_emiss = ant_ch*dest_states[ant_ch] + 1
	local fin       = j-1
	-- si hemos llegado al final de la matriz, entonces iremos de
	-- ini_emiss hasta msize, en otro caso sera de ini_emiss a j-1
	if j == msize then
	  fin  = msize
	  size = size + 1
	end

	-- sacamos el numero de tramas por estado y las que me han sobrado
	local tramas_por_estado = math.floor(size / dest_states[ant_ch])
	local sobran            = size % dest_states[ant_ch]
	local aux_vec_estados = {}
	for k=1,dest_states[ant_ch] do aux_vec_estados[k] = tramas_por_estado end
	local shuffled_vector = rnd:shuffle(dest_states[ant_ch])
	for k=1,sobran do
	  local v = aux_vec_estados[shuffled_vector[k]]
	  aux_vec_estados[shuffled_vector[k]] = v + 1
	end
	local sum=0
	for k=1,dest_states[ant_ch] do
	  sum = sum +aux_vec_estados[k]
	end
	--	print("sum ",sum,size,j-ini_pos)
	for k=1,dest_states[ant_ch] do
	  for rep=1,aux_vec_estados[k] do
	    t[ini_pos] = ini_emiss + k - 1
	    --print(ds:getPattern(ini_pos)[1], t[ini_pos], size, tramas_por_estado, sobran, j)
	    ini_pos    = ini_pos + 1
	  end
	end
	--	print("wop",ini_pos,j)
      end
      ant_ch   = ch
    end
    ant_emiss = pat[1]
  end
  -- comprobamos que no haya ERRORES
  if #t ~= msize then
    error("#t no es correcta ".. #t .. " " .. msize)
  end
  local dest_file_name = dest_dir .. "/" .. file_basename
  local auxm = matrix(#t, t)
  matrix.savefile(auxm, dest_file_name, "binary")
  print(i, file_basename)
end
