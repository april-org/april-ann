----------------------------------------------------------------------
--                           argumentos
file_name_frames  = arg[1]
origin_directory  = arg[2]
destiny_directory = arg[3]
origin_numstates  = tonumber(arg[4])
dest_numstates    = tonumber(arg[5])
espace_index      = 53

----------------------------------------------------------------------
--                          funciones
----------------------------------------------------------------------
segmentation = {} -- namespace

function segmentation.load_hmm_seq(tbl, numstates)
  -- recibe un vector de tipos de emision a nivel de estado HMM,
  -- devuelve otro vector de pares {fin,tipo}, asume que todos los
  -- estados a nivel HMM son left to right con el mismo numero de
  -- estados
  local numframes,resul
  resul = {}
  local counter = 0
  local prev_state_index
  local prev_model
  for i,state_index in ipairs(tbl) do
    state_index = state_index-1 -- pq empiezan en 1
    local model = math.floor(state_index/numstates)
    -- print(i,state_index,model)
    if model ~= prev_model or state_index < prev_state_index then
      if prev_model then
	table.insert(resul,{counter,prev_model})
	-- print("inserto",counter,prev_model)
      end
    end
    prev_model       = model
    prev_state_index = state_index
    counter          = counter + 1
  end
  if prev_model then
    table.insert(resul,{counter,prev_model})
    -- print("inserto",counter,prev_model)
  end
  return resul
end

function segmentation.crop(tbl, initial_space, final_space)
  local index_begin = 1
  local index_end   = #tbl
  local time_offset = 0
  if #tbl > 0 and tbl[1][2] == initial_space then
    index_begin = 2
    time_offset = tbl[1][1]
  end
  if tbl[#tbl] ~= nil and tbl[#tbl][2] == final_space then
    index_end = index_end-1
  end
  local resul = {}
  for i=index_begin,index_end do
    table.insert(resul, {tbl[i][1]-time_offset,tbl[i][2]})
  end
  return resul
end

-- tbl es una tabla con pares {instante_fin, modelo}
function segmentation.save_hmm_seq(tbl, numframes, numstates)
  local r = random()
  local resul = {}
  local last  = tbl[#tbl][1]
  local ratio = numframes/last
  local prevframe = 0
  for i,pareja in ipairs(tbl) do
    local lastframe = math.round(pareja[1] * ratio)
    -- print('proceso',i,pareja[1],pareja[2],lastframe)
    if i==#tbl then lastframe = numframes end
    local model     = pareja[2]
    local num = lastframe - prevframe
    prevframe = lastframe
    local ristra = {}
    local aux    = {}
    -- lo que todos se llevan seguro
    local atodos = math.floor(num/numstates)
    for i=1,numstates do
      ristra[i] = atodos
      aux[i]    = i
    end
    local sobran = num-atodos*numstates
    aux = r:shuffle(aux)
    for i=1,sobran do
      ristra[aux[i]] = ristra[aux[i]]+1
    end
    local aux=0
    for i=1,numstates do
      local st = model*numstates+i
      for j=1,ristra[i] do
	aux = aux+1
	-- print("   ",aux,"inserto", st)
	table.insert(resul,st)
      end
    end
  end
  return resul
end

function mat2tab(mat) -- auxiliar
  resul = {}
  d = mat:dim()
  if #d ~= 1 then error("expected unidimensional matrix") end
  d = d[1]
  --  print('dimension',d)
  for i=1,d do
    table.insert(resul,mat:getElement(i))
  end
  return resul
end

function tab2mat(tbl)
  resul = matrix(#tbl)
  for i,j in ipairs(tbl) do
    resul:setElement(i,j)
  end
  return resul
end

----------------------------------------------------------------------
--                       programa principal
----------------------------------------------------------------------

for line in io.lines(file_name_frames) do
  line      = string.tokenize(line)
  filename  = line[1]
  numframes = line[2]
  print(filename)
  f = io.open(origin_directory .. "/" .. filename,"r")
  if f == nil then
    print("ERROR, NO ENCUENTRA FICHERO ", filename)
  else
    m = matrix.fromString(f:read("*a"))
    f:close()
    tbl = mat2tab(m)
    tbl = segmentation.load_hmm_seq(tbl,origin_numstates)
    tbl = segmentation.crop(tbl, espace_index, espace_index)
    tbl = segmentation.save_hmm_seq(tbl, numframes, dest_numstates)
    m = tab2mat(tbl)
    m:toFilename(destiny_directory .. "/" .. filename,"ascii")
  end
  --os.exit()
end

