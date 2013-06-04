-- modelo = {name="nombre",
--  transitions={
--   {from="...", to="....", prob=... , id="....", emission="...",output="..."}
--    ...
--  }
--  initial="...",
--  final="..."
-- }

HMMTrainer = HMMTrainer or {} -- namespace


-- Recibe una tabla con estos argumentos:
-- file
-- filename, solo se usa si no hay file
-- vocabulary es OPCIONAL
-- bccue es OPCIONAL
-- bccue es OPCIONAL
function HMMTrainer.read_arpa(tbl)
  local line,which_n,how_many_ngrams,orig,last_orig,dest,word,dest_backoff
  local n = 1
  local nstates, ntrans
  local ngram_count={}
  
  local next_state=0
  local next_trans=0
  local states={} -- string -> number
  local next_word=1
  local vocabulary -- string -> number
  if tbl.vocabulary then 
    vocabulary = tbl.vocabulary
  else
    vocabulary = lexClass()
  end
  local log10 = math.log(10)
  local logZero = -1e12 -- representación de log(-infinito)
  local logOne = 0
  local function arpa_prob(x)
    if x <= -99 then return logZero  end 
    return x*log10
  end
  local file =  tbl.file or io.open(tbl.filename)
  local bccue = tbl.bccue or '<s>'  -- begin context cue
  local eccue = tbl.eccue or '</s>' -- end   context cue
  -- saltarselo todo hasta llegar a \data\
  repeat
    line = file:read('*l')
  until line == '\\data\\'
  -- leer lineas tipo 'ngram numero=numero'
  repeat
    line = file:read('*l')
    _,_, which_n,how_many_ngrams = string.find(line,"^ngram (%d+)=(%d+)$")
    which_n = tonumber(which_n)
    how_many_ngrams = tonumber(how_many_ngrams)
    
    if which_n ~= nil then 
      ngram_count[which_n]=how_many_ngrams
      if which_n>n then n = which_n end
    end
  until which_n == nil
  
  -- El numero de estados esta acotado superiormente por
  -- el numero de i-gramas + 1 con i=1..n-1
  -- El numero de transiciones no-lambda esta acotado por
  -- el numero de i-gramas con i=1..n (algunos indican
  -- transiciones al estado final, que no codificamos expresamente)
  nstates=0
  for i=1,n-1 do
    nstates = nstates + ngram_count[i]
  end
  ntrans = nstates + ngram_count[n] + 1 -- uno mas para poner el centinela en la tabla
  nstates = nstates + 2 	
  local model = {}

  model.name = tbl.filename
  model.final = "final"
  model.transitions={}
  
  -- 0-grama
  states["_"] = next_state
  next_state = next_state + 1
  last_orig=0

  for i=1,n-1 do -- leemos hasta n-1 gramas
    -- buscamos la cadena
    abuscar = string.format("\\%d-grams:",i)
    while (line ~= abuscar) do
      line = file:read('*l')
    end
    --print(line)    
    line = string.gsub(file:read('*l'),"_","\\_")
    while line ~= "" do
      w = string.tokenize(line)
      -- example: -0.544068 b a -0.3521825
      ----------------------------------------------
      prob = arpa_prob(tonumber(w[1]))
      -- i-grama w1 w2 ... wi-1 wi
      -- índices 2  3  ... i    i+1
      word = w[i+1]
      orig_name = '_'..table.concat(w,"_",2,i)
      orig = states[orig_name]
      if orig < last_orig then
	t = table.invert(states)
	print (last_orig, t[last_orig], "->", orig, t[orig])
	error (".arpa file is not sorted")
      elseif orig > last_orig then
	-- si cambia orig, establecemos first_transition del nuevo origen
	--for rft = last_orig+1,orig do
	  --model:set_state_first_transition(rft, next_trans)
	--end
      end
	last_orig=orig
      if word == eccue then -- final
	-- caso w1 ... w_{i-1} es final con prob prob
	--model:set_state_final_prob(orig, prob)
        table.insert(model.transitions, {from=orig_name, to="final", lprob=prob, emission=0}) 
	if not vocabulary:getWordId(word) then
	  if tbl.vocabulary then
	    error("read_arpa found unknown word when using a fixed vocabulary: " .. word)
	  end
	  vocabulary:addWord{
	    id     = next_word,
	    word   = word,
	    outsym = word,
	  }
	  next_word = next_word + 1
	end
      else
	if w[i+2] then
	  bow = arpa_prob(tonumber(w[i+2]))
	else
	  bow = logOne
	end
	-- creamos una transición para subir:
	-- de w1_...w_{i-1} a w1_...w_i
	-- esto implica asignar un numero nuevo al
	-- estado destino
        dest_name = '_'..table.concat(w,"_",2,i+1)
	states[dest_name] = next_state
	dest = next_state
	--model:set_state_name(dest,'_'..table.concat(w,"_",2,i+1)) 
	next_state = next_state + 1
	
	if not vocabulary:getWordId(word) then
	  if tbl.vocabulary then
	    error("read_arpa found unknown word when using a fixed vocabulary: " .. word)
	  end
	  vocabulary:addWord{
	    id     = next_word,
	    word   = word,
	    outsym = word,
	  }
	  next_word = next_word + 1
	end
	if prob > logZero then -- este if hay que validarlo
	  table.insert(model.transitions,
		       { from=orig_name, to=dest_name, lprob=prob, emission=word})
	  --model:set_trans_word(next_trans,vocabulary[word])
	  --model:set_trans_prob(next_trans,prob)
	  --model:set_trans_dest(next_trans,dest)
	  next_trans = next_trans + 1
	end
	
	-- transición de bajada:
	-- transicion nula w1_...w_{i} a w2_...w_{i}
	-- orig_backoff=dest
        dest_backoff_name = '_'..table.concat(w,"_",3,i+1)
	dest_backoff = states[dest_backoff_name]
	-- print("dest_backoff=", '_'..table.concat(w,"_",3,i+1), dest_backoff)
	if prob > logZero then -- este if hay que validarlo
	  table.insert(model.transitions, 
		       {from=dest_name, to=dest_backoff_name, lprob=bow, emission=0})
	end
	--model:set_state_bo_prob(dest, bow)
	--model:set_state_bo_dest_state(dest, dest_backoff)
      end
      ----------------------------------------------
      line = string.gsub(file:read('*l'),"_","\\_")
    end
  end
  --leemos n gramas
  abuscar = string.format("\\%d-grams:",n)
  while (line ~= abuscar) do
    line = file:read('*l')
  end
  --print(line)
  line = string.gsub(file:read('*l'),"_","\\_")
  while line ~= "" do
    w = string.tokenize(line)
    ----------------------------------------------
    prob = arpa_prob(tonumber(w[1]))
    -- n-grama w1 w2 ... wn-1 wn
    -- índices 2  3  ... n    n+1
    word = w[n+1]
    orig_name = '_'..table.concat(w,"_",2,n)
    orig = states[orig_name]
    if orig < last_orig then
      t = table.invert(states)
      print (last_orig, t[last_orig], "->", orig, t[orig])
      error (".arpa file is not sorted")
    elseif orig > last_orig then
      -- si cambia orig, establecemos first_transition del nuevo origen
      --for rft = last_orig+1,orig do
	--model:set_state_first_transition(rft, next_trans)
      --end
    end
    last_orig=orig
    if word == eccue then -- final
      -- caso w1 ... w_{n-1} es final con prob prob
      --model:set_state_final_prob(orig, prob)
      if prob > logZero then -- este if hay que validarlo
	table.insert(model.transitions, {from=orig_name, to="final", lprob=prob, emission=0})
      end
      if not vocabulary:getWordId(word) then
	if tbl.vocabulary then
	  error("read_arpa found unknown word when using a fixed vocabulary: " .. word)
	end
	vocabulary:addWord{
	  id     = next_word,
	  word   = word,
	  outsym = word,
	}
	next_word = next_word + 1
      end
    else
      -- creamos una transición de un n-grama a otro:
      -- de w1_...w_{n-1} a w2_...w_n
      dest_name = '_'..table.concat(w,"_",3,n+1)
      dest = states[dest_name]
      if not vocabulary:getWordId(word) then
	if tbl.vocabulary then
	  error("read_arpa found unknown word when using a fixed vocabulary: " .. word)
	end
	vocabulary:addWord{
	  id     = next_word,
	  word   = word,
	  outsym = word,
	}
	next_word = next_word + 1
      end
      if prob > logZero then -- este if hay que validarlo
	table.insert(model.transitions, {from=orig_name, to=dest_name, lprob=prob, emission=word})
	--model:set_trans_word(next_trans,vocabulary[word])
	--model:set_trans_prob(next_trans,prob)
	--model:set_trans_dest(next_trans,dest)
	next_trans = next_trans + 1
      end
      
    end
    ----------------------------------------------
    line = string.gsub(file:read('*l'),"_","\\_")
  end
  
  -- crear centinela en tabla transiciones
  --for rft = last_orig+1,next_state do
    --model:set_state_first_transition(rft, next_trans)
  --end
  -- establecer estado inicial:
  if n>1 then
    --model:set_initial_state(states["_"..bccue])
    model.initial = "_"..bccue
  else
    --model:set_initial_state(states["_"])
    model.initial = "_"
  end

  -- cerrar el fichero, si toca
  if tbl.file == nil then -- lo hemos abierto nosotros con io.open
    file:close()
  end

  -- establecer numero de estados y de transiciones
  --model:set_number_states_and_trans(next_state, next_trans)
  return model,vocabulary,states
end

-- descomentar para testear:
-- f = io.open(arg[1],"r")
-- m = ngram.read_arpa(f)
-- m:print()
