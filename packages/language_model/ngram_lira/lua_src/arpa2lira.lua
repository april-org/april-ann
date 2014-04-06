get_table_from_dotted_string("ngram.lira.arpa2lira", true)

-- Recibe una tabla con estos argumentos:
--  input_file      fichero de entrada
--  input_filename  solo se usa si no hay input_file
--  limit_vocab     OPCIONAL, se usa para limitar el vocabulario
--  vocabulary      OPCIONAL, se asume que es un lexClass
--  bccue           OPCIONAL
--  eccue           OPCIONAL
--  output_file     fichero de salida
--  output_filename nombre fichero de salida
-- escribe en el fichero con formato .lira
local function arpa2lira(tbl)
  local tbl = get_table_fields(
    {
      input_filename  = { mandatory = true },
      output_filename = { mandatory = true },
      limit_vocab     = { mandatory = false },
      vocabulary      = { mandatory = false }
    }, tbl)
  local theTrie = util.trie()
  local log10   = math.log(10)
  local logZero = -1e12 -- representación de log(-infinito)
  local logOne  = 0
  local limit_vocab = tbl.limit_vocab
  local function arpa_prob(x)
    if x <= -99 then return logZero  end
    return x*log10
  end

  local line,which_n,word,how_many_ngrams
  local orig_state,dest_state,dest_backoff,bow
  local n = 1 -- la n del n-grama, se lee del .arpa

  -- estas tablas contienen el automata
  --local state2transitions={} -- estado -> [ destino,palabra,prob ] * num_trans
  local state2transitions=ngram.lira.arpa2lira.State2Transitions()
  -- la lista de transiciones es vector de {destino,palabra,prob}
  local backoffs={} -- estado -> {estado,prob bajada}

  local max_tr_prob   = {} -- estado -> mayor prob transitar desde el
  --                                    sin usar backoff
  local a_eliminar    = {} -- estados con 0 tr salida y no finales
  local fanout2states = {} -- fanout -> lista de estados con dicho fanout

  local function consider_state(state)
    --if state2transitions[state] == nil then
    if not state2transitions:exists(state) then
      --state2transitions[state] = {}
      state2transitions:create(state)
      max_tr_prob[state]       = logZero
    end
  end

  local vocabulary
  local check_word
  local line_counter = 0 -- para garbage collection

  local verbosity = tbl.verbosity or 0

  if tbl.limit_vocab then
    vocabulary = tbl.limit_vocab
    check_word = function (word)
		   word = string.gsub(word, "\\_", "_")
		   if not vocabulary:getWordId(word) then return false end
		   return true
		 end
  elseif tbl.vocabulary then 
    vocabulary = tbl.vocabulary
    check_word = function (word)
		   word = string.gsub(word, "\\_", "_")
		   if not vocabulary:getWordId(word) then
		     error("arpa2lira found unknown word when using a fixed vocabulary: " .. word)
		   end
		   return true
		 end
  else
    vocabulary = lexClass()
    check_word = function (word)
		   word = string.gsub(word, "\\_", "_")
		   if not vocabulary:getWordId(word) then
		     vocabulary:addPair{
		       word   = word,
		     }
		   end
		   return true
		 end
  end
  
  print(input_filename)
  print(output_filename)
  local input_file  = tbl.input_file  or io.open(tbl.input_filename, "r")
  local output_file = tbl.output_file or io.open(tbl.output_filename,"w")
  local bccue       = tbl.bccue or '<s>'  -- begin context cue
  local eccue       = tbl.eccue or '</s>' -- end   context cue
  -- anyadirlos al diccionario y convertirlos en su id
  if not check_word(bccue) then error("Not found " .. bccue .. " in vocabulary") end
  if not check_word(eccue) then error("Not found " .. eccue .. " in vocabulary") end
  bccue = vocabulary:getWordId(bccue)
  eccue = vocabulary:getWordId(eccue)
  
  -- saltarselo todo hasta llegar a \data\
  repeat
    line = input_file:read('*l')
  until line == '\\data\\'
  -- leer lineas tipo 'ngram numero=numero'
  local ngram_counts = {}
  repeat
    line = input_file:read('*l')
    _,_, which_n,how_many_ngrams = string.find(line,"^ngram (%d+)=(%d+)$")
    which_n         = tonumber(which_n)
    how_many_ngrams = tonumber(how_many_ngrams)    
    if which_n ~= nil then ngram_counts[which_n] = how_many_ngrams end
    if which_n ~= nil and which_n>n then n = which_n end
  until which_n == nil
  
  -- en este punto ya sabemos la n del n-grama

  -- 0-grama, solamente esta el estado "_"
  local zeroGramState = theTrie:find({})
  consider_state(zeroGramState) --  consider_state("_")
  -- un solo estado final:
  local final_state = theTrie:reserveId() -- consider_state("__final__")
  consider_state(final_state)

  local initial_state, lowest_state
  lowest_state = theTrie:find({})
  if n>1 then
    initial_state = theTrie:find({bccue})
  else
    initial_state = lowest_state
  end

  local function process_arpa_line(w, n, nmax)
    --------------------------------------------------------------
    -- example: -0.544068 b a -0.3521825
    -- -0.544068  -> probabilidad de ir de b   hasta b_a
    -- -0.3521825 -> probabilidad de ir de b_a hasta a via backoff
    --------------------------------------------------------------
    -- n-grama w1 w2 ... wn-1 wn
    -- indices 2  3  ... n    n+1

    line_counter = line_counter + 1
    if line_counter > 100000 then
      line_counter = 0
      collectgarbage("collect")
    end

    local num_checked_words = 0
    for i=2,n+1 do
      if check_word(w[i]) then num_checked_words = num_checked_words + 1 end
      w[i] = vocabulary:getWordId(string.gsub(w[i], "\\_", "_"))
    end
    if num_checked_words < n then return end

    local trans_prob = arpa_prob(tonumber(w[1]))
    local orig_state = theTrie:find(table.slice(w,2,n))
    local word       = w[n+1]
    
    local function dest_state()
      if w[n+1] == eccue then return final_state end
      local from = (n<nmax and 2) or 3
      return theTrie:find(table.slice(w,from,n+1))
    end

    local function backoff_dest_state()
      -- encuentra el estado destino de un backoff, bajando a n-gramas de orden menor
      -- hasta dar con un estado ya definido previamente (es decir, que tenga transiciones
      -- de salida o peso de backoff).
      --
      -- n-grama w1 w2 ... wn-1 wn
      -- indices 2  3  ... n    n+1
      -- si n == nmax el estado destino es w2 ... wn-1 y baja a w3 ... wn-1
      local backoff_search_start = (n < nmax and 3) or 4
      local dest_backoff
      repeat
        dest_backoff = theTrie:find(table.slice(w,backoff_search_start,n+1))
        backoff_search_start = backoff_search_start + 1
      --until state2transitions[dest_backoff] ~= nil
      until state2transitions:exists(dest_backoff)
      return dest_backoff
    end

    local function backoff_weight()
      if w[n+2] then return arpa_prob(tonumber(w[n+2]))
      else return logOne
      end
    end

    return orig_state, dest_state(), word, trans_prob, backoff_dest_state(), backoff_weight()
  end
    
  for current_n=1,n do -- leemos hasta n gramas
    -- buscamos la cadena
    local linea_centinela = string.format("\\%d-grams:",current_n)
    while line ~= linea_centinela do
      line = input_file:read('*l')
    end
    line = input_file:read('*l')
    if line then line = string.gsub(line ,"_","\\_") end
    printf("Reading %d-grams\n", current_n)
    io.stdout:flush()
    num_proceseed_lines = 0
    while line ~= "" do
      local w = string.tokenize(line)
      local orig_state, dest_state, word, trans_prob, 
            backoff_dest_state, backoff_weight = process_arpa_line(w, current_n, n)
     if orig_state then
       consider_state(orig_state)
       consider_state(dest_state)
       if dest_state == final_state and backoff_weight ~= logOne then 
	 error("found a transition to final </s> with back-off weight")
       end

       --if trans_prob > logZero then -- este if hay que validarlo
       
       
       --table.insert(state2transitions[orig_state], dest_state)
       --table.insert(state2transitions[orig_state], word)
       --table.insert(state2transitions[orig_state], trans_prob)
       state2transitions:insert(orig_state, dest_state, word, trans_prob)
       
       
       -- para calcular getBestProb(state) y getBestProb()
       max_tr_prob[orig_state] = math.max(max_tr_prob[orig_state],trans_prob)
       --end
       
       -- transición de backoff, como procesamos valores crecientes de n
       -- si ya existe no la tocamos, si no existe estamos arreglando el
       -- problema producido por n-gramas previos inexistentes
       if backoffs[dest_state] == nil and backoff_weight > logZero then
	 backoffs[dest_state] = { backoff_dest_state, backoff_weight }
       end
       
       ----------------------------------------------
     end
      line = input_file:read('*l')
      if line then line = string.gsub(line ,"_","\\_") end
      if math.mod(num_proceseed_lines,10000) == 0 then
	printf("\r%3.0f%%", num_proceseed_lines/ngram_counts[current_n]*100)
	io.stdout:flush()
      end
      num_proceseed_lines = num_proceseed_lines + 1
    end
    printf("\r100%%\n")
  end

  theTrie = nil
  collectgarbage("collect")

  -- cerrar el fichero .arpa, si toca
  if tbl.input_file == nil then -- lo hemos abierto nosotros con io.open
    input_file:close()
  end

  -- calcular para cada estado una cota superior de la mejor forma de
  -- salir de el teniendo en cuenta bajadas por backoff, se utiliza en
  -- getBestProb()
  local bestProb = logZero
  local upperBoundBestProb = {} -- state -> bound
  for st,bestOutProb in pairs(max_tr_prob) do
    local bound       = bestOutProb
    local backoffsum  = 0
    local downst      = st
    while downst ~= zeroGramState and backoffs[downst] ~= nil do
      backoffsum = backoffsum + backoffs[downst][2]
      downst     = backoffs[downst][1]
      bound      = math.max(bound, backoffsum + max_tr_prob[downst])
    end
    upperBoundBestProb[st] = bound
    bestProb = math.max(bestProb,bound)
  end
  max_tr_prob = nil -- porque ya no se vuelve a usar

  -- localizar estados a eliminar, aquellos con 0 transiciones de
  -- salida que no sean el estado final:

  --for st,trans in pairs(state2transitions) do
  local it = state2transitions:beginIt()
  local end_it = state2transitions:endIt()
  while it:notEqual(end_it) do
    local trans = it:getTransitions()
    local st    = it:getState()
    --if #trans==0 and st ~= final_state then
    if trans:size() == 0 and st ~= final_state then
      a_eliminar[st] = true
      if verbosity > 0 then
	fprintf(output_file,"# removing state: %s\n",i)
      end
    end
    it:next()
  end

  -- comprobar si un estado baja a otro a eliminar
  --for st,trans in pairs(state2transitions) do
  local it = state2transitions:beginIt()
  local end_it = state2transitions:endIt()
  while it:notEqual(end_it) do
    local st = it:getState()
    while backoffs[st] and a_eliminar[backoffs[st][1]] do
      backoffs[st][2] = backoffs[st][2] + backoffs[backoffs[st][1]][2]
      backoffs[st][1] = backoffs[backoffs[st][1]][1]
    end
    it:next()
  end

  -- vamos a ordenar el vector de aristas por la palabra
  local incs = { 463792, 198768, 86961, 33936,
		 13776, 4592, 1968, 861, 336, 
		 112, 48, 21, 7, 3, 1 }
  
  -- tbl  la tabla a ordenar que contiene tripletas destino,palabra,prob
  --      y se ordena por palabra
  local function shellsort(tbl)
    local sz = #tbl/3
    for _, gap in ipairs(incs) do
      for i = gap + 1, sz do
	local v1,v2,v3 = tbl:get(i) --tbl[i*3-2],tbl[i*3-1],tbl[i*3]
	for j = i - gap, 1, -gap do
	  local w1,w2,w3 = tbl:get(j)
	  if not (v2 < w2) then break end --tbl[j*3-1]
	  tbl:set(i, w1, w2, w3)
	  -- 	  tbl[i*3-2] = tbl[j*3-2]
	  -- 	  tbl[i*3-1] = tbl[j*3-1]
	  -- 	  tbl[i*3  ] = tbl[j*3  ]
	  i = j
        end
	tbl:set(i, v1, v2, v3)
	--         tbl[i*3-2] = v1
	--         tbl[i*3-1] = v2
	--         tbl[i*3  ] = v3
      end 
    end
  end

  -- eliminar estados cambiando las transiciones que llegan a ellos
  --for st,trans in pairs(state2transitions) do
  local it = state2transitions:beginIt()
  local end_it = state2transitions:endIt()
  while it:notEqual(end_it) do
    local st = it:getState()
    local trans = it:getTransitions()
    local maxprob
    --for i = 1,#trans,3 do
    for i=1,trans:size() do
      --local dest,prob = trans[i],trans[i+2]
      local dest,word,prob = trans:get(i)
      while a_eliminar[dest] do
	local info = backoffs[dest] -- info es destino,bow
	dest = info[1]
	prob = prob+info[2]
      end
      --trans[i],trans[i+2] = dest,prob
      trans:set(i,dest,word,prob)
    end
    -- ordenamos las transiciones por el id de la palabra
    --shellsort(trans)
    trans:sortByWordId()
    it:next()
  end

  -- eliminar efectivamente los "estados a eliminar"
  for st,trans in pairs(a_eliminar) do
    backoffs[st]          = nil
    --state2transitions[st] = nil
    state2transitions:erase(st)
  end

  collectgarbage("collect")

  -- ordenar los estados por numero de transiciones que salen de ellos
  local fan_out_list    = {} -- numero de transiciones que aparecen
  local num_transitions = 0
  local num_states      = 0
  --for st,trans in pairs(state2transitions) do
  local it = state2transitions:beginIt()
  local end_it = state2transitions:endIt()
  while it:notEqual(end_it) do
    local st = it:getState()
    local trans = it:getTransitions()
    --local fanout = #trans/3 -- fan out del estado st
    local fanout = trans:size()
    num_states = num_states + 1 -- contamos estados
    num_transitions = num_transitions + fanout -- contamos todas las transiciones
    if fanout2states[fanout] == nil then -- lista de estados con un fan out dado
      fanout2states[fanout] = {}
      table.insert(fan_out_list,fanout) -- lista de fan outs aparecidos
    end
    table.insert(fanout2states[fanout],st)
    it:next()
  end

  ----------------------------------------------------------------------
  -- empezamos a escribir el formato lira
  ----------------------------------------------------------------------

  local tabla_vocabulario = vocabulary:getWordVocabulary()
  fprintf(output_file,"# number of words and words\n%d\n%s\n",#tabla_vocabulario,
	  table.concat(tabla_vocabulario,"\n"))
  fprintf(output_file,
	  "# max order of n-gram\n%d\n# number of states\n%d\n# number of transitions\n%d\n"..
	    "# bound max trans prob\n%f\n",
	  n,num_states,num_transitions,bestProb)

  ----------------------------------------------------------------------
  -- darle un numero a cada estado
  ----------------------------------------------------------------------
  local state2cod = {} -- state_name   -> state_number
  local cod2state = {} -- state_number -> state_name
  local num_state = -1
  fprintf(output_file,"# how many different number of transitions\n%d\n" ..
	  "# \"x y\" means x states have y transitions\n",
	#fan_out_list)
  -- recorremos los estados ORDENADOS por numero de transiciones de
  -- salida
  table.sort(fan_out_list)
  for i,fan_out in ipairs(fan_out_list) do
    -- codificamos todos los estados con fan_out transiciones
    fprintf(output_file,"%d %d\n",
	    #fanout2states[fan_out],fan_out)
    for k,st in ipairs(fanout2states[fan_out]) do
      num_state            = num_state+1
      state2cod[st]        = num_state
      cod2state[num_state] = st
    end
  end

  fprintf(output_file,"# initial state, final state and lowest state\n%d %d %d\n",
	  state2cod[initial_state],state2cod[final_state],state2cod[lowest_state])

  -- para cada estado imprimimos su estado backoff destino, la
  -- probabilidad de bajar y su cota superior de transitar
  fprintf(output_file,"# state backoff_st 'weight(state->backoff_st)' [max_transition_prob]\n")
  fprintf(output_file,"# backoff_st == -1 means there is no backoff\n")
  -- stcod es el codigo definitivo que sale en lira, statename es el
  -- codigo que le asigno el trie
  for stcod=0,num_state do
    local statename = cod2state[stcod]
    local s         = string.format("%d",stcod)
    local info      = backoffs[statename]
    if info then
      if state2cod[info[1]] == nil then
	error(string.format("se intenta bajar por backoff al estado %s que no existe",
			    info[1]))
      end
      s = s .. string.format(" %d %f",state2cod[info[1]],info[2])
    else
      s = s .. " -1 -1" -- valores especiales para indicar que no hay backoff
    end
    if upperBoundBestProb[statename] then
      s = s .. string.format(" %f", upperBoundBestProb[statename])
    end
    fprintf(output_file,"%s\n",s)
    if math.mod(stcod,100000) == 0 then collectgarbage("collect") end
  end

  -- las transiciones
  fprintf(output_file,"# transitions\n# orig dest word prob\n")
  for stcod=0,num_state do
    local statename = cod2state[stcod]
    --local trans = state2transitions[statename]
    local trans = state2transitions:getTransitions(statename)
    --for i = 1,#trans,3 do
    for i=1,trans:size() do
      local dest,word,prob = trans:get(i) --trans[i],trans[i+1],trans[i+2]
      if verbosity > 1 then
	fprintf(output_file,"# %s -> %s %s %g\n",statename,dest,word,prob)
      end
      fprintf(output_file,"%d %d %d %g\n",
	      stcod,state2cod[dest],word,prob)
    end
    if math.mod(stcod,10000) == 0 then collectgarbage("collect") end
  end
  
  -- cerrar el fichero, si toca
  if tbl.output_file == nil then -- lo hemos abierto nosotros con io.open
    output_file:close()
  end

end

setmetatable(ngram.lira.arpa2lira, { __call=arpa2lira })
