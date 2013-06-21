-- modelo = {name="nombre",
--  transitions={
--   {from="...", to="....", prob=... , id="....", emission="...",output="..."}
--    ...
--  }
--  initial="...",
--  final="..."
-- }

-- Namespace HMMTrainer
-- 	Clase HMMTrainer.trainer
-- 		metodos:
-- 		constructor -> crea un trainer vacio
-- 		model -> toma una descripcion de un modelo como la de la
-- 		         tabla de arriba y devuelve un objeto Lua de tipo 
-- 		         HMMTrainer.model, que ademas incorpora al diccionario 
-- 		         self.models
--
--              add_to_dict -> para añadir un model al diccionario, estos modelos
--                             se utilizan para expandir otros modelos
--
--              get_cls_transition_prob -> devuelve la probabilidad de transitar
--                                         de una clase de transicion
--
-- 		expand -> utilizado en model:generate_C_model() para expandir
-- 		          recursivamente un modelo
--
-- 	Clase HMMTrainer.model
-- 		metodos:
-- 		generate_C_model -> crea un objeto C++ de tipo hmm_trainer_model
-- 		         con los datos contenidos en el modelo Lua, realizando
-- 		         expansión recursiva de las transiciones segun sea 
-- 		         necesario
-- 	Namespace utils
-- 		funciones:
-- 		generate_hmm3st_desc -> crea descripcion de modelo lineal de 3 estados
-- 		str2model_desc -> crea una descripcion de modelo a partir de una palabra

class("HMMTrainer.trainer")

function HMMTrainer.trainer:__call()
  local o = {
    trainer    = hmm_trainer(), -- Objeto trainer en C
    
    -- tabla cadena_id --> par (cls_state, cls_transition) indices 
    -- en el vector del hmm_trainer C++
    linked_ids = {},
    
    -- tabla nombre_modelo --> modelo lua
    -- TODO: Cuando se meten cosas aqui?
    models     = {}
    
  }

  class_instance(o, self, true)

  return o
end

class("HMMTrainer.model")

function HMMTrainer.trainer:model(m)
  -- Creamos una tabla lua con las transiciones de m, 
  -- sustituyendo los nombres de los estados en .from y .to
  -- por numeros
  -- Asignamos el numero 1 al inicial y el 2 al final
  local state_numbers = {}
  state_numbers[m.initial] = 1
  state_numbers[m.final] = 2
  if m.initial == m.final then
    error("El estado inicial y el final no pueden coincidir: ".. m.name)
  end

  local num = 3
  local newtrans = {}
  for i, t in ipairs(m.transitions) do
    if not(state_numbers[t.from]) then
      state_numbers[t.from] = num
      num = num+1
    end
    
    if not(state_numbers[t.to]) then
      state_numbers[t.to] = num
      num = num+1
    end

    newtrans[i] = {
      from = state_numbers[t.from],
      to = state_numbers[t.to],
      emission = t.emission,
      lprob = t.lprob or math.log(t.prob),
      id = t.id,
      output = t.output,
    }
  end

  -- Creamos un modelo lua con las transiciones renumeradas
  local res = {}
  class_instance(res, HMMTrainer.model, true)
  res.name        = m.name
  res.initial     = state_numbers[m.initial]
  res.final       = state_numbers[m.final]
  res.transitions = newtrans
  res.trainer     = self
  res.num_states  = num-1

  return res
end

function HMMTrainer.model:update_probs()
  for i,t in ipairs(self.transitions) do
    t.lprob = self.trainer:get_cls_transition_logprob(t.id)
  end
end

function HMMTrainer.model.from_table(t)
  local obj = t
  setmetatable(obj, HMMTrainer.model)
  return obj
end

function HMMTrainer.model:to_string()
  local str = {}
  table.insert(str, "{\n")
  table.insert(str, "\tname        = '".. string.gsub(self.name,"'","\\'")
	       .. "',")
  table.insert(str, "\tinitial     = " .. self.initial    .. ",")
  table.insert(str, "\tfinal       = " .. self.final      .. ",")
  table.insert(str, "\tnum_states  = " .. self.num_states .. ",")
  table.insert(str, "\ttransitions = {")
  for i,t in ipairs(self.transitions) do
    table.insert(str,"\t\t{")
    table.insert(str, "\t\t\tfrom     = " .. t.from     ..",")
    table.insert(str, "\t\t\tto       = " .. t.to       ..",")
    table.insert(str, "\t\t\temission = " .. t.emission ..",")
    table.insert(str, "\t\t\tlprob    = " .. t.lprob    ..",")
    if t.id then
      table.insert(str, "\t\t\tid       = '" .. string.gsub(t.id,"'","\\'")
		   .."',")
    end
    if t.output then
      table.insert(str, "\t\t\toutput   = '" ..
		   string.gsub(t.output,"'","\\'").."'")
    end
    table.insert(str,"\t\t},")
  end
  table.insert(str, "\t}")
  table.insert(str, "}")
  return table.concat(str, "\n")
end

-- Ejemplo de un automata en SLF:

--     # Define size of network: N=num nodes and L=num arcs
--     N=6 L=7
--     # List nodes: I=node-number, W=word
--     I=0 W=start
--     I=1 W=end
--     I=2 W=bit
--     I=3 W=but
--     I=4 W=!NULL
--     I=5 W=!NULL
--     # List arcs: J=arc-number, S=start-node, E=end-node 
--     # l=x is used to attach log transition probability
--     J=0 S=0 E=4 l=-1.1
--     J=1 S=4 E=2
--     J=2 S=4 E=3
--     J=3 S=2 E=5
--     J=4 S=3 E=5
--     J=5 S=5 E=4
--     J=6 S=5 E=1

function HMMTrainer.model:to_slf()
  local next_state      = 1
  local next_transition = 1
  local str             = {}
  local id2state        = {}
  local state2id        = {}
  for i,t in ipairs(self.transitions) do
    local from
    local   to
    if state2id[t.from] then
      from = state2id[t.from]
    else
      from             = next_state
      state2id[t.from] = next_state
      next_state       = next_state + 1
    end
    if state2id[t.to] then
      to = state2id[t.to]
    else
      to             = next_state
      state2id[t.to] = next_state
      next_state     = next_state + 1
    end
    if t.emission == 0 then
      -- transicion LAMBDA
      table.insert(str,
		   "J=" .. next_transition ..
		     " S=" .. from ..
		     " E=".. to ..
		     " l=".. t.lprob)
      id2state[from]   = t.from
      id2state[to]     = t.to
      next_transition  = next_transition+1
    else
      table.insert(str,
		   "J=" .. next_transition ..
		     " S=" .. from ..
		     " E=" .. next_state ..
		     " l=" .. t.lprob)
      table.insert(str,
		   "J=" .. next_transition+1 ..
		     " S=" .. next_state ..
		     " E=" .. to)
      id2state[from]         = t.from
      id2state[next_state]   = t.emission
      id2state[to]           = t.to
      next_state             = next_state+1
      next_transition        = next_transition+2
    end
  end
  local first_state = state2id[self.initial]
  local last_state  = state2id[self.final]
  local str2 = {}
  table.insert(str2, "VERSION=1.0")
  table.insert(str2, "N=".. (next_state+1) .. " L=".. (next_transition+1))
  table.insert(str2,"I=0 W=!NULL") -- FIRST STATE DUMMY
  for i=1,next_state-1 do
    if type(id2state[i])=='string' then
      table.insert(str2,"I=".. i .." W=".. id2state[i])
    else
      table.insert(str2,"I=".. i .." W=!NULL")
    end
  end
  table.insert(str2,"I=".. next_state .." W=!NULL") -- LAST STATE DUMMY
  table.insert(str2,
	       "J=0 S=0 E=".. first_state .." l=0") -- TRANSITION TO FIRST STATE FROM DUMMY
  table.insert(str,
	       "J=".. next_transition ..
		 " S="..last_state ..
		 " E=".. next_state .." l=0") -- TRANSITION TO LAST STATE FROM DUMMY
  return table.concat(str2, "\n").."\n"..table.concat(str, "\n").."\n"
end

function HMMTrainer.trainer:add_to_dict(m, name)
  if name==nil then
    name = m.name
  end
  
  self.models[name] = m
end

function HMMTrainer.trainer:get_cls_transition_prob(tr_id)
  if self.linked_ids[tr_id] then
    local index = self.linked_ids[tr_id][2] -- cls_t
    return self.trainer:get_transition_probability(index)
  else
    -- FIXME: ATENCION este if puede causar que algunos bugs no se
    -- puedan encontrar... por eso la razon del print de abajo ;)
    --      print ("EOEO, esto es para debug... estamos solicitando informacion"..
    --	     " de una transicion que no existe en el modelo")
    return 0.0
  end
end

function HMMTrainer.trainer:get_cls_transition_logprob(tr_id)
  if self.linked_ids[tr_id] then
    local index = self.linked_ids[tr_id][2] -- cls_t
    return self.trainer:get_transition_logprobability(index)
  else
    -- FIXME: ATENCION este if puede causar que algunos bugs no se
    -- puedan encontrar... por eso la razon del print de abajo ;)
    --      print ("EOEO, esto es para debug... estamos solicitando informacion"..
    --	     " de una transicion que no existe en el modelo")
    return 0.0
  end
end

-- Expande recursivamente el modelo C referenciado por c_obj
-- utilizando el modelo lua m
function HMMTrainer.trainer:expand(m, c_obj, upper_prob, ini, fin, 
				   cls_ini, cls_fin, upper_output)
  local indices={
    [m.initial]=ini,
    [m.final]=fin,
  }
  local cls_states={
    [ini]=cls_ini,
    [fin]=cls_fin
  }
  upper_output = upper_output or ""

  for _, t in ipairs(m.transitions) do
    local prb = t.lprob
    if t.from == m.initial then 
      prb = prb + upper_prob
    end

    indices[t.from] = indices[t.from] or c_obj:new_state()
    indices[t.to] = indices[t.to] or c_obj:new_state()
    local orig = indices[t.from]
    local dest = indices[t.to]

    local cls_s = cls_states[orig]
    if t.id == "fixed" then
      if (cls_s ~= nil and cls_s ~= -1) then
	print("warning conflicto en transicion de tipo fixed modelo "..
	      m.name.." transicion "..orig.."->"..dest.."cls_s = "..cls_s)
      end
      cls_states[orig] = -1
      cls_s = -1
    end
    if type(t.emission) == 'string' then -- expandible
      if self.models[t.emission]==nil then
	error("Se esta intentando expandir un modelo inexistente: "..t.emission)
      end
      local ci,cf
      ci,cf = self:expand(self.models[t.emission], c_obj, prb, orig, dest, 
			  cls_states[orig], cls_states[dest],
			  (t.output or "")..upper_output)
      cls_states[orig] = ci
      cls_states[dest] = cf
    else -- no es expandible
      local cls_t
      if t.id then -- transicion con nombre
	if t.id == "fixed" then
	  cls_t = self.trainer:new_cls_transition(-1)
	elseif self.linked_ids[t.id] then -- nombre ya en la tabla
	  if cls_s ~= nil and cls_s ~= self.linked_ids[t.id][1] then
	    error ("Automata inconsistente")
	  end
	  cls_t = self.linked_ids[t.id][2]
	else -- con nombre pero no en la tabla (todavia)
	  if not(cls_s) then
	    cls_s = self.trainer:new_cls_state()
	    cls_states[orig] = cls_s
	  end
	  cls_t = self.trainer:new_cls_transition(cls_s)
	  self.linked_ids[t.id] = {cls_s, cls_t}
	end
      else -- transicion anonima y no fija
	if (not cls_s) then
	  cls_s = self.trainer:new_cls_state()
	  cls_states[orig] = cls_s
	end
	cls_t = self.trainer:new_cls_transition(cls_s)	
      end
      local outstr = t.output
      if dest == fin then
	outstr = (outstr or "")..upper_output
	if outstr == "" then outstr = nil end
      end
      c_obj:new_transition(orig, dest, t.emission, cls_t, 
			   prb, outstr)
    end
  end
  return cls_states[ini],cls_states[fin]
end

function HMMTrainer.model:generate_C_model()
  -- genera un objeto C de tipo hmm_trainer_model a
  -- partir de un objeto Lua tipo HMMTrainer.model
  -- expandiendo las transiciones sobre la marcha directamente en C
  local c_obj = hmm_trainer_model(self.trainer.trainer)
  local ini = c_obj:new_state()
  local fin = c_obj:new_state()
  
  c_obj:set_initial_state(ini)
  c_obj:set_final_state(fin)
  
  self.trainer:expand(self, c_obj, 0, ini, fin, nil, nil)
  if not c_obj:prepare_model() then
    error("El modelo no es correcto")
  end
  return c_obj
end

HMMTrainer.utils= HMMTrainer.utils or {} -- otro namespace

function HMMTrainer.utils.name_mangling_unit(unit)
  local ret = unit
  if #unit == 1 then
    ret = "-".. unit .."+"
  else
    if string.find(unit, "-") == nil then
      ret = "-" .. ret
    end
    if string.find(unit, "+") == nil then
      ret = ret .. "+"
    end
  end
  return ret
end

function HMMTrainer.utils.generate_allograph_hmm_desc(name,allographs,emissions,ploops,pskips,output)
  if not pskips then pskips = {} end
  local transitions={}
  local result = {
    name = name
  }
  local num_emissions = table.getn(emissions)
  -- transicion estado inicial
  table.insert(transitions, {
		 from = name.."ini",
		 to   = name.."1",
		 prob = 1,
		 emission = 0
	       })
  -- primera transicion, siempre es necesaria
  table.insert(transitions, {
		 from = name.."1",
		 to   = name.."2",
		 prob = 1,
		 emission = emissions[1],
		 id   = name.."1_2"
	       })
  
  for j=2,num_emissions do
    local pskip = 0.0
    -- skip de j a j+2
    if pskips[j - 1] and j+1 <= num_emissions then
      if pskips[j-1] > 0.0 then
	-- TODO: Esto solo permite skips del primer al tercer estado
	-- del modelo
	table.insert(transitions, {
		       from = name .. tostring(j),
		       to   = name .. tostring(j+2),
		       prob = pskips[j-1],
		       emission = 0,
		       id   = name .. tostring(j) .. "_" .. tostring(j+2)
		     })
	pskip = pskips[j-1]
      end
    end
    -- loop sobre j
    table.insert(transitions, {
		   from = name .. tostring(j),
		   to   = name .. tostring(j),
		   prob = ploops[j-1],
		   emission = emissions[j-1],
		   id   = name .. tostring(j) .. "_" .. tostring(j)
		 })
    -- de j a j+1
    table.insert(transitions, {
		   from = name .. tostring(j),
		   to   = name .. tostring(j+1),
		   prob = 1-ploops[j-1]-pskip,
		   emission = emissions[j],
		   id   = name .. tostring(j) .. "_" .. tostring(j+1)
		 })
  end
  -- loop sobre num_emissions + 1
  table.insert(transitions, {
		 from = name .. tostring(num_emissions+1),
		 to   = name .. tostring(num_emissions+1),
		 prob = ploops[num_emissions],
		 emission = emissions[num_emissions],
		 id   = name .. tostring(num_emissions+1) .. "_" ..
		   tostring(num_emissions+1)})
  table.insert(transitions, {
		 from = name .. tostring(num_emissions+1),
		 to   = name.."fin",
		 prob = 1-ploops[num_emissions],
		 emission = 0,
		 id   = name .. tostring(num_emissions+1) .. "_fin",
		 output = output
	       })
  result.transitions = transitions
  result.initial = name.."ini"
  result.final   = name.."fin"
  return result
end

function HMMTrainer.utils.generate_lr_hmm_desc(name,emissions,ploops,pskips,output)
  if not pskips then pskips = {} end
  local transitions={}
  local result = {
    name = name
  }
  local num_emissions = table.getn(emissions)
  -- transicion estado inicial
  table.insert(transitions, {
		 from = name.."ini",
		 to   = name.."1",
		 prob = 1,
		 emission = 0
	       })
  -- primera transicion, siempre es necesaria
  table.insert(transitions, {
		 from = name.."1",
		 to   = name.."2",
		 prob = 1,
		 emission = emissions[1],
		 id   = name.."1_2"
	       })
  
  for j=2,num_emissions do
    local pskip = 0.0
    -- skip de j a j+2
    if pskips[j - 1] and j+1 <= num_emissions then
      if pskips[j-1] > 0.0 then
	-- TODO: Esto solo permite skips del primer al tercer estado
	-- del modelo
	table.insert(transitions, {
		       from = name .. tostring(j),
		       to   = name .. tostring(j+2),
		       prob = pskips[j-1],
		       emission = 0,
		       id   = name .. tostring(j) .. "_" .. tostring(j+2)
		     })
	pskip = pskips[j-1]
      end
    end
    -- loop sobre j
    table.insert(transitions, {
		   from = name .. tostring(j),
		   to   = name .. tostring(j),
		   prob = ploops[j-1],
		   emission = emissions[j-1],
		   id   = name .. tostring(j) .. "_" .. tostring(j)
		 })
    -- de j a j+1
    table.insert(transitions, {
		   from = name .. tostring(j),
		   to   = name .. tostring(j+1),
		   prob = 1-ploops[j-1]-pskip,
		   emission = emissions[j],
		   id   = name .. tostring(j) .. "_" .. tostring(j+1)
		 })
  end
  -- loop sobre num_emissions + 1
  table.insert(transitions, {
		 from = name .. tostring(num_emissions+1),
		 to   = name .. tostring(num_emissions+1),
		 prob = ploops[num_emissions],
		 emission = emissions[num_emissions],
		 id   = name .. tostring(num_emissions+1) .. "_" ..
		   tostring(num_emissions+1)})
  table.insert(transitions, {
		 from = name .. tostring(num_emissions+1),
		 to   = name.."fin",
		 prob = 1-ploops[num_emissions],
		 emission = 0,
		 id   = name .. tostring(num_emissions+1) .. "_fin",
		 output = output
	       })
  result.transitions = transitions
  result.initial = name.."ini"
  result.final   = name.."fin"
  return result
end

function HMMTrainer.utils.generate_hmm3st_desc(g, e, p1, p2, p3)
  print("HMMTrainer.utils.generate_hmm3st_desc IS DEPRECATED")
  return HMMTrainer.utils.generate_lr_hmm_desc(g,{e,e+1,e+2},{p1,p2,p3})
  --    -- Crea una tabla con una descripcion de un modelo
  -- 	-- lineal con 4 estados para el grafema g

  --         local result={}
  --         result.name = g
  --         result.transitions = {
  --                 {from=g.."ini", to=g.."1",   prob=1,    emission=0,   }, --id=g.."ini1"
  --                 {from=g.."1",   to=g.."2",   prob=1,    emission=e[1],   id=g.."12"},
  --                 {from=g.."2",   to=g.."2",   prob=p1,   emission=e[1],   id=g.."22"},
  --                 {from=g.."2",   to=g.."3",   prob=1-p1, emission=e[2], id=g.."23"},
  --                 {from=g.."3",   to=g.."3",   prob=p2,   emission=e[2], id=g.."33"},
  --                 {from=g.."3",   to=g.."4",   prob=1-p2, emission=e[3], id=g.."34"},
  --                 {from=g.."4",   to=g.."4",   prob=p3,   emission=e[3], id=g.."44"},
  --                 {from=g.."4",   to=g.."fin", prob=1-p3, emission=0,   id=g.."4fin"},
  --         }

  --         result.initial=g.."ini"
  --         result.final=g.."fin"

  --         return result
end

function HMMTrainer.utils.str2model_desc(str,optional_symbols,output)
  optional_symbols = optional_symbols or {}
  -- optional_symbols es un vector del tipo
  -- {'a'=0.5,'b'=0.7,...}
  local stringlen = string.len(str)
  local result={
    name   =str,
    transitions={},
    initial="0",
    final  =tostring(stringlen),
  }
  for i = 1,stringlen do
    local chr = string.sub(str,i,i)
    if i==stringlen and output ~= nil then
      -- la ultima transición emite el output
      table.insert(result.transitions,
		   {from=tostring(i-1), to=tostring(i), prob=1, emission=chr, id="fixed",
		     output = output } )
    else
      table.insert(result.transitions,
		   {from=tostring(i-1), to=tostring(i), prob=1, emission=chr, id="fixed"} )
    end
    if optional_symbols[chr] then
      if i==stringlen and output ~= nil then
	table.insert(result.transitions,
		     {from=tostring(i-1), to=tostring(i),
		       prob=optional_symbols[chr],
		       emission=0, id="fixed",
		       output = output} )
      else
	table.insert(result.transitions,
		     {from=tostring(i-1), to=tostring(i),
		       prob=optional_symbols[chr],
		       emission=0, id="fixed"} )
      end
    end
  end
  return result
end

function HMMTrainer.utils.tbl2model_desc(tbl,optional_symbols,output)
  optional_symbols = optional_symbols or {}
  -- optional_symbols es un vector del tipo
  -- {'a'=0.5,'b'=0.7,...}
  local tablelen = #tbl
  local result={
    name   =table.concat(tbl),
    transitions={},
    initial="0",
    final  =tostring(tablelen),
  }
  for i = 1,tablelen do
    local chr = tbl[i]
    if i==tablelen and output ~= nil then
      -- la ultima transición emite el output
      table.insert(result.transitions,
		   {from=tostring(i-1), to=tostring(i), prob=1, emission=chr, id="fixed",
		     output = output } )
    else
      if output == nil then
	table.insert(result.transitions,
		     {from=tostring(i-1), to=tostring(i), prob=1, emission=chr, id="fixed"} )
      else
	table.insert(result.transitions,
		     {from=tostring(i-1), to=tostring(i), prob=1, emission=chr, id="fixed" } )
      end
    end
    if optional_symbols[chr] then
      if i==stringlen and output ~= nil then
	table.insert(result.transitions,
		     {from=tostring(i-1), to=tostring(i),
		       prob=optional_symbols[chr],
		       emission=0, id="fixed",
		       output = output} )
      else
	table.insert(result.transitions,
		     {from=tostring(i-1), to=tostring(i),
		      prob=optional_symbols[chr],
		      emission=0, id="fixed" } )
      end
    end
  end
  return result
end

-- Esta funcion recibe un diccionario abierto como fichero (tipo HTK),
-- un tiedlist y una tabla de simbolos opcionales, y genera para cada
-- palabra del diccionario una descripcion tipo HMMTrainer en forma de
-- mini lexico en arbol con las pronunciaciones multiples de cada
-- palabra. Esta tabla de descripciones debe despues ser utilizada
-- para generar modelos C de un trainer. De este modo se permite poder
-- coger una secuencia de palabras de la cual no se tiene la
-- transcripcion fonetica y expandar las palabras con los mini lexicos
-- en arbol generados aqui.
function HMMTrainer.utils.dictionary2lextree(dict_file, tied, optional_symbols, trainer)
  local mangling = {}
  optional_symbols = optional_symbols or
    error ("Needs a table ['a'=0.5, 'b'=0.7, ...] "..
	   "with optional symbols (for silences")
  -- tabla donde dejar asociado a cada palabra el conjunto de
  -- transcripciones que se han encontrado
  local word2units = {}
  -- para cad linea del diccionario
  for line in dict_file:lines() do
    -- troceamos las linea y sacamos los datos. El formato es el
    -- siguiente:
    -- OUTSYM [WORD] PROB U0 U1 U2 U3 ...
    local l = string.tokenize(line)
    local outsym = l[1]
    local begin_units = 2
    local prob        = 1
    local w           = outsym
    if #l>1 and string.sub(l[2],1,1)=="[" and string.sub(l[2],#l[2],#l[2])=="]" then
      -- leemos la palabra
      w = string.sub(l[2], 2, #l[2]-1)
      begin_units = begin_units + 1
    end
    if w ~= outsym then error ("Mandatary: word == outsym!!!") end
    local units = {}
    if tonumber(l[begin_units]) ~= nil then
      -- leemos la probabilidad
      prob        = tonumber(l[begin_units])
      begin_units = begin_units + 1
    end
    -- leemos las unidades
    for i=begin_units,#l do
      table.insert(units, l[i])
    end
    word2units[w] = word2units[w] or {}
    if #units > 0 then
      -- guardamos la secuencia en la tabla
      table.insert(word2units[w], units)
    end
  end

  -- En esta tabla se guarda para cada palabra una descripcion de un
  -- modelo HMMTrainer en forma de arbol con todas sus posibles
  -- pronunciaciones
  local word2lextree = {}
  for word,units in pairs(word2units) do
    -- Funcion auxiliar que comprueba si ya existe una transicion en el
    -- modelo desde from hasta to
    function exists_transition(model, from, to)
      for _, t in pairs(model.transitions) do
	if t.from==from and t.to==to and t.emission ~= 0 then
	  return true
	end
      end
      return false
    end
    
    -- Funcion auxiliar que comprueba si ya existe una transicion
    -- lambda en el modelo desde from hasta to
    function exists_lambda(model, from, to)
      for _, t in pairs(model.transitions) do
	if t.from==from and t.to==to and t.emission==0 then
	  return true
	end
      end
      return false
    end
    
    -- rellenamos la tabla result con el arbol de prefijos
    local result = {}
    result.name=word
    result.initial="__initial"
    result.final="__final"
    result.transitions={}
    
    -- para cada posible pronunciacion
    for _,units_seq in ipairs(units) do
      local cur_prefix="__initial"
      local ant_prefix
      local triphone_seq = tied:search_triphone_sequence(units_seq)
      for i=1,#units_seq do
	ant_prefix=cur_prefix
	cur_prefix=cur_prefix ..  " " .. units_seq[i] -- prefijo de longitud i
	if not exists_transition(result, ant_prefix, cur_prefix) then
	  table.insert(result.transitions,
		       {from=ant_prefix,
			to=cur_prefix,
			emission=tied:get_model(triphone_seq[i]),
			prob=1, id="fixed" } )
	end
	if (optional_symbols[units_seq[i]] and
	    not exists_lambda(result, ant_prefix, cur_prefix, units_seq[i])) then
	  table.insert(result.transitions,
		       {from=ant_prefix,
			to=cur_prefix,
			emission=0,
			prob=optional_symbols[units_seq[i]],
			id="fixed" } )
	end
      end
      if not exists_transition(result, cur_prefix, "__final") then
	table.insert(result.transitions,
		     {from=cur_prefix,
		      to="__final",
		      emission=0,
		      prob=1, id="fixed"} )
      end
    end
    if #units > 0 then
      table.insert(word2lextree, result)
      if trainer then
	local aux = trainer:model(result)
	mangling[word] = "__"..word.."__"
	trainer:add_to_dict(aux, mangling[word])
      end
    end
  end
  return word2lextree,mangling
end

function HMMTrainer.utils.strtable2tree(str_tbl, voc, exclude_words)
  if not exclude_words then exclude_words = {} end
  -- Funcion auxiliar que comprueba si ya existe una transicion en el
  -- modelo desde from hasta to
  function exists_transition(model, from, to)
    for _, t in pairs(model.transitions) do
      if t.from==from and t.to==to then
	return true
      end
    end
    return false
  end
  
  local invert_str_tbl = table.invert(str_tbl)
  local result={}
  result.name="prefixtree" -- FIXME: Cambiar?
  result.initial="__initial"
  result.final="__final"
  result.transitions={}

  for _, word in pairs(str_tbl) do
    if not exclude_words[word] then
      -- Recorremos los prefijos de word y vamos creando
      -- transiciones segun sea necesario
      i=1
      local cur_prefix="__initial"
      local ant_prefix
      local cur_chr
      while i<=string.len(word) do
	ant_prefix=cur_prefix
	cur_prefix=string.sub(word, 1, i) -- prefijo de longitud i
	cur_chr=string.sub(word, i, i)
	if not exists_transition(result, ant_prefix, cur_prefix) then
	  table.insert(result.transitions,
		       {from=ant_prefix,
			 to=cur_prefix,
			 emission=cur_chr,
			 prob=1, id="fixed"} )
	end
	i=i+1
      end

      -- La ultima transicion lleva como output la palabra
      if not exists_transition(result, cur_prefix, "__final") then
	local out = cur_prefix
	if voc then out = voc[invert_str_tbl[cur_prefix]] end
	table.insert(result.transitions,
		     {from=cur_prefix,
		       to="__final",
		       emission=0,
		       prob=1, id="fixed",
		       output=out} )
      end
    end
  end

  return result
end

function HMMTrainer.utils.dictionary2tree(dictionary, tied, exclude_words)
  error ("DEPRECATED")
--   if not exclude_words then exclude_words = {} end
--   -- Funcion auxiliar que comprueba si ya existe una transicion en el
--   -- modelo desde from hasta to
--   function exists_transition(model, from, to)
--     for _, t in pairs(model.transitions) do
--       if t.from==from and t.to==to then
-- 	return true
--       end
--     end
--     return false
--   end
  
--   local result={}
--   result.name="prefixtree" -- FIXME: Cambiar?
--   result.initial="__initial"
--   result.final="__final"
--   result.transitions={}

--   for word, word_info in pairs(dictionary.lex) do
--     if not exclude_words[word] then
--       for _,word_phones in ipairs(word_info.units) do
-- 	local word_tied_phones = word_phones
-- 	if tied then
-- 	  word_tied_phones = tied:search_triphone_sequence(word_phones)
-- 	end
-- 	-- Recorremos los prefijos de word_tied_phones y vamos creando
-- 	-- transiciones segun sea necesario
-- 	i=1
-- 	local cur_prefix="__initial"
-- 	local ant_prefix
-- 	local cur_chr
-- 	while i<=#word_tied_phones do
-- 	  ant_prefix=cur_prefix
-- 	  cur_prefix=table.concat(word_tied_phones, "", 1, i-1)
-- 	  cur_chr=HMMTrainer.utils.name_mangling_unit(word_tied_phones[i])
-- 	  if not exists_transition(result, ant_prefix, cur_prefix) then
-- 	    table.insert(result.transitions,
-- 			 {from=ant_prefix,
-- 			   to=cur_prefix,
-- 			   emission=cur_chr,
-- 			   prob=1, id="fixed"} )
-- 	  end
-- 	  i=i+1
-- 	end
	
-- 	-- La ultima transicion lleva como output la palabra
-- 	if not exists_transition(result, cur_prefix, "__final") then
-- 	  local out = word_info.outsym
-- 	  table.insert(result.transitions,
-- 		       {from=cur_prefix,
-- 			 to="__final",
-- 			 emission=0,
-- 			 prob=1, id="fixed",
-- 			 output=out} )
-- 	end
--       end
--     end
--   end

--   return result
end

-- DE MOMENTO ESTA FUNCION SOLO GENERA MODELOS LR CON SKIPS
-- TODO: Hacer que pueda generar cualquier tipo de descripcion
--       j           j+1          j+2          j+3         j+4
-- 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
-- 0.000000e+00 7.768173e-01 2.231827e-01 0.000000e+00 0.000000e+00
-- 0.000000e+00 0.000000e+00 6.739026e-01 3.260975e-01 0.000000e+00
-- 0.000000e+00 0.000000e+00 0.000000e+00 4.722403e-01 5.277597e-01
-- 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
function  HMMTrainer.utils.generate_hmm_desc_from_transP(name,
							 emissions,
							 transition_matrix,
							 output)
  -- numero de estados en la matriz
  local numstates     = transition_matrix:dim()[1]
  -- tabla de transiciones del modelo
  local transitions   = { {from = "ini", to = name.."1", prob=1, emission=0} }
  -- índice de las transiciones de la tabla transitions que llegan al
  -- estado final
  local transitions_to_last_state = {}
  -- indica si el estado final de la matriz tiene transiciones de
  -- salida
  local exists_last_state_transitions = false
  -- Recorremos la matriz generando transiciones
  for state=1,numstates do
    for deststate=1,numstates do
      -- probabilidad en la matriz
      local prob = transition_matrix:get(state,deststate)
      if prob > 0.0 then
	-- el indice de emision siempre es el estado al que llegamos
	-- con la transicion, menos 1
	local emissid = deststate - 1
	table.insert(transitions, {
		       from     = name .. tostring(state),
		       to       = name .. tostring(deststate),
		       prob     = prob,
		       emission = emissions[emissid] or 0,
		       id       = name .. tostring(state) .. "_" .. tostring(deststate)
		     })
	-- Guardamos las transiciones que llegan al último estado
	-- (emitirán el output)
	if deststate == numstates then table.insert(transitions_to_last_state,
						    #transitions) end
	-- El último estado tiene transiciones?
	if state == numstates then exists_last_state_transitions = true end
      end
    end
  end
  -- indicamos que estados son iniciales y finales
  local ini = "ini"
  local fin = name..tostring(numstates)
  if exists_last_state_transitions then
    -- si el último estado tenía transiciones, creamos uno nuevo que
    -- no las tenga, que tendrá el output y transita con lambda
    table.insert(transitions, {
		   from     = name .. tostring(numstates),
		   to       = name .. tostring(numstates+1),
		   prob     = 1,
		   emission = 0,
		   output   = output,
		 })
    -- el nuevo estado será el final
    fin = name..tostring(numstates+1)
  else
    -- si el último estado no tiene transiciones, entonces metemos en
    -- todas las transiciones que van a él el output indicado para
    -- este modelo
    for i,tr in ipairs(transitions_to_last_state) do
      transitions[tr].output = output
    end
  end
  -- devolvemos la tabla:
  return {
    name    = name,
    initial = ini,
    final   = fin,
    transitions = transitions,
  }
end

function  HMMTrainer.utils.apply_gsf_to_desc(desc,factor)
  -- iterar sobre transitions de desc
  for i,t in ipairs(desc.transitions) do
    if t.lprob == nil then
      t.lprob = math.log(t.prob)
      t.prob = nil
    end
    t.lprob = t.lprob * factor -- elevar prob al factor
  end
end

function  HMMTrainer.utils.print_dot(model)
  print("digraph automata {");
  for i,v in ipairs(model.transitions) do
    printf("%s -> %s [label=\"" .. v.emission .. "/" .. (v.lprob or math.log(v.prob)) .. "/" .. (v.output or "").."\"];\n",
	   string.gsub(v.from, "[<>]", ""), string.gsub(v.to, "[<>]", ""))
  end
  print("}")
end
