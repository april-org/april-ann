function generate_models_from_alphabet(alphabet)
	local result = {}
	local i=1
	local chr

	while i<= string.len(alphabet) do
		chr = string.sub(alphabet,i,i)
		result[chr]=generate_model(chr, 3*(i-1)+1, 0.5, 0.5, 0.5)
		i=i+1
	end

	return result
end


function generate_alphabet_from_wordlist(wordlist)
	local alphabet={}
	local i
	local chr
	local result=""
	for _, word in pairs(wordlist) do
		i=1
		while i <= string.len(word) do
			chr=string.sub(word, i,i)
			alphabet[chr]=true
			i=i+1
		end
	end

	for k,v in pairs(alphabet) do
		result=result..k
	end

	return result
end


function generate_models_from_wordlist(wordlist, alphabet_models)
	local result = {}
	
	for _,word in pairs(wordlist) do
		result[word]=str2model(word, alphabet_models)
	end

	return result
end

function normalizar_mdt(lista_matrices)
	local utable = {}
	for i,mat in pairs(lista_matrices) do
		table.insert(utable,dataset.matrix(mat))
	end
	
	local dts = dataset.union(utable)
	local lctable = {}
	local m,dt
	
	m,dt = dts:mean_deviation()
	
	for i=1,dts:patternSize() do
	  table.insert(lctable,{{0,-m[i]/dt[i]},{i,1/dt[i]}})
	end
	
	local daux = dataset.linearcomb(dts,lctable)
	
	-- normalizamos
	for i=1,dts:numPatterns() do
		dts:putPattern(i,daux:getPattern(i))
	end
end

function normalizar_minmax(lista_matrices)
	local utable = {}
	for i,mat in pairs(lista_matrices) do
		table.insert(utable,dataset.matrix(mat))
	end
	
	local dts = dataset.union(utable)
	local lctable = {}
	local m,dt
	
	mi,ma = dts:min_max()
	
	for i=1,dts:patternSize() do
	  local r  = 1/(ma[i]-mi[i])
	  table.insert(lctable,{{0,-mi[i]},{i,r}})
	end
	
	local daux = dataset.linearcomb(dts,lctable)
	
	-- normalizamos
	for i=1,dts:numPatterns() do
		dts:putPattern(i,daux:getPattern(i))
	end
end



function load_list_images(filename)
  local name  = {}
  local image = {}
  local file=io.open(filename)
  for l in file:lines() do
    local t = string.tokenize(l,"\t")
    if (t[1]~=nil and t[2]~=nil) then
      table.insert(image,t[1])
      table.insert(name,t[2])
    end
  end
  file:close()
  return name,image
end


-- recibe una tabla con los nombres de fichero de las imagenes y la
-- cadena con los parametros,
-- devuelve una tabla con los parametros
function load_and_param_images(imagetable, params)
  local params_table = {}
  for i, imgname in ipairs(imagetable) do
    printf("loading %s ...\n", IMG_DIR.."/"..imgname)
    local img = Image.load_pgm_gz(IMG_DIR.."/"..imgname)
    table.insert(params_table, param.extract(img, params))
    -- = param_moises(img_s)
  end
  normalizar_mdt(params_table)
  --normalizar_minmax(params_table)

  return params_table
end

function param_m2d(mat,ctxtizq,ctxtder)
  local conf = {
    patternSize = {ctxtizq+1+ctxtder,mat:dim()[2]},
    offset      = {-ctxtizq,0},
    numSteps    = {mat:dim()[1],1},
    stepSize    = {1,1}, -- second is don't care
  }
  --print("mat es",mat,table.concat(mat:dim(),"x"))
  return dataset.matrix(mat,conf)
end

-- recibe dos tablas, la primera con el nombre y la segunda con la
-- matriz de parametros de la imagen correspondiente en el mismo
-- indice
function images_each_word(name_t, param_t,ctxtizq,ctxtder)
  local iew = {}
  for i, name in ipairs(name_t) do
    if iew[name] == nil then 
      iew[name] = {} 
    end
    table.insert(iew[name],
		 param_m2d(param_t[i],ctxtizq,ctxtder))
  end
  return iew
end

function classify_word(models,emismatrix)
  local minscore,bestword
  for w,m in pairs(models) do
    local scr = m:viterbi(emismatrix)
    if (minscore==nil or minscore>scr) then
      minscore=scr
      bestword=w
    end
  end
  return bestword,minscore
end

-- Devuelve una tabla con la secuencia de emisiones
-- de una palabra
function word_emissions(word, alphabet)
	local result={}
	local chr
	local emission
	local i=1
	
	while i <= string.len(word) do
		chr=string.sub(word,i,i)
		emission = 3*(string.find(alphabet, chr, 1, true) - 1)+1
		result[3*(i-1)+1] = emission
		result[3*(i-1)+2] = emission+1
		result[3*(i-1)+3] = emission+2
		i = i+1
	end

	return result
end

----------------------------------------------------------------------
-- programa ppal
----------------------------------------------------------------------

PARAMETROS = "Tqah" --"Tsiepdqa"
filename = arg[1]
IMG_DIR  = arg[2]

-- Cargamos la lista de imagenes
name_tbl,imagename_tbl = load_list_images(filename)

-- Parametrizamos las imagenes
params_tbl = load_and_param_images(imagename_tbl, PARAMETROS)


-- En caso de querer fijar un alfabeto,modificar aqui
alphabet=generate_alphabet_from_wordlist(name_tbl)

-- Generamos los modelos para cada letra y para cada palabra,
model_tbl=generate_models_from_alphabet(alphabet)
temp_mtbl=generate_models_from_wordlist(name_tbl, model_tbl)
for k,v in pairs(temp_mtbl) do
	-- Evitamos reemplazar los modelos de las letras individuales
	-- porque valen igual y tienen menos estados "dummy"
	if not model_tbl[k] then model_tbl[k]=v end
end

-- Generamos el modelo en arbol
tree = strtable2tree(name_tbl)
tree = expand_model(tree, model_tbl)

function print_model(m)
        print("name=", m.name)
        print("transitions={")
        for _,t in pairs(m.transitions) do
                printf("{from=%s, to=%s, prob=%s, emission=%s, output=%s, id=%s}\n",
                        t.from, t.to, t.prob, t.emission or "nil", t.output or "", t.id or "nil")
        end
        print"}"
        print("initial=", m.initial)
        print("final=", m.final)
end

print_model(tree)


model_tbl["__tree"]=tree


-- ahora vamos a entrenar, para ello necesitamos una red neuronal con
-- tantas entradas como: num_parametros * (ctxt_izq+1+ctxt_der)
num_parametros = string.len(PARAMETROS)
ctxt_izq = 5
ctxt_der = 5

iew = images_each_word(name_tbl,params_tbl,ctxt_izq,ctxt_der)

-- iew es una tabla que a cada palabra del lexico le asocia una lista
-- de imagenes ya parametrizadas en forma de datasets

-- numero de salidas de la red neuronal:
num_salidas = 3*string.len(alphabet)

-- dataset auxiliar:
casiuno  = 1 -- 0.9
casicero = (1-casiuno)/(num_salidas-1)
ds_ident = dataset.identity(num_salidas,casicero,casiuno)

ma_alignments = {} -- matrices
ds_alignments = {} -- matrices

for word,dstable in pairs(iew) do
  -- ahora generamos una tabla similar a iew, reemplazamos cada
  -- dataset por una matriz de dim 1 y de talla el numPatterns,
  -- inicializamos la matriz, la metemos en un dataset.index
  ma_alignments[word] = {}
  ds_alignments[word] = {}
  for i,ds in ipairs(dstable) do
    local seq_emission = word_emissions(word, alphabet)
    local a = hmm_trainer.initial_emission_alignment(seq_emission, ds:numPatterns())
    local b = dataset.indexed(dataset.matrix(a), {ds_ident}):toMatrix() 
    table.insert(ma_alignments[word], b)
    table.insert(ds_alignments[word], dataset.matrix(b) )
--     printf("word %s seqemission = %s\nalignment:\n",word,table.concat(seq_emission,","))
--     for r,s in dataset.matrix(b):patterns() do
--       printf("%3d -> %s\n",r,table.concat(s,","))
--     end
    
  end
end

-- creamos la red neuronal...
semilla = 1234
num_entradas = num_parametros*(1+ctxt_izq+ctxt_der)
num_ocultas1 = 50
num_ocultas2 = 100
aleat = random(semilla)
lared = Mlp(string.format("%d inputs %d logistic %d logistic %d softmax",
			  num_entradas,
			  num_ocultas1,
			  num_ocultas2,
			  num_salidas))
lared:generate(aleat, -0.7, 0.7)

-- creamos el HMMTrainer y los modelos

trainer, models = create_hmm_trainer_from_models_list(model_tbl, num_salidas)

-- recopilamos datos para entrenar:
todas_entradas = {}
todas_salidas  = {}
for word,dstable in pairs(iew) do
  local t = ds_alignments[word]
  for i,ds in ipairs(dstable) do
    table.insert(todas_entradas,ds)
    table.insert(todas_salidas,t[i])
  end
end
train_input  = dataset.union(todas_entradas)
train_output = dataset.union(todas_salidas)

datosentrenar = {
   learning_rate  = 0.2,
   momentum       = 0.1,
   weight_decay   = 1e-7,
   input_dataset  = train_input,
   output_dataset = train_output,
   shuffle        = aleat
}

printf([[
Entrenamos una red neuronal con %d patrones de talla %d
y %d valores de salida
]],
train_input:numPatterns(),
train_input:patternSize(),
train_output:patternSize())

veces_train = 0 -- debe ser 20 
veces_em    = 500

printf("medias: %s\n",table.concat(train_output:mean(),","))

trainer.trainer:set_a_priori_emissions(train_output:mean())

for iterem = 1,veces_em do
  printf("------------------------------\nIteracion %d de EM\n",
	 iterem)

  -- entrenamos la red neuronal
  for epoch = 1,veces_train do
    errortrain = lared:train(datosentrenar)
    printf("em %4d epoch %4d mse_train %.7f\n",
	   iterem,epoch,errortrain)
  end

  veces_train = 10
  datosentrenar.learning_rate = 0.05
  datosentrenar.momentum      = 0.05

  aciertos = 0
  totales  = 0

  -- la utilizamos para realizar alineamiento forzado viterbi:

  trainer.trainer:begin_expectation()
  
  for word,dstable in pairs(iew) do
    themodel = models[word]
    for i,mat in ipairs(ma_alignments[word]) do
      -- reentrenar la cadena mat:

      -- primero, generar la matriz con las emisiones:
      local dsparam    = dstable[i]

      --print("antes de aplicar use:")
      --for i,j in ds_alignments[word][i]:patterns() do
      --  printf("%3d -> %s\n",i,table.concat(j,","))
      --end
      
      -- utilizarla con la red neuronal
      --[[
      lared:use_dataset{
	input_dataset  = dsparam,
	output_dataset = ds_alignments[word][i] 
      }
      --]]
      
      --print("despues de aplicar use:")
      --for i,j in ds_alignments[word][i]:patterns() do
      --  printf("%3d -> %s\n",i,table.concat(j,","))
      --end

      --rword,minscr = classify_word(model,matemision)

      printf("mat es una matrix %s\n",table.concat(mat:dim(),"x"))

      prob, rword = models["__tree"]:viterbi{
        input_emission      = mat,
        do_expectation      = false,
      }
      
      fprintf(io.stderr,"%-10s -> %-10s %f\n",word,rword,prob)
      
      os.exit(0)

      totales = totales+1
      if (word == rword) then
	aciertos = aciertos+1
      end
      
      themodel:viterbi{
      	input_emission = mat,
	do_expectation = true,
	output_emission= mat
      }

    end
  end

  trainer.trainer:end_expectation()
 
  printf("Porcentaje de aciertos entrenamiento: %.2f%%\n",
	 100*aciertos/totales)

  filenet = string.format("hmm%d.net",iterem)
  print("salvando "..filenet)
  lared:save(filenet,"ascii")

end




