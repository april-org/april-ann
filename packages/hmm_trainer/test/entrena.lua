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
function load_and_param_images(IMG_DIR, imagetable, params)
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

PARAMETROS = "Tsiepdqa"
filename_trn = "../entrenamiento/lista.txt"
filename_val = "../validacion/lista.txt"
filename_tst = "../test/lista.txt"
dir_trn = "../entrenamiento"
dir_val = "../validacion"
dir_tst = "../test"

-- Cargamos la lista de imagenes
name_tbl_trn,imagename_tbl_trn = load_list_images(filename_trn)
name_tbl_val,imagename_tbl_val = load_list_images(filename_val)
name_tbl_tst,imagename_tbl_tst = load_list_images(filename_tst)

-- Parametrizamos las imagenes
params_tbl_trn = load_and_param_images(dir_trn,imagename_tbl_trn, PARAMETROS)
params_tbl_val = load_and_param_images(dir_val,imagename_tbl_val, PARAMETROS)
params_tbl_tst = load_and_param_images(dir_tst,imagename_tbl_tst, PARAMETROS)

inv_name_tbl = {}
for key,tbl in ipairs{name_tbl_trn,name_tbl_val,name_tbl_tst} do
  for i,j in ipairs(tbl) do
    inv_name_tbl[j] = true
  end
end
name_tbl = {}
for nam,v in pairs(inv_name_tbl) do
  table.insert(name_tbl,nam)
end
table.sort(name_tbl)

printf("lista nombres: %s\n",table.concat(name_tbl,";"))

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
ctxt_izq = 4
ctxt_der = 4

iew_trn = images_each_word(name_tbl_trn,params_tbl_trn,ctxt_izq,ctxt_der)
iew_val = images_each_word(name_tbl_val,params_tbl_val,ctxt_izq,ctxt_der)
iew_tst = images_each_word(name_tbl_tst,params_tbl_tst,ctxt_izq,ctxt_der)

-- iew es una tabla que a cada palabra del lexico le asocia una lista
-- de imagenes ya parametrizadas en forma de datasets

-- numero de salidas de la red neuronal:
num_salidas = 3*string.len(alphabet)

-- dataset auxiliar:
casiuno  = 0.9
casicero = (1-casiuno)/(num_salidas-1)
ds_ident = dataset.identity(num_salidas,casicero,casiuno)

ma_alignments = {} -- matrices
ds_alignments = {} -- matrices

for word,dstable in pairs(iew_trn) do
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
  end
end

-- creamos la red neuronal...
semilla = 1234
num_entradas = num_parametros*(1+ctxt_izq+ctxt_der)
num_ocultas1 = 100
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
for word,dstable in pairs(iew_trn) do
  local t = ds_alignments[word]
  for i,ds in ipairs(dstable) do
    table.insert(todas_entradas,ds)
    table.insert(todas_salidas,t[i])
  end
end
train_input  = dataset.union(todas_entradas)
train_output = dataset.union(todas_salidas)

datosentrenar = {
   learning_rate  = 0.05,
   momentum       = 0.05,
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

veces_train = 10
veces_em    = 500

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

  iews = {
    entrenamiento= iew_trn, 
    validacion   = iew_val,
    test         = iew_tst,
  }
  for partitionname,iew in pairs(iews) do
    aciertos = 0
    totales  = 0
    mconf = {}
    for word,dstable in pairs(iew) do
      themodel = models[word]
      collectgarbage() -- para evitar crecimiento excesivo de memoria
      for otherkey,dsparam in ipairs(dstable) do
	-- reentrenar la cadena mat:
	
	-- primero, generar la matriz con las emisiones:
	local matemission = matrix(dsparam:numPatterns(),num_salidas)
	-- utilizarla con la red neuronal
	lared:use_dataset{
	  input_dataset  = dsparam,
	  output_dataset = dataset.matrix(matemission)
	}
	
	prob, rword = models["__tree"]:viterbi{
	  input_emission      = matemission,
	  do_expectation      = false,
	}

	mconf[word] = mconf[word] or {}
	mconf[word][rword] = (mconf[word][rword] or 0)+1
	--printf("confmat%3d%s: %-10s -> %-10s\n",iterem,partitionname,word,rword)
	
	totales = totales+1
	if (word == rword) then
	  aciertos = aciertos+1
	end
	
      end

    end -- recorre iew

    for a,b in pairs(mconf) do
      for c,d in pairs(b) do
	printf("conf: %3d %-10s %-10s (%3d)-> %-10s\n",iterem,partitionname,a,d,c)
      end
    end
    printf("Porcentaje de aciertos iter %d en %s: %.2f%%\n",
	   iterem,partitionname,100*aciertos/totales)

--     printf("matriz confusión:\n%14s","")
--     for a,b in ipairs(name_tbl) do
--       printf("%10s ",b);
--     end
--     printf("\n")
--     local empt = {}
--     for a,b in ipairs(name_tbl) do
--       printf("%14s",b)
--       for c,d in ipairs(name_tbl) do
-- 	e = mconf[b] or empt
-- 	printf("%14d ",(e[d] or 0))
--       end
--       printf("\n")
--     end


  end -- recorre iews

  -- EXPECTATION

  trainer.trainer:begin_expectation()
  
  for word,dstable in pairs(iew_trn) do
    themodel = models[word]
    for i,mat in ipairs(ma_alignments[word]) do
      -- reentrenar la cadena mat:

      -- primero, generar la matriz con las emisiones:
      local dsparam    = dstable[i]

      -- utilizarla con la red neuronal
      lared:use_dataset{
	input_dataset  = dsparam,
	output_dataset = ds_alignments[word][i] 
      }
      
      themodel:viterbi{
      	input_emission = mat,
	do_expectation = true,
	output_emission= mat
      }

    end
  end

  trainer.trainer:end_expectation()
 
  filenet = string.format("redes/hmm%d.net",iterem)
  print("salvando "..filenet)
  lared:save(filenet,"ascii")

end




