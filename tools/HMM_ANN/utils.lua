recog = recog or {}

local aux_local_vars = {}
--
local valid_recog_parameters=table.invert{
  "input_lat_filename",
  "corpus_data_manager", "filename_mfc",
  "filename_fon", "filename_sgm",
  "prepfilename", "txtfilename",
  "convert_MFCname_TRname",
  "convert_MFCname_FRname",
  "silences", "transcription_filter", "hmmtrainer", "tied",
  "models", "left_context", "right_context",
  "mlp_output_dictionary",  "format",
  "hmm_name_mangling", "means_and_devs",
  "dataset_perturbation_conf",
  "dataset_step", "begin_sil", "end_sil", "phdict"
}
function check_params(args)
  for key,v in pairs(args) do
    if not valid_recog_parameters[key] then
      error ("The parameter '" .. key .. "' is incorrect!!")
    end
  end
end

function remove_extensions(path)
  if string.sub(path, #path-2, #path) == ".gz" then
    path = string.remove_extension(path)
  end
  return string.remove_extension(path)
end

function generate_frontiers(segmenter, mask, numFrames, a_lo_bunke)
  local frontiers      = {}
  local ant_frames     = 0
  for i,img_name in ipairs(glob(mask.."*")) do
    local img_m    = matrix.loadImage(img_name, "gray")
    local img      = Image(img_m)
    local _,w,h    = img:info()
    local segments = segmenter:segment(img)
    -- print("Segments: "..table.concat(segments," "))
    -- dividimos por 20 para sacar la trama a la que corresponde este
    -- pixel
    local inc      = (a_lo_bunke == "a_lo_bunke" and 1) or (h/20)
    local ant      = 0
    -- primera frontera
    if ant_frames == 0 then table.insert(frontiers, ant_frames) end
    for j,pixel_pos in ipairs(segments) do
      local current = math.floor(pixel_pos/inc)
      -- printf("divido %f entre %f y da %d\n",pixel_pos,inc,current)
      -- si el frame no cambia, no hacemos nada
      if current ~= ant then
	table.insert(frontiers, current + ant_frames)
	ant = current
      end
    end
    -- metemos el ultimo frame
    local last_frame = math.round(w/inc)
    if ant ~= last_frame then
      table.insert(frontiers, ant_frames + last_frame)
    end
    ant_frames = ant_frames + last_frame
  end
  if frontiers[#frontiers] < numFrames then
    table.insert(frontiers, numFrames)
  end
  return frontiers
end

function load_models_from_hmm_lua_desc(initial_hmm, hmmtrainer)
  local num_models    = 0
  local num_emissions = 0
  local m             = dofile(initial_hmm)
  local models        = m[1]
  local aprioris      = m[2]
  for name,model_info in pairs(models) do
    num_models    = num_models + 1
    num_emissions = num_emissions + #model_info.emissions
    model_info.model.trainer = hmmtrainer
    hmmtrainer:add_to_dict(model_info.model, name)
  end
  --hmmtrainer.trainer:set_a_priori_emissions(m[2])
  return models,num_models,num_emissions
end

function generate_hmm_models_from_tiedlist(tied,
					   num_states,
					   hmmtrainer)
  local emiss_to_hmm  = {}
  local num_models    = 0
  local hmms          = {}
  local next_emission = 1
  local ploops        = {}
  for i=1,num_states do ploops[i] = 0.5 end
  for model_id,name in ipairs(tied.id2name) do
    m = tied.tiedlist[name]
    if name == m then
      num_models = num_models + 1
      local hmm_emissions = {}
      -- generamos el vector de emisiones
      for i=1,num_states do
	hmm_emissions[i] = next_emission
	emiss_to_hmm[next_emission] = name
	next_emission    = next_emission + 1
      end
      -- este es el representante
      local desc = HMMTrainer.utils.generate_lr_hmm_desc(name,
							 hmm_emissions,
							 ploops, {},
							 name)
      local model = hmmtrainer:model(desc)
      hmms[name] = {
	model     = model,
	emissions = hmm_emissions,
      }
      -- TODO: IMPORTANTE!!! esto puede petar si palabra = name,
      -- PETA!!!
      hmmtrainer:add_to_dict(model, name)
    end
  end
  return hmms, num_models, emiss_to_hmm
end


--
-- esta funcion carga un objeto tipo field_manager
-- con todos los datos que necesitamos extraer del corpus
--
function recog.generate_filenames(args)
  check_params(args)
  --
  local corpus_data_manager    = args.corpus_data_manager
  local filename               = args.filename_mfc
  local input_lat_filename      = args.input_lat_filename
  local prepfilename           = args.prepfilename
  local txtfilename            = args.txtfilename
  local phonfilename           = args.filename_fon
  local sgmfilename            = args.filename_sgm
  --------------------
  -- Carga de datos --
  --------------------
  -- CORPUS_DATA_MANAGER
  -- cargamos las listas de nombres de ficheros
  -- nombres de ficheros de audio
  if filename then
    corpus_data_manager:load_lines{
      field    = 'mfcc_filename',
      filename = filename,
    }
  end
  if input_lat_filename then
    corpus_data_manager:load_lines{
      field    = 'input_lat_filename',
      filename = input_lat_filename,
    }
  end
  if sgmfilename then
    corpus_data_manager:load_lines{
      field    = 'sgm_filename',
      filename = sgmfilename,
    }
  end
  if phonfilename then
    corpus_data_manager:load_lines{
      field    = 'phon_filename',
      filename = phonfilename,
    }
  end
  if prepfilename then
    corpus_data_manager:load_lines{
      field    = 'prep_filename',
      filename = prepfilename,
    }
  end
  if txtfilename then
    corpus_data_manager:load_lines{
      field    = 'transcription_filename',
      filename = txtfilename,
    }
  end
end

function recog.load_frames(args)
  check_params(args)
  --
  local corpus_data_manager   = args.corpus_data_manager
  local format                = args.format
  -- CORPUS_DATA_MANAGER
  --  cargamos las tramas acusticas
  corpus_data_manager:apply{
    input_fields   = {'mfcc_filename'},
    output_fields = {
      'frames',     -- campo con matrices: frames acusticos de cada frase
      'frames_size' -- campo con el numero de frames de cada frase
    },
    the_function  =
      function (filename)
	printf ("# Loading MFCCs '%s'\n", filename)
	collectgarbage("collect")
	local m,size
	if format == "png" then
	  local img_gray = ImageIO.read(filename):to_grayscale():invert_colors()
	  m              = img_gray:rotate90cw(1):info()
	  size = m:dim()[1]
	elseif format == "mat" then
	  m,size = load_matrix(filename)
	else
	  m,size = load_mfcc(filename)
	end
	return m,size
      end
  }
end

-- DEPRECATED ????
-- function recog.load_transcription_string(args)
--   check_params(args)
--   --
--   local corpus_data_manager = args.corpus_data_manager
--   local tied                = args.tied
--   -- cargamos la secuencia de texto
--   corpus_data_manager:apply{
--     input_fields   = {
--       'transcription_filename',
--     },
--     output_fields = {
--       'transcription_string' -- campo con la transcripcion de la frase
--     },
--     the_function  =
--       function (filename)
-- 	printf ("# Cargando '%s'\n",filename)
-- 	local f = io.open(filename, "r") or
-- 	error ("No se ha encontrado ".. filename)
-- 	local tr_str = f:read("*l")
-- 	f:close()
-- 	-- hay que substituir el silencio por word_silence
-- 	return (string.gsub(tr_str, " ", word_silence))
--     end
--   }
-- end

function recog.load_fon_string(args)
  check_params(args)
  --
  local corpus_data_manager = args.corpus_data_manager
  local tied                = args.tied
  local phdict              = args.phdict
  -- cargamos la secuencia fonetica
  corpus_data_manager:apply{
    input_fields   = {
      'phon_filename',
    },
    output_fields = {
      'phon_table',
    },
    the_function  =
      function (ph_filename)
	printf ("# Reading '%s'\n",ph_filename)
	local f = io.open(ph_filename, "r") or
	  error ("No se ha encontrado ".. ph_filename)
	local ph_str = f:read("*l")
	f:close()
	if transcription_filter then
	  ph_str = transcription_filter(ph_str)
	end
	ph_tbl = string.tokenize(ph_str)
	if not phdict then
	  ph_tbl = tied:search_triphone_sequence(ph_tbl)
	end
	return ph_tbl
      end
  }
end

function recog.generate_initial_equidistant_hmm_emission_sequence(args)
  check_params(args)
  --
  local tied                = args.tied
  local models              = args.models
  local corpus_data_manager = args.corpus_data_manager
  --
  -- generamos secuencia de emisiones en partes iguales a partir de
  -- la secuencia fonetica y del numero de frames
  corpus_data_manager:apply{
    input_fields   = {
      'phon_table',  -- campo con la transcripcion de la frase
      'frames_size',
      'phon_filename',
    },
    output_fields = {
      'hmm_emission_sequence',  -- tabla con lista emisiones corresp. a cada frase
    },
    the_function  =
      function (tbl,numframes,name)
	print("# Generating equidistant segmentation " .. name)
	return generate_initial_hmm_emission_sequence(tbl,
						      numframes,
						      tied,
						      models)
      end
  }
end

function recog.generate_hmm_models(args)
  check_params(args)
  --
  local corpus_data_manager = args.corpus_data_manager
  local hmmtrainer             = args.hmmtrainer
  local tied                = args.tied
  local silences            = args.silences
  local phdict              = args.phdict
  local begin_sil           = args.begin_sil
  local end_sil             = args.end_sil
  local transcription_filter= args.transcription_filter
  ---------
  -- HMM --
  ---------
  -- generamos los modelos para las frases
  corpus_data_manager:apply{
    input_fields   = {'phon_table','phon_filename'},
    output_fields = {
      'hmm_c_models',     -- modelo hmm en c generado con el hmmtrainer
    },
    the_function  =
      function (tbl,name)
	print("# Generating HMM model for: "..
	      name)
	if not phdict then
  	  if begin_sil then tbl = string.tokenize(begin_sil .. " " .. table.concat(tbl, " ")) end
	  if end_sil then table.insert(tbl, end_sil) end
	  return generate_models_from_sequences(tbl,
						hmmtrainer,
						tied,
						silences)
	else
	  return generate_models_from_words(tbl,
					    hmmtrainer,
					    tied,
					    silences,
					    begin_sil,
					    end_sil,
					    transcription_filter)
	end
      end
  }
end

function recog.load_initial_hmm_emission_sequence(args)
  check_params(args)
  --
  local corpus_data_manager = args.corpus_data_manager
  -- generamos secuencia de emisiones en partes iguales a partir de
  -- la secuencia fonetica y del numero de frames
  corpus_data_manager:apply{
    input_fields   = {
      'sgm_filename',
    },
    output_fields = {
      'hmm_emission_sequence',  -- tabla con lista emisiones corresp. a cada frase
    },
    the_function  =
      function (sgm_filename,numframes)
	print("# Loading initial segmentation " .. sgm_filename)
	local m = matrix.loadfile(sgm_filename)
	local t = m:toTable()
	collectgarbage("collect")
	return t
      end
  }  
end

function recog.generate_datasets(args)
  check_params(args)
  --
  local mlp_output_dictionary = args.mlp_output_dictionary
  local left_context          = args.left_context
  local right_context         = args.right_context
  local corpus_data_manager   = args.corpus_data_manager
  local means_and_devs        = args.means_and_devs
  local dataset_perturbation_conf = args.dataset_perturbation_conf
  local dataset_step              = args.dataset_step
  --------------------------
  -- creamos los datasets --
  --------------------------

  corpus_data_manager:apply{
    input_fields   = {'frames'},
    output_fields = {
      'contextCCdataset',     -- campo con dataset contextualizado de frames acusticos
      'nocontextDataset',              -- campo con dataset de frames acusticos
    },
    the_function  =
      function (frames)
	Cds,ds = contextCCdataset(frames,
				  left_context,
				  right_context,
				  means_and_devs,
				  dataset_perturbation_conf,
				  dataset_step)
	return Cds,ds
      end
  }
  corpus_data_manager:apply{
    input_fields   = {'hmm_emission_sequence','frames_size', 'phon_filename'},
    output_fields = {
      'segmentation_matrix',   -- campo con matriz de emisiones del mejor camino
      -- version que guarda secuencia de indices de emision
      'segmentation_dataset',  -- dataset para entrenar la red neuronal con la matriz anterior
    },
    the_function  =
      function (segmentation,numframes,filename)
	print("# Generating emitions matrix and dataset: ", filename)
	return generate_emitions_matrix_and_dataset(segmentation,
						    numframes,
						    mlp_output_dictionary)
      end
  }
end

--
-- esta funcion carga un objeto tipo field_manager
-- con todos los datos que necesitamos extraer del corpus
--
function corpus_from_MFC_filename(args)
  check_params(args)
  --
  --------------------
  -- Carga de datos --
  --------------------
  -- CORPUS_DATA
  -- cargamos las listas de nombres de ficheros
  corpus_data_manager = field_manager()
  args.corpus_data_manager = corpus_data_manager
  -- nombres de ficheros
  recog.generate_filenames(args)
  collectgarbage("collect")
  -- fonetica
  recog.load_fon_string(args)
  collectgarbage("collect")
  -- modelos HMM
  if args.phdict then
    recog.load_phdict(args)
  end
  recog.generate_hmm_models(args)
  collectgarbage("collect")
  -- tramas acusticas
  recog.load_frames(args)
  collectgarbage("collect")
  -- emisiones de los HMMs
  if args.filename_sgm then
    recog.load_initial_hmm_emission_sequence(args)
  else
    recog.generate_initial_equidistant_hmm_emission_sequence(args)
  end
  collectgarbage("collect")
  print("# Generating datasets")
  -- datasets :)
  recog.generate_datasets(args)
  collectgarbage("collect")
  --
  return corpus_data_manager
end

---------------------------------------------
-- Funcion para recalcular las matrices de --
-- emisiones del modelo acustico           --
---------------------------------------------

function generate_new_segmentation(args)
  local corpus_data    = args.field_manager
  local func          = args.func
  local do_expectation = args.do_expectation
  local num_emissions  = args.num_emissions
  local emission_in_log_base = args.emission_in_log_base or false
  local without_context = args.without_context or false
  local current        = 1
  local count_value    = args.count_value
  --
  -- resegmentamos validacion
  corpus_data:apply{
    input_fields   = {
      'hmm_c_models',
      'segmentation_matrix',
      'contextCCdataset',
      'nocontextDataset',
    },
    output_fields = {
    },
    the_function  =
      function (themodel,segmentation_matrix,contextCCdataset, nocontextDataset)
	print ("# Segmentation... "..current)
	collectgarbage("collect")
	-- matriz de salida para la red neuronal
	local mat_full = matrix(segmentation_matrix:dim()[1],
				num_emissions)
	local mat_full_ds = dataset.matrix(mat_full)
	--	local auxm = nocontextDataset:toMatrix()
	-- 	matrix.saveImage(auxm, string.format("tmp/image_%03d.pnm",
	-- 					     current))
	
	-- generamos la matriz de emisiones dada la
	-- parametrizacion de la frase actual
	if without_context then
	  func:use_dataset{
	    input_dataset  = nocontextDataset,  -- parametrizacion
	    output_dataset = mat_full_ds        -- matriz de emisiones
	  }
	else
	  func:use_dataset{
	    input_dataset = contextCCdataset,   -- parametrizacion
	    output_dataset = mat_full_ds        -- matriz de emisiones
	  }
	end
	
	-- DEBUG BEGIN
	--	first = nocontextDataset:getPattern(1)
	--	out = func:use(first)
	--	print(table.concat(out, " "))
	--	print("========================")
	--	print(table.concat(mat_full_ds:getPattern(1), " "))
	-- DEBUG END
	
	-- 	mat_full:adjust_range(0,1)
	-- 	matrix.saveImage(mat_full,
	--	string.format("tmp/mlp_matrix_%03d.pnm",
	-- 						 current))
	-- ahora generamos la salida de Viterbi
	themodel:viterbi{
	  input_emission       = mat_full,
	  do_expectation       = do_expectation,
	  output_emission_seq  = segmentation_matrix,
	  -- output_emission      = mat_full,              -- para DEBUG
	  emission_in_log_base = emission_in_log_base,
	  count_value          = count_value,
	}
	-- mat_full:adjust_range(0,1)
	-- matrix.saveImage(mat_full, string.format("tmp/hmm_matrix_%03d.pnm",
	--	            current))
	current = current + 1
      end
  }
  --
end

-------------------------------------------------
-------------------------------------------------
-- FUNCIONES PARA TRABAJAR CON UN SOLO FICHERO --
-------------------------------------------------
-------------------------------------------------

function load_frontiers(w, h, frames_size)
  local c = dofile(frontiers_filename)
  local t = {}
  local inc = c.height/20
  local ant = 0
  -- primera frontera
  table.insert(t, ant)
  for i,fr_pos in ipairs(c.frontiers) do
    -- dividimos por 20 para sacar la trama a la que corresponde este
    -- pixel
    local current = math.floor(fr_pos/inc)
    -- si el frame no cambia, no hacemos nada
    if current ~= ant then
      table.insert(t, current)
      ant = current
    end
  end
  -- metemos el ultimo frame
  if ant ~= frames_size then
    table.insert(t, frames_size)
  end
  return t
end

-- carga la matriz de CCs a partir de un nombre de fichero
function load_matrix(MFCCfilename)
  -- cargamos la parametrizacion de MATRIX
  --if MFCCfilename == 'corpus/n02-082a-s05.fea' then os.exit() end
  
  local f = io.open(MFCCfilename) or
  error ("No se ha podido encontrar "..MFCCfilename)
  --print ("#    OPEN END")
  local aux = f:read("*a")
  --print ("#    READ END")
  local ad  = matrix.fromString(aux)
  --print ("#    MATRIX END")
  f:close()
  --print ("# END")
  return ad,ad:dim()[1]
end

-- carga la matriz de CCs a partir de un nombre de fichero
function load_mfcc(MFCCfilename)
  -- cargamos la parametrizacion de MATRIX
  --if MFCCfilename == 'corpus/n02-082a-s05.fea' then os.exit() end
  
  tmpname = os.tmpname()
  local ad
  if string.sub(MFCCfilename, #MFCCfilename - 2, #MFCCfilename) == ".gz" then
    os.execute("gzip -d ".. MFCCfilename .." -c > ".. tmpname)
    ad = htk_interface.read_matrix_from_mfc_file(tmpname)
    os.execute("rm -f ".. tmpname)
  else
    ad = htk_interface.read_matrix_from_mfc_file(MFCCfilename)
  end
  
  return ad,ad:dim()[1]
end

-- genera la segmentacion a partes iguales
function generate_initial_hmm_emission_sequence(tbl,
						numframes,
						tied,
						models)
  local ini
  local fin
  local m
  local emitions
  local num_emitions
  local seg_tbl = {}
  local size_tbl  = #tbl
  local act_frame = 1
  -- recorremos todos los fonemas del fichero
  for i=1,size_tbl do
    m            = tied:get_model(tbl[i])
    ini          = act_frame
    fin          = i*(numframes/size_tbl)
    if not models[m] then error ("Error, modelo no definido: "..m) end
    emitions     = models[m].emissions
    num_emitions = #emitions
    -- dividimos el fonema en sus emisiones
    for k=1,num_emitions do
      -- metemos una entrada por cada frame emitido
      while act_frame < ini + k*(fin-ini)/num_emitions do
	table.insert (seg_tbl,
		      emitions[k])
	act_frame = act_frame + 1
      end
    end
    -- rellenamos hasta el final
    while act_frame <= fin do
      table.insert(seg_tbl,
		   emitions[num_emitions])
      act_frame = act_frame + 1
    end
  end
  --
  while act_frame <= numframes do
    table.insert(seg_tbl,
		 emitions[num_emitions])
    act_frame = act_frame + 1
  end
  --
  return seg_tbl
end


-------------------------------
------------ HMM --------------
-------------------------------

function recog.load_phdict(args)
  -- si nos dan el diccionario fonetico, generamos modelos para las
  -- palabras
  aux_local_vars.mangling = nil -- sirve para el name mangling de las palabras
  print("# Using dictionary for generate phonetic transcriptions")
  _,aux_local_vars.mangling=HMMTrainer.utils.dictionary2lextree(io.open(args.phdict),
								args.tied,
								args.silences,
								args.hmmtrainer)
  collectgarbage("collect")
end

function generate_models_from_words(tbl,
				    hmmtrainer,
				    tied,
				    silences,
				    begin_sil,
				    end_sil,
				    transcription_filter)
  local tr_string = table.concat(tbl, " ")
  -- filtramos la transcripcion
  if transcription_filter then
    tr_string = transcription_filter(tr_string)
  end
  -- anyadimos los silencios
  if begin_sil then
    tr_string = begin_sil .. " " .. tr_string
  end
  if end_sil then 
    tr_string = tr_string .. " " .. end_sil
  end
  local themodel
  -- generamos el modelo a partir del diccionario fonetico, necesita
  -- name mangling
  local tbl = string.tokenize(tr_string)
  for i=1,#tbl do
    if not aux_local_vars.mangling[tbl[i]] and not silences[tbl[i]] then
      error("Word without transcription: "..tbl[i])
    end
    tbl[i]=aux_local_vars.mangling[tbl[i]] or tbl[i]
  end
  return generate_models_from_sequences(tbl,
					hmmtrainer,nil,silences)
end



function generate_models_from_sequences(tbl,
					hmmtrainer,
					tied,
					silences,
					output)
  local result = {}
  if tied then
    w = tied:search_triphone_sequence(tbl)
    for i=1,#w do
      w[i] = tied:get_model(w[i])
    end
  else w = tbl
  end
  local desc = HMMTrainer.utils.tbl2model_desc(w,
					       silences,
					       output)
  local aux = hmmtrainer:model(desc)
  --  print(aux:to_string())
  collectgarbage("collect")
  return aux:generate_C_model()
end


--------------
-- datasets --
--------------



-- contextualiza la parametrizacion
function contextCCdataset(m,leftcontext,rightcontext,means_and_devs,dataset_perturbation_conf,dataset_step)
  local numFrames = m:dim()[1]
  local numParams = m:dim()[2] -- nCCs+1
  local parameters = {
    patternSize = {dataset_step, numParams},
    offset      = {0,0}, -- default value
    stepSize    = {dataset_step, 0}, -- default value, second value is not important
    numSteps    = {numFrames/dataset_step, 1}
  }
  local orig_ds = dataset.matrix(m, parameters)
  if means_and_devs then
    --print(means_and_devs.means,#means_and_devs.means,means_and_devs.devs,#means_and_devs.devs)
    orig_ds:normalize_mean_deviation(means_and_devs.means,
				     means_and_devs.devs) end
  if dataset_perturbation_conf then
    dataset_perturbation_conf.dataset = orig_ds
    orig_ds = dataset.perturbation(dataset_perturbation_conf)
  end
  local ds = dataset.contextualizer (orig_ds,leftcontext,rightcontext)
  return ds,orig_ds
end

-- carga las etiquetas de salida de la red
-- a partir de la tabla de segmentaciones
function dslabels(mat,dictionary)
  local numFrames =  mat:dim()[1] -- table.getn(m_tbl)
  local parameters = {
    patternSize = {1},
    offset      = {0},
    stepSize    = {1}, -- default value
    numSteps    = {numFrames}
  }
  local ds_aux = dataset.matrix(mat,parameters)
  local ds = dataset.indexed(ds_aux,{dictionary})
  return ds
end

-------------------------------------

function generate_emitions_matrix_and_dataset(seg_tbl, frames, mlp_output_dictionary)
  local previous = 1
  local i = 0
  local time,phone
  local emissions
  local num_emissions
  -- creamos la matriz expandiendo los valores
  local aux_mat = matrix(frames,seg_tbl)
  local aux_ds = dslabels(aux_mat,
			  mlp_output_dictionary)
  return aux_mat,aux_ds
end

function asisted_recognition(dataflow, parser, correcta, correcta_words, dictionary, cronometro)
  -- RECONOCIMIENTO
  function corregir_reconocimiento(path, reconocida, correcta_words)
    local valid_prefix = {}
    for i=1,#reconocida do
      if correcta_words[i] == reconocida[i] then
	valid_prefix[i] = path[i]
      else
	break
      end
    end
    if #valid_prefix >= #correcta_words - 1 then
      if #valid_prefix ~= #correcta_words then
	return valid_prefix,1
      end
      return valid_prefix,0
    end
    return valid_prefix,1
  end
  ------------------------------------------------------------
  local num_corrections = 0
  
  print ("RUN")
  t1 = cronometro:read()
  cronometro:go()
  resul=the_dataflow:run()
  cronometro:stop()
  t2 = cronometro:read()
  
  print(t2-t1)
  
  if not resul then
    print("Error al hacer dataflow:run()")
  end
  
  -- GET BESTPATH
  path,prob,logprob    = model_parser:get_best_path()
  
end
