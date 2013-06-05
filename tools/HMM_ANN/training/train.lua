-- This script is for HMM-ANN training using EM algorithm
--
-- Following algorithm is a general description of how it works:
--
-- Viterbi initial segmentation or use a given one
-- For each em iteration or until stop criterion
--   Train MLP until stop criterion
--   Viterbi segmentation using previous MLP
-- End
--
april_print_script_header(arg)

dofile(string.get_path(arg[0]) .. "../utils.lua")
optargs = dofile(string.get_path(arg[0]) .. "/cmdopt.lua")

table.unpack_on(optargs, _G)
trainfile_mfc    = optargs.train_m
trainfile_fon    = optargs.train_f
trainfile_sgm    = optargs.train_s
valfile_mfc      = optargs.val_m
valfile_fon      = optargs.val_f
valfile_sgm      = optargs.val_s
means_and_devs   = feats_norm
dataset_step     = dataset_step
format           = feats_format
error_function   = "multi_class_cross_entropy"
bunch_size       = 32

if trainfile_sgm and not valfile_sgm then
  error ("Needed validation initial segmentation")
end

if initial_mlp then
  -- sin segmentacion inicial
  trainfile_sgm = nil
  valfile_sgm   = nil
end

if not num_states and not initial_hmm then
  error ("Needs a number of states per HMM or an initial HMM file")
end

if not silences then
  fprintf(io.stderr, "# WARNING!!! NOT SILENCES TABLE DEFINED\n")
end

if transcription_filter then
  fprintf(io.stderr, "# Using transcription filter")
end

-- pasar esto como parametro
if var > 0 then
  dataset_perturbation_conf = {
    dataset   = nil,
    random    = seedp,
    mean      = mean,
    variance  = var,
  }
end

if initial_mlp or initial_em_epoch then
  if not initial_hmm or not initial_em_epoch or not initial_mlp then
    error("Continue a training needs: initial-em-epoch, initial-mlp, initial-hmm")
  end
end

------------------------------------------------------------------------------

-- parametros LOG
logfile    = "train.log"
dir_models = "models"
dir_redes  = dir_models.."/redes"
dir_hmms   = dir_models.."/hmms"

--------------------
-- parametros HMM --
--------------------
hmm = {}
hmm.num_states = optargs.num_states
hmm.num_params = optargs.n

hmm.silences = {}
for _,sil in ipairs(silences) do
  hmm.silences[sil] = 1.0
end

--------------------
-- parametros RNA --
--------------------
ann = {}
ann.left_context           = context
ann.right_context          = context
ann.num_hidden1           = h1
ann.num_hidden2           = h2
ann.first_learning_rate    = firstlr
ann.num_epochs_first_lr    = epochs_firstlr
ann.learning_rate       = lr
ann.momentum            = mt
ann.weight_decay           = wd
ann.weights_seed            = seed1
ann.shuffle_seed        = seed2
ann.shuffle_seed_val    = seed3
ann.replacement            = (train_r ~= 0 and train_r) or nil
ann.replacement_val        = (val_r   ~= 0 and val_r  ) or nil
ann.rndw                   = rndw

-------------------
-- parametros EM --
-------------------
em = {}
--em.epoca_cambio_fb_viterbi = 0
em.num_epochs_without_validation          = epochs_wo_val
em.max_epochs_without_improvement         = epochs_wo_imp
em.num_emiterations_without_expectation   = epochs_wo_exp
em.num_maximization_iterations            = epochs_max
em.num_maximization_iterations_first_em = epochs_first_max
em.veces_em                               = em_iterations

--------------------------------------------------------------

--------------------
--     Corpus     --
--------------------
corpus = {}
corpus.filename_trn_mfc = trainfile_mfc
corpus.filename_val_mfc = valfile_mfc
corpus.filename_trn_fon = trainfile_fon
corpus.filename_val_fon = valfile_fon
corpus.filename_trn_sgm = trainfile_sgm
corpus.filename_val_sgm = valfile_sgm

os.execute("mkdir "..dir_models)
os.execute("mkdir "..dir_redes)
os.execute("mkdir "..dir_hmms)

-- objeto con informacion sobre modelos ligados
tied = tied_model_manager(io.open(tiedfile))

---------
-- HMM --
---------

-- creamos el trainer y le metemos los modelos acusticos
hmmtrainer = HMMTrainer.trainer()

-- modelos HMM
models = {}
if initial_hmm then
   models,num_models,num_emissions = load_models_from_hmm_lua_desc(initial_hmm,
                                                                   hmmtrainer)
else
  models,num_models = generate_hmm_models_from_tiedlist(tied,
							hmm.num_states,
							hmmtrainer)
  num_emissions     = num_models * hmm.num_states
end
ann.output_dictionary = dataset.identity(num_emissions, 0.0, 1.0)
collectgarbage("collect")

-- ENTRADAS Y SALIDAS DE LA ANN
ann.num_entradas = hmm.num_params*(ann.left_context+ann.right_context+1)*dataset_step
ann.num_outputs  = num_emissions

------------------
-- Red neuronal --
------------------
--
-- estructura de la red
ann.thenet = nil
local mlp_str    = nil
if ann.num_hidden2 > 0 then
  mlp_str = string.format("%d inputs %d logistic %d logistic %d log_softmax",
			  ann.num_entradas,
			  ann.num_hidden1,
			  ann.num_hidden2,
			  ann.num_outputs)
else
  mlp_str = string.format("%d inputs %d logistic %d log_softmax",
			  ann.num_entradas,
			  ann.num_hidden1,
			  ann.num_outputs)
end

-- generamos la red
if initial_mlp then
  ann.trainer = trainer.supervised_trainer.load(initial_mlp,
						ann.loss[error_function],
						bunch_size)
  ann.thenet  = ann.trainer:get_component()
else
  ann.thenet  = ann.mlp.all_all.generate(mlp_str)
  ann.trainer = trainable.supervised_trainer(ann.thenet,
					     ann.loss[error_function],
					     bunch_size)
  ann.trainer:randomize_weights{
    inf = ann.rndw,
    sup = ann.rndw,
    random = random(ann.weights_seed)
  }
end

collectgarbage("collect")

--------------------
-- Carga de datos --
--------------------
--
--[[
La funcion corpus_from_audio_filename carga un field manager con estos campos:
  - audio_filename        => Nombre de los ficheros de audio
  - frames        => Acusticas de cada fichero
  - frames_size   => Numero de tramas en cada fichero
  - hmm_emission_sequence => Tabla con la emision que corresponde a cada trama
  - hmm_c_models          => Modelo de Markov de cada fichero
  - contextCCdataset      => Dataset contextualizado con las tramas
  - segmentation_matrix   => Matriz con la segmentacion inicial (la mejor)
  - segmentation_dataset  => Dataset para la segmentation_matrix
--]]

--
-- TRAINING
print ("# LOADING TRAINING...")
training = {}
if not string.match(corpus.filename_trn_mfc, "%.lua$") then
  if count_values[1] then
    error ("Count values are forbidden without distribution!!!")
  end
  table.insert(training,
	       corpus_from_MFC_filename{
		 filename_mfc           = corpus.filename_trn_mfc,
		 filename_fon           = corpus.filename_trn_fon,
		 filename_sgm           = corpus.filename_trn_sgm,
		 phdict                 = train_phdict,
		 begin_sil              = begin_sil,
		 end_sil                = end_sil,
		 hmmtrainer                = hmmtrainer,
		 tied                   = tied,
		 left_context           = ann.left_context,
		 right_context          = ann.right_context,
		 mlp_output_dictionary  = ann.output_dictionary,
		 models                 = models,
		 silences               = hmm.silences,
		 transcription_filter   = transcription_filter,
		 hmm_name_mangling      = false,
		 dataset_step           = dataset_step,
		 means_and_devs         = means_and_devs,
		 dataset_perturbation_conf = dataset_perturbation_conf,
		 format = format,
	       })
else
  local mfc = dofile(corpus.filename_trn_mfc)
  local fon = dofile(corpus.filename_trn_fon)
  local sgm = {}
  if #mfc ~= #fon then
    error ("Incorrect number of fields in Lua MFC or FON file")
  end
  if corpus.filename_trn_sgm then
    sgm = dofile(corpus.filename_trn_sgm)
    if #mfc ~= #sgm then
      error ("Incorrect number of fields in Lua SGM file")
    end
  end
  corpus.distribution = {}
  training = {}
  for i=1,#mfc do
    if not count_values[i] then
      error ("Incorrect number of count_values parameter")
    end
    sgm[i] = sgm[i] or {}
    
    if not sgm[i][1] then
      if not initial_mlp then
	print("WARNING!!! training with equidistant initial segmentation for " .. mfc[i][1])
	fprintf(flog, "# WARNING!!! training with equidistant initial segmentation for %s\n",
		mfc[i][1])
      else
	print("Ignoring initial segmentation, MLP given")
	fprint(flog, "# Ignoring initial segmentation, MLP given")
      end
    end
    
    table.insert(training,
		 corpus_from_MFC_filename{
		   filename_mfc           = mfc[i][1] or error("Incorrect MFC field"),
		   filename_fon           = fon[i][1] or error("Incorrect FON field"),
		   filename_sgm           = sgm[i][1],
		   phdict                 = train_phdict,
		   begin_sil              = begin_sil,
		   end_sil                = end_sil,
		   hmmtrainer                = hmmtrainer,
		   tied                   = tied,
		   left_context           = ann.left_context,
		   right_context          = ann.right_context,
		   mlp_output_dictionary  = ann.output_dictionary,
		   models                 = models,
		   silences               = hmm.silences,
		   transcription_filter   = transcription_filter,
		   hmm_name_mangling      = false,
		   dataset_step           = dataset_step,
		   means_and_devs         = means_and_devs,
		   dataset_perturbation_conf = dataset_perturbation_conf,
		   format = format,
		 })
    table.insert(corpus.distribution, {
		   --                                 OJO: i == #training
		   input_dataset  = dataset.union(training[i]:get_field('contextCCdataset')),
		   output_dataset = dataset.union(training[i]:get_field('segmentation_dataset')),
		   probability    = tonumber(mfc[i][2])
		 })
  end
end
-- VALIDATION
print ("# LOADING VALIDATION...")
validate = corpus_from_MFC_filename{
  filename_mfc           = corpus.filename_val_mfc,
  filename_fon           = corpus.filename_val_fon,
  filename_sgm           = corpus.filename_val_sgm,
  phdict                 = val_phdict,
  begin_sil              = begin_sil,
  end_sil                = end_sil,
  hmmtrainer                = hmmtrainer,
  tied                   = tied,
  left_context           = ann.left_context,
  right_context          = ann.right_context,
  mlp_output_dictionary  = ann.output_dictionary,
  models                 = models,
  silences               = hmm.silences,
  transcription_filter   = transcription_filter,
  hmm_name_mangling      = false,
  dataset_step           = dataset_step,
  means_and_devs         = means_and_devs,
  format = format,
}

-- ESTO ERA PARA EVITAR QUE PETARA CON LOS MODELOS QUE NO APARECIESEN
-- EN LAS FRASES:
--aux_c_tbl = {}
--for i=1,table.getn(modeloAcustico.phones_list) do
--  local ph = modeloAcustico.phones_list[i].name[1]
--  local desc = HMMTrainer.utils.str2model_desc(ph)
--  aux = hmmtrainer:model(desc)
--  table.insert(aux_c_tbl, aux:generate_C_model())
--end
--

--------------------------
-- creamos los datasets --
--------------------------
-- TRAINING
if not corpus.distribution then
  corpus.input_ds_trn  = dataset.union(training[1]:get_field('contextCCdataset'))
  corpus.output_ds_trn = dataset.union(training[1]:get_field('segmentation_dataset'))
end
-- VALIDATION
corpus.input_ds_val  = dataset.union(validate:get_field('contextCCdataset'))
corpus.output_ds_val = dataset.union(validate:get_field('segmentation_dataset'))

printf ("\n")
printf ("# RANDOM:     Weights: %4d Shuffle: %4d  VAL RPL shuffle: %4d\n",
	ann.weights_seed, ann.shuffle_seed, ann.shuffle_seed_val)
printf ("# ANN PARAMS: step:%d l%d r%d %d %d  lr: %f  mt: %f  wd: %g\n",
	dataset_step, ann.left_context, ann.right_context,
	ann.num_hidden1, ann.num_hidden2,
	ann.learning_rate, ann.momentum, ann.weight_decay)
printf ("#             first lr: %f    epochs: %d\n",
	ann.first_learning_rate, ann.num_epochs_first_lr)
printf ("# ANN STR:    %s\n", mlp_str)
printf ("# HMM PARAMS: states: %d   numparams: %d\n",
	hmm.num_states or 0, hmm.num_params)
printf ("# EM  PARAMS:  val: %d  stop: %d  expectation: %d  it: %d %d  em: %d\n",
	em.num_epochs_without_validation,
	em.max_epochs_without_improvement,
	em.num_emiterations_without_expectation,
	em.num_maximization_iterations_first_em,
	em.num_maximization_iterations,
	em.veces_em)
if not corpus.distribution then
  printf ("# Patrones de entrenamiento: %d\n", corpus.input_ds_trn:numPatterns())
else
  printf ("# Patrones de entrenamiento:")
  for i=1,#corpus.distribution do
    printf (" %d (%f)", corpus.distribution[i].input_dataset:numPatterns(),
	    corpus.distribution[i].probability)
  end
  printf ("\n")
end
printf ("# Patrones de validacion: %d\n", corpus.input_ds_val:numPatterns())
printf ("# Replacement:     %d\n", ann.replacement or 0)
printf ("# Replacement val: %d\n", ann.replacement_val or 0)

-- log
if initial_mlp then
  os.execute(string.format("mv -f %s %s.%s",
			   logfile, logfile, os.date("%Y-%m-%d-%H:%M")))
end
flog = io.open(logfile, "w")
fprintf(flog,"# HOST:\t %s\n", (io.popen("hostname", "r"):read("*l")))
fprintf(flog,"# DATE:\t %s\n", (io.popen("date", "r"):read("*l")))
fprintf(flog,"# CMD: \t %s\n", table.concat(arg, " "))

if not trainfile_sgm then
  if not initial_mlp then
    print("WARNING!!! training with equidistant initial segmentation")
    fprintf(flog, "# WARNING!!! training with equidistant initial segmentation\n")
  else
    print("Ignoring initial segmentation")
    fprint(flog, "# Ignoring initial segmentation")
  end
end

if initial_mlp then
  fprintf (flog, "# CONTINUE TRAINING: %s %s %d\n",
	   initial_mlp, initial_hmm, initial_em_epoch)
elseif initial_hmm then
  fprintf (flog, "# INITIAL HMM: %s\n", initial_hmm)
end
fprintf (flog, "# RANDOM:     Weights: %4d Shuffle: %4d VAL REPL: %4d\n",
	 ann.weights_seed, ann.shuffle_seed, ann.shuffle_seed_val)
fprintf (flog, "# ANN PARAMS: step:%d l%d r%d %d %d  lr: %f  mt: %f  wd: %g\n",
	 dataset_step, ann.left_context, ann.right_context,
	 ann.num_hidden1, ann.num_hidden2,
	 ann.learning_rate, ann.momentum, ann.weight_decay)
fprintf (flog, "#             first lr: %f    epochs: %d    err_func: %s\n",
	 ann.first_learning_rate, ann.num_epochs_first_lr, error_function)
fprintf (flog, "# ANN STR:    %s\n", mlp_str)
fprintf (flog, "# HMM PARAMS: states: %d   numparams: %d\n",
	 hmm.num_states or 0, hmm.num_params)
fprintf (flog, "# EM  PARAMS:  val: %d  stop: %d  expectation: %d  it: %d %d  em: %d\n",
	 em.num_epochs_without_validation,
	 em.max_epochs_without_improvement,
	 em.num_emiterations_without_expectation,
	 em.num_maximization_iterations_first_em,
	 em.num_maximization_iterations,
	 em.veces_em)
if not corpus.distribution then
  fprintf (flog, "# Training num patterns: %d\n", corpus.input_ds_trn:numPatterns())
else
  fprintf (flog, "# Training num patterns:")
  for i=1,#corpus.distribution do
    fprintf (flog, " %d (%f)", corpus.distribution[i].input_dataset:numPatterns(),
	     corpus.distribution[i].probability)
  end
  fprintf (flog, "\n")
end
fprintf (flog, "# Validation num patterns: %d\n", corpus.input_ds_val:numPatterns())
fprintf (flog, "# Replacement:     %d\n", ann.replacement or 0)
fprintf (flog, "# Replacement val: %d\n", ann.replacement_val or 0)
flog:flush()

ann.trainingdata = {
  shuffle        = random(ann.shuffle_seed),
  replacement    = ann.replacement,
}

if not corpus.distribution then
  ann.trainingdata.input_dataset  = corpus.input_ds_trn
  ann.trainingdata.output_dataset = corpus.output_ds_trn
else
  ann.trainingdata.distribution = corpus.distribution
end

ann.validationdata = {
  input_dataset  = corpus.input_ds_val,
  output_dataset = corpus.output_ds_val,
}

if ann.replacement_val then
  ann.validationdata.replacement = ann.replacement_val
  ann.validationdata.shuffle     = random(ann.shuffle_seed_val)
end

ann.thenet:set_option("learning_rate", ann.first_learning_rate)
ann.thenet:set_option("momentum",      ann.momentum)
ann.thenet:set_option("weight_decay",  ann.weight_decay)

collectgarbage("collect")

--------------------------------------------------

--------------
-- TRAINING --
--------------
totaltrain = 0
em_iteration = 1

bestmse      = 1
bestmse_em   = 1
bestepoch    = 0
bestepoch_em = 0
best_trainer = ann.trainer

if initial_em_epoch then
  em_iteration = initial_em_epoch
  
  -- resegmentamos validacion
  generate_new_segmentation{
    field_manager  = validate,
    func           = ann.trainer,
    num_emissions  = num_emissions,
    do_expectation = false,
    emission_in_log_base = true,
  }
  
  -- reinicializamos contadores del trainer de hmm
  hmmtrainer.trainer:begin_expectation()
  --
  -- EXPECTATION
  --
  -- resegmentamos training con expectacion
  for i=1,#training do
    generate_new_segmentation{
      field_manager  = training[i],
      func           = ann.trainer,
      num_emissions  = num_emissions,
      count_value    = count_values[i],
      do_expectation = true,
      emission_in_log_base = true,
    }
  end
  
  --
  collectgarbage("collect")
  --
  hmmtrainer.trainer:end_expectation{
    update_trans_prob = (em_iteration > em.num_emiterations_without_expectation),
    update_a_priori_emission = true,
  }
  -- actualizamos las probabilidades de transitar de los modelos
  for name,model_info in pairs(models) do
    model_info.model:update_probs()
  end

end

while em_iteration <= em.veces_em do
  --
  -- entrenamos la red neuronal
  max = em.num_maximization_iterations
  if em_iteration == 1 then max = em.num_maximization_iterations_first_em end
  bestepoch = totaltrain
  global_best_trainer = ann.trainer:clone()
  epoch = 0
  for epoch = 1,em.num_epochs_without_validation do
    totaltrain = totaltrain+1
    if totaltrain > ann.num_epochs_first_lr or initial_mlp then
      ann.thenet:set_option("learning_rate", ann.learning_rate)
    end
    errortrain = ann.trainer:train_dataset(ann.trainingdata)
    errorval   = ann.trainer:validate_dataset(ann.validationdata) -- NO SE UTILIZA
    printf("em %4d epoch %4d totalepoch %4d mse_train %.7f mse_val "..
	     "%.7f %10d %.7f\n",
	   em_iteration,epoch,totaltrain,errortrain,errorval,bestepoch,
	   errorval)
    fprintf(flog, "em %4d epoch %4d totalepoch %4d mse_train %.7f mse_val "..
	      "%.7f %10d %.7f\n",
	    em_iteration,epoch,totaltrain,errortrain,errorval,bestepoch,
	    errorval)
    flog:flush()
  end
  bestmse      = ann.trainer:validate_dataset(ann.validationdata)
  firstbestmse = bestmse
  printf ("# FIRST BEST MSE FOR EM: %.7f\n", firstbestmse)
  bestepoch    = totaltrain
  for epoch = em.num_epochs_without_validation+1,max do
    totaltrain = totaltrain+1
    if totaltrain > ann.num_epochs_first_lr then
      --ann.trainingdata.learning_rate = ann.learning_rate
      if ( em_iteration == 1 and
	     math.mod(totaltrain, 10) == 1 and
	   ann.thenet:get_option("learning_rate") > 0.001 ) then
	ann.thenet:set_option("learning_rate",
			      ann.thenet:get_option("learning_rate") - 0.001)
      end
    end
    errortrain = ann.trainer:train_dataset(ann.trainingdata)
    errorval   = ann.trainer:validate_dataset(ann.validationdata)
    if errorval < bestmse then
      bestmse    = errorval
      bestepoch  = totaltrain
      best_trainer = ann.trainer:clone()
      collectgarbage("collect")
    end
    if totaltrain - bestepoch > em.max_epochs_without_improvement then
      --and em_iteration > 1 then
      break
    end
    printf("em %4d epoch %4d totalepoch %4d mse_train %.7f mse_val "..
	   "%.7f %10d %.7f\n",
	 em_iteration,epoch,totaltrain,errortrain,errorval,bestepoch,
	 bestmse)
    fprintf(flog, "em %4d epoch %4d totalepoch %4d mse_train %.7f mse_val "..
	    "%.7f %10d %.7f\n",
	  em_iteration,epoch,totaltrain,errortrain,errorval,bestepoch,
	  bestmse)
    flog:flush()
    --      if totaltrain - bestepoch > 5 and epoch > 5 then
    --     break
    --      end
  end
  -- nos quedamos con la mejor red ;)
  totaltrain   = bestepoch
  ann.trainer  = best_trainer
  ann.thenet   = best_trainer:get_component()
  --print("tiempo entrenar red ",os.clock() - t1)
  if firstbestmse <= bestmse then
    ann.trainer = global_best_trainer
    ann.thenet  = global_best_trainer:get_component()
    --    break -- BREAK DEL EM
  end
  if bestmse < bestmse_em then
    bestmse_em   = bestmse
    bestepoch_em = bestepoch
  end
  collectgarbage("collect")
  
  -- AHORA USAMOS LA RED ENTRENADA PARA REALIZAR ALINEAMIENTO FORZADO
  -- VITERBI Y RESEGMENTAR
  
  -- resegmentamos validacion
  generate_new_segmentation{
    field_manager  = validate,
    func           = ann.trainer,
    num_emissions  = num_emissions,
    do_expectation = false,
    emission_in_log_base = true,
  }
  
  -- reinicializamos contadores del trainer de hmm
  hmmtrainer.trainer:begin_expectation()
  --
  -- EXPECTATION
  --
  -- resegmentamos training con expectacion
  for i=1,#training do
    generate_new_segmentation{
      field_manager  = training[i],
      func           = ann.trainer,
      num_emissions  = num_emissions,
      count_value    = count_values[i],
      do_expectation = true,
      emission_in_log_base = true,
    }
  end

  --
  collectgarbage("collect")
  --
  hmmtrainer.trainer:end_expectation{
    update_trans_prob = (em_iteration > em.num_emiterations_without_expectation),
    update_a_priori_emission = true,
  }
  -- actualizamos las probabilidades de transitar de los modelos
  for name,model_info in pairs(models) do
    model_info.model:update_probs()
  end

  -- salvamos a disco los modelos
  local filem = string.format(dir_hmms .. "/ci%d_cd%d_em%d.lua",
			      ann.left_context,
			      ann.right_context,
			      em_iteration)
  print("salvando "..filem)
  local modelf = io.open(filem, "w")
  modelf:write("return { {\n")
  for name,model_info in pairs(models) do
    modelf:write("['".. string.gsub(name,"'","\\'") .. "'] = {\n")
    modelf:write("\tmodel=HMMTrainer.model.from_table(\n")
    modelf:write(model_info.model:to_string())
    modelf:write("),\n")
    modelf:write("\temissions={".. table.concat(model_info.emissions,
						",") .."}\n")
    modelf:write("},\n")
  end
  modelf:write("},\n")
  modelf:write("{".. table.concat(hmmtrainer.trainer:get_a_priori_emissions(),
				  ",") .."}\n")
  modelf:write("}\n")
  modelf:close()
  --
  collectgarbage("collect")
  
  -- salvamos la red
  filenet = string.format(dir_redes .. "/ci%d_cd%d_em%d.net",
			  ann.left_context,
			  ann.right_context,
			  em_iteration)
  print("salvando "..filenet)
  ann.trainer:save(filenet,"binary")
  
  --
  
  flog:flush()
  
  collectgarbage("collect")
  
  em_iteration = em_iteration + 1
end
flog:close()
