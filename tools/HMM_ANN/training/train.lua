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
dataset_step     = optargs.step
format           = feats_format
error_function   = "multi_class_cross_entropy"

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

if not silences or #silences==0 then
  fprintf(io.stderr, "# WARNING!!! NOT SILENCES TABLE DEFINED\n")
end

if not begin_sil or not end_sil then
  fprintf(io.stderr, "# WARNING!!! NOT BEGIN AND END SILENCE SYMBOLS DEFINED\n")
end

if transcription_filter then
  fprintf(io.stderr, "# Using transcription filter\n")
end

-- pasar esto como parametro
local perturbation_random = random(seedp)
if var > 0 then
  dataset_perturbation_function = function(ds)
    return dataset.perturbation{
      dataset   = ds,
      random    = perturbation_random,
      mean      = mean,
      variance  = var,
    }
  end
end

if salt > 0 then
  local old_dataset_perturbation_function = dataset_perturbation_function
  dataset_perturbation_function = function(ds)
    return dataset.salt_noise{
      dataset   = old_dataset_perturbation_function(ds),
      vd        = salt,
      zero      = 0,
      random    = perturbation_random,
    }
  end
end

if initial_mlp or initial_em_epoch then
  if not initial_hmm or not initial_em_epoch or not initial_mlp then
    error("Continue a training needs: initial-em-epoch, initial-mlp, initial-hmm")
  end
end

------------------------------------------------------------------------------

models_dir = "models"
nets_dir   = models_dir.."/nets"
hmms_dir   = models_dir.."/hmms"

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
ann_table = {}
ann_table.left_context         = context
ann_table.right_context        = context
ann_table.num_hidden1          = h1
ann_table.num_hidden2          = h2
ann_table.first_learning_rate  = firstlr
ann_table.num_epochs_first_lr  = epochs_firstlr
ann_table.learning_rate        = lr
ann_table.momentum             = mt
ann_table.max_norm_penalty     = mp
ann_table.weight_decay         = wd
ann_table.weights_seed         = seed1
ann_table.shuffle_seed         = seed2
ann_table.shuffle_seed_val     = seed3
ann_table.replacement          = (train_r ~= 0 and train_r) or nil
ann_table.replacement_val      = (val_r   ~= 0 and val_r  ) or nil
ann_table.rndw                 = rndw

-------------------
-- parametros EM --
-------------------
em = {}
--em.epoca_cambio_fb_viterbi = 0
em.num_epochs_without_validation          = epochs_wo_val
em.max_epochs_without_improvement         = epochs_wo_imp
em.num_emiterations_without_expectation   = epochs_wo_exp
em.num_maximization_iterations            = epochs_max
em.num_maximization_iterations_first_em   = epochs_first_max
em.em_max_iterations                      = em_it

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

os.execute("mkdir "..models_dir)
os.execute("mkdir "..nets_dir)
os.execute("mkdir "..hmms_dir)

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
  num_emissions = num_models * hmm.num_states
  hmmtrainer.trainer:check_cls_emission(num_emissions)
end
ann_table.output_dictionary = dataset.identity(num_emissions, 0.0, 1.0)
collectgarbage("collect")

-- ENTRADAS Y SALIDAS DE LA ANN
ann_table.num_entradas = hmm.num_params*(ann_table.left_context+ann_table.right_context+1)*dataset_step
ann_table.num_outputs  = num_emissions

------------------
-- Red neuronal --
------------------
--
-- estructura de la red
local best_trainer
local global_best_trainer
local mlp_str
local weights_random = random(ann_table.weights_seed)
function initialize_neural_network()
  print("# INITIALIZING THE ANN")
  ann_table.thenet = nil
  mlp_str = nil
  if ann_table.num_hidden2 > 0 then
    mlp_str = string.format("%d inputs %d logistic %d logistic",
			    ann_table.num_entradas,
			    ann_table.num_hidden1,
			    ann_table.num_hidden2)
  else
    mlp_str = string.format("%d inputs %d logistic",
			    ann_table.num_entradas,
			    ann_table.num_hidden1)
  end
  -- generamos la red
  if initial_mlp then
    mlp_str = initial_mlp
    ann_table.trainer = trainable.supervised_trainer.load(initial_mlp,
							  ann.loss[error_function](ann_table.num_outputs),
							  bunch_size)
    ann_table.thenet  = ann_table.trainer:get_component()
  elseif pretrained_mlp then
    mlp_str = pretrained_mlp
    ann_table.trainer = trainable.supervised_trainer.load(pretrained_mlp,
							  ann.loss[error_function](ann_table.num_outputs),
							  bunch_size)
    ann_table.thenet  = ann_table.trainer:get_component()
  else
    ann_table.thenet = ann.components.stack()
    mlp_str = string.format("%d inputs",ann_table.num_entradas)
    --
    if ann_table.num_hidden1 > 0 then
      mlp_str = string.format("%s %d logistic", mlp_str, ann_table.num_hidden1)
      ann_table.thenet:push(ann.components.hyperplane{
			      dot_product_name="w1",
			      bias_name="b1",
			      dot_product_weights="w1",
			      bias_weights="b1",
			      input = ann_table.num_entradas,
			      output = ann_table.num_hidden1,
						     }):
      push(ann.components.actf.logistic{ name="actf1" })
    end
    --
    if ann_table.num_hidden2 > 0 then
      mlp_str = string.format("%s %d logistic", mlp_str, ann_table.num_hidden2)
      ann_table.thenet:push(ann.components.hyperplane{
			      dot_product_name="w2",
			      bias_name="b2",
			      dot_product_weights="w2",
			      bias_weights="b2",
			      input = ann_table.num_hidden1,
			      output = ann_table.num_hidden2,
						     }):
      push(ann.components.actf.logistic{ name="actf2" })
    end
    --
    mlp_str = string.format("%s %d log_softmax", mlp_str, ann_table.num_outputs)
    ann_table.thenet:push(ann.components.hyperplane{
			    dot_product_name="wN",
			    bias_name="bN",
			    dot_product_weights="wN",
			    bias_weights="bN",
			    output = ann_table.num_outputs,
						   }):
    push(ann.components.actf.log_softmax{ name="OUPUT" })
    --
    ann_table.trainer = trainable.supervised_trainer(ann_table.thenet,
						     ann.loss[error_function](ann_table.num_outputs),
						     bunch_size)
    ann_table.trainer:build()
    ann_table.trainer:randomize_weights{
      inf =  ann_table.rndw,
      sup = -ann_table.rndw,
      random = weights_random
    }
  end
  best_trainer = ann_table.trainer
  global_best_trainer = ann_table.trainer
end

initialize_neural_network()

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
		 left_context           = ann_table.left_context,
		 right_context          = ann_table.right_context,
		 mlp_output_dictionary  = ann_table.output_dictionary,
		 models                 = models,
		 silences               = hmm.silences,
		 transcription_filter   = transcription_filter,
		 hmm_name_mangling      = false,
		 dataset_step           = dataset_step,
		 means_and_devs         = means_and_devs,
		 dataset_perturbation_function = dataset_perturbation_function,
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
	print("# WARNING!!! training with equidistant initial segmentation for " .. mfc[i][1])
      else
	print("# Ignoring initial segmentation, MLP given")
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
		   left_context           = ann_table.left_context,
		   right_context          = ann_table.right_context,
		   mlp_output_dictionary  = ann_table.output_dictionary,
		   models                 = models,
		   silences               = hmm.silences,
		   transcription_filter   = transcription_filter,
		   hmm_name_mangling      = false,
		   dataset_step           = dataset_step,
		   means_and_devs         = means_and_devs,
		   dataset_perturbation_function = dataset_perturbation_function,
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
  left_context           = ann_table.left_context,
  right_context          = ann_table.right_context,
  mlp_output_dictionary  = ann_table.output_dictionary,
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

printf ("# RANDOM:     Weights: %4d Shuffle: %4d  VAL RPL shuffle: %4d\n",
	ann_table.weights_seed, ann_table.shuffle_seed, ann_table.shuffle_seed_val)
printf ("# ANN PARAMS: step:%d l%d r%d %d %d  lr: %f  mt: %f  wd: %g\n",
	dataset_step, ann_table.left_context, ann_table.right_context,
	ann_table.num_hidden1, ann_table.num_hidden2,
	ann_table.learning_rate, ann_table.momentum, ann_table.weight_decay)
printf ("#             first lr: %f    epochs: %d\n",
	ann_table.first_learning_rate, ann_table.num_epochs_first_lr)
printf ("# ANN STR:    %s\n", mlp_str)
printf ("# HMM PARAMS: states: %d   numparams: %d\n",
	hmm.num_states or 0, hmm.num_params)
printf ("# EM  PARAMS:  val: %d  stop: %d  expectation: %d  it: %d %d  em: %d\n",
	em.num_epochs_without_validation,
	em.max_epochs_without_improvement,
	em.num_emiterations_without_expectation,
	em.num_maximization_iterations_first_em,
	em.num_maximization_iterations,
	em.em_max_iterations)
if not corpus.distribution then
  printf ("# Input pattern size: %d\n", corpus.input_ds_trn:patternSize())
  printf ("# Training num patterns: %d\n", corpus.input_ds_trn:numPatterns())
else
  printf ("# Input pattern size: %d\n", corpus.distribution[1].input_dataset:patternSize())
  printf ("# Training num patterns:")
  for i=1,#corpus.distribution do
    printf (" %d (%f)", corpus.distribution[i].input_dataset:numPatterns(),
	    corpus.distribution[i].probability)
  end
  printf ("\n")
end
printf ("# Validation num patterns: %d\n", corpus.input_ds_val:numPatterns())
printf ("# Replacement:     %d\n", ann_table.replacement or 0)
printf ("# Replacement val: %d\n", ann_table.replacement_val or 0)

-- log
if not trainfile_sgm then
  if not initial_mlp then
    print("WARNING!!! training with equidistant initial segmentation")
  else
    print("# Ignoring initial segmentation")
  end
end

if initial_mlp and initial_hmm then
  printf ("# CONTINUE TRAINING: %s %s %d\n",
	  initial_mlp, initial_hmm, initial_em_epoch)
elseif initial_hmm then
  printf ("# INITIAL HMM: %s\n", initial_hmm)
elseif initial_mlp then
  printf ("# INITIAL MLP: %s\n", initial_mlp)
end

ann_table.trainingdata = {
  shuffle        = random(ann_table.shuffle_seed),
  replacement    = ann_table.replacement,
}

if not corpus.distribution then
  ann_table.trainingdata.input_dataset  = corpus.input_ds_trn
  ann_table.trainingdata.output_dataset = corpus.output_ds_trn
else
  ann_table.trainingdata.distribution = corpus.distribution
end

ann_table.validationdata = {
  input_dataset  = corpus.input_ds_val,
  output_dataset = corpus.output_ds_val,
}

if ann_table.replacement_val then
  ann_table.validationdata.replacement = ann_table.replacement_val
  ann_table.validationdata.shuffle     = random(ann_table.shuffle_seed_val)
end

collectgarbage("collect")

--------------------------------------------------

--------------
-- TRAINING --
--------------
totaltrain = 1
em_iteration = 1

firstbestce = 11111111111
bestce      = 11111111111
bestce_em   = 11111111111
bestepoch    = 0
bestepoch_em = 0

if initial_em_epoch then
  em_iteration = initial_em_epoch
  
  -- resegmentamos validacion
  local va_score = generate_new_segmentation{
    field_manager  = validate,
    func           = ann_table.trainer,
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
  local tr_score = 0
  for i=1,#training do
    local sc = generate_new_segmentation{
      field_manager  = training[i],
      func           = ann_table.trainer,
      num_emissions  = num_emissions,
      count_value    = count_values[i],
      do_expectation = true,
      emission_in_log_base = true,
    }
    tr_score = tr_score + sc
  end
  
  print("# INITIAL HMM SCORE TR= ", tr_score)
  print("# INITIAL HMM SCORE VA= ", va_score)
  
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

while em_iteration <= em.em_max_iterations do
  --
  -- entrenamos la red neuronal
  max = em.num_maximization_iterations
  if em_iteration == 1 then max = em.num_maximization_iterations_first_em end
  bestepoch = totaltrain
  global_best_trainer = ann_table.trainer:clone()
  if em_iteration > 1 then
    ann_table.thenet:set_option("learning_rate", ann_table.learning_rate)
  else
    ann_table.thenet:set_option("learning_rate", ann_table.first_learning_rate)
  end
  ann_table.thenet:set_option("momentum",         ann_table.momentum)
  ann_table.thenet:set_option("weight_decay",     ann_table.weight_decay)
  ann_table.thenet:set_option("max_norm_penalty", ann_table.max_norm_penalty)
  if dropout > 0 then
    iterator(ipairs(dropout_list)):
    iterate(function(i,name)
	      return ann_table.trainer:iterate_components(name)
	    end):
    apply(function(cname,component)
	    printf("# DROPOUT OF COMPONENT %s\n", cname)
	    component:set_option("dropout_factor", dropout)
	    component:set_option("dropout_seed",   dropout_seed)
	  end)
  end
  --------------------------------------
  -- ANN TRAINING (MAXIMIZATION STEP) --
  --------------------------------------
  local result = ann_table.trainer:train_holdout_validation{
    training_table = ann_table.trainingdata,
    validation_table = ann_table.validationdata,
    epochs_wo_validation = em.num_epochs_without_validation,
    min_epochs = em.num_epochs_without_validation,
    max_epochs = max,
    stopping_criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_absolute(em.max_epochs_without_improvement),
    update_function = function(t)
      printf("em %4d epoch %4d totalepoch %4d ce_train %.7f ce_val "..
	       "%.7f %10d %.7f  %.7f\n",
	     em_iteration, t.current_epoch, totaltrain,
	     t.train_error, t.validation_error, bestepoch,
	     t.best_val_error, ann_table.thenet:get_option("learning_rate"))
      totaltrain = totaltrain + 1
      if (totaltrain > ann_table.num_epochs_first_lr and
	    em_iteration == 1 and
	    totaltrain % 10 == 1 and
	  ann_table.thenet:get_option("learning_rate") > ann_table.learning_rate ) then
	ann_table.thenet:set_option("learning_rate",
				    ann_table.thenet:get_option("learning_rate") - 0.001)
      end
      if t.current_epoch == em.num_epochs_without_validation then
	firstbestce = t.validation_error
	printf ("# FIRST BEST CE FOR EM: %.7f\n", firstbestce)
      end
      io.stdout:flush()
    end,
    best_function = function(best_trainer)
      bestepoch = totaltrain
    end,
  }
  -- nos quedamos con la mejor red ;)
  totaltrain = bestepoch
  ann_table.trainer = result.best
  ann_table.thenet  = result.best:get_component()
  --print("tiempo entrenar red ",os.clock() - t1)
  if bestepoch == em.num_epochs_without_validation then
    ann_table.trainer = global_best_trainer:clone()
    ann_table.thenet  = ann_table.trainer:get_component()
    --    break -- BREAK DEL EM
  end
  if result.best_val_error < bestce_em then
    bestce_em    = result.best_val_error
    bestepoch_em = bestepoch
  end
  collectgarbage("collect")
  
  ------------------------------------------
  -- VITERBI ALIGNMENT (EXPECTATION STEP) --
  ------------------------------------------
  
  -- resegmentamos validacion
  local va_score = generate_new_segmentation{
    field_manager  = validate,
    func           = ann_table.trainer,
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
  local tr_score = 0
  for i=1,#training do
    local sc = generate_new_segmentation{
      field_manager  = training[i],
      func           = ann_table.trainer,
      num_emissions  = num_emissions,
      count_value    = count_values[i],
      do_expectation = true,
      emission_in_log_base = true,
    }
    tr_score = tr_score + sc
  end

  print("# HMM SCORE TR= ", tr_score)
  print("# HMM SCORE VA= ", va_score)
  
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
  local filem = string.format(hmms_dir .. "/lc%d_rc%d_em%d.lua",
			      ann_table.left_context,
			      ann_table.right_context,
			      em_iteration)
  save_models_from_hmm_lua_desc(filem, models, hmmtrainer)
  --
  collectgarbage("collect")
  
  -- salvamos la red
  filenet = string.format(nets_dir .. "/lc%d_rc%d_em%d.net",
			  ann_table.left_context,
			  ann_table.right_context,
			  em_iteration)
  print("# Saving "..filenet)
  ann_table.trainer:save(filenet,"binary")
  --
  
  collectgarbage("collect")
  
  em_iteration = em_iteration + 1


  if em_iteration <= dropped_em_iterations then
    initialize_neural_network()
  end

end
