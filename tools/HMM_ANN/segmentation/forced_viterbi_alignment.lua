-- FIXME: Este SCRIPT solo tiene en cuenta trifonemas dentro de las
-- palabras, pero entre palabras

april_print_script_header(arg)
dofile(string.get_path(arg[0]) .. "../utils.lua")

cmdOptTest = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "Force Viterbi Alignment with April toolkit",
  {
    index_name="defopt",
    description="Load configuration file (a lua tabla)",
    short="f",
    argument="yes",
    filter=dofile,
  },
  { index_name="n", -- antes filenet
    description = "MLP file",
    short    = "n",
    argument = "yes",
    mode     = "always",
  },
  {
    index_name  = "m", -- antes filem
    description = "hmms file",
    short       = "m",
    argument    = "yes",
  },
  {
    index_name="hmmdefs", -- antes filehmmdefs
    description = "hmms HTK file",
    long = "hmmdefs",
    argument = "yes",
  },
  { index_name="t", -- antes tiedfile
    description = "HTK unit's tied list",
    short = "t",
    argument = "yes",
    mode = "always",
  },
  { index_name="p", -- antes testfile
    description = "File with the corpus mfccs data",
    short = "p",
    argument = "yes",
    mode = "always",
  },
  { index_name="txt", -- antes txtfile
    description = "File with the corpus transcriptions",
    long = "txt",
    argument = "yes",
    mode = "always",
  },
  { index_name="phdict",
    description = "Dictionary for generate transcriptions (only if txtfile is not a phonetic transcription)",
    long = "phdict",
    argument = "yes",
  },
  {
    index_name="silences",
    description="HMM models for silences (a blank separated list)",
    long="silences",
    argument="yes",
    filter=string.tokenize,
  },
  {
    index_name="begin_sil",
    description="Add this model as Begin Silence (must be in --silences list)",
    long="begin-sil",
    argument="yes",
  },
  {
    index_name="end_sil",
    description="Add this model as Begin Silence (must be in --silences list)",
    long="end-sil",
    argument="yes",
  },
  {
    index_name="transcription_filter",
    description="Filter the transcriptions to generate phonetic sequences",
    long="transcription-filter",
    argument="yes",
    filter=dofile,
  },
  {
    index_name  = "context", -- antes ann_context_size
    description = "Size of ann context",
    long        = "context",
    argument    = "yes",
    filter = tonumber,
    mode = "always",
    default_value = 4,
  },
  {
    index_name  = "feats_format",
    description = "Format of features mat or mfc",
    long        = "feats-format",
    argument    = "yes",
    mode = "always",
    default_value = "mat",
  },
  {
    index_name  = "feats_norm",
    description = "Table with means and devs for features",
    long        = "feats-norm",
    argument    = "yes",
    filter = dofile,
  },
  {
    index_name  = "step",
    description = "Dataset step",
    long        = "step",
    argument    = "yes",
    mode = "always",
    filter = tonumber,
    default_value = "1",
  },
  { index_name="force",
    description = "Force overwritten output files",
    long     = "force",
    argument = "no",
    default_value = true,
  },
  {
    index_name = "dir",
    description = "Output dir",
    long = "dir",
    argument="yes",
    mode = "always",
  },
  {
    index_name = "phondir",
    description = "Output dir for phonetic transcription",
    long = "phondir",
    argument="yes",
    mode = "always",
    default_value = nil,
  },
  {
    index_name = "cores",
    description = "Number of cores",
    long = "cores",
    argument="yes",
    mode = "always",
    default_value = "2",
    filter = tonumber,
  },
  {
    description = "shows this help message",
    short = "h",
    long = "help",
    argument = "no",
    action = function (argument) 
	       print(cmdOptTest:generate_help()) 
	       os.exit(1)
	     end    
  }
}


local optargs = cmdOptTest:parse_without_check()
if type(optargs) == "string" then error(optargs) end
local initial_values
if optargs.defopt then
  initial_values = optargs.defopt
  optargs.defopt=nil
end
optargs = cmdOptTest:check_args(optargs, initial_values)

filenet     = optargs.n
filem       = optargs.m
filehmmdefs = optargs.hmmdefs
tiedfile    = optargs.t
valfile     = optargs.p
txtfile     = optargs.txt
dir         = optargs.dir
phondir     = optargs.phondir
context     = optargs.context
step        = optargs.step
format      = optargs.feats_format
transcription_filter = optargs.transcription_filter
phdict      = optargs.phdict
silences    = optargs.silences
begin_sil   = optargs.begin_sil
end_sil     = optargs.end_sil
cores       = optargs.cores
feats_mean_and_devs = optargs.feats_norm
force_write = optargs.force
if not silences then
  fprintf(io.stderr, "# WARNING!!! NOT SILENCES TABLE DEFINED\n")
end

if (not filenet or not filem) and not fileHTK then
  error ("Needs a HMM/ANN or HTK file")
end

if fileHTK then
  error ("Not revised!!! probably not implemented!!!")
end

ann_trainer = trainable.supervised_trainer.load( filenet, nil, 128)

--------------------
-- parametros RNA --
--------------------
ann = {}
ann.left_context           = context
ann.right_context          = context

--------------------
--     Corpus     --
--------------------
corpus = {
}
corpus.filename_val = valfile
corpus.filename_txt = txtfile

-- objeto con informacion sobre modelos ligados
tied = tied_model_manager(io.open(tiedfile))

---------
-- HMM --
---------

-- creamos el trainer y le metemos los modelos acusticos
hmm_trainer = HMMTrainer.trainer()

-- cargamos los HMMs
m = dofile(filem)

hmm = {}
hmm.silences = {}
hmm.silence_tbl = {}
for i=1,#silences do
  hmm.silences[silences[i]] = 1.0
  table.insert(hmm.silence_tbl, silences[i])
end
if begin_sil and (not silences or not hmm.silences[begin_sil]) then
  error ("Begin silence must be in --silences list")
end
if end_sil and (not silences or not hmm.silences[end_sil]) then
  error ("End silence must be in --silences list")
end

hmm.models       = m[1]
hmm.aprioris     = m[2]

num_emissions = ann_trainer:get_output_size()
emiss_to_hmm  = {}
-- anyadimos al trainer los modelos
num_models = 0
for name,model_info in pairs(hmm.models) do
  model_info.model.trainer = hmm_trainer
  for _,t in ipairs(model_info.model.transitions) do
    emiss_to_hmm[t.emission] = name
  end
  hmm_trainer:add_to_dict(model_info.model, name)
  num_models = num_models + 1
end
ann.output_dictionary = dataset.identity(num_emissions, 0.0, 1.0)
collectgarbage("collect")

-- si nos dan el diccionario fonetico, generamos modelos para las
-- palabras
local mangling -- sirve para el name mangling de las palabras
if phdict then
  print("# WARNING!!! Verify that your dictionary contains optional silences "..
	  "as alternative pronunciations")
  print("# Using dictionary for generate phonetic transcriptions")
  _,mangling=HMMTrainer.utils.dictionary2lextree(io.open(phdict),
						 tied,
						 hmm.silences,
						 hmm_trainer)
  collectgarbage("collect")
end

--------------------
-- Carga de datos --
--------------------
--
-- nombres de ficheros
test = field_manager()
recog.generate_filenames{
  corpus_data_manager    = test,
  filename_mfc           = valfile,
  filename_fon           = txtfile,
}
--

local frames_loader
if format == "mat" then
  frames_loader = load_matrix
else
  frames_loader = load_mfcc
end

collectgarbage("collect")

list = test:get_field('mfcc_filename')
which_i_am,child_pid = util.split_process(cores)

for index=which_i_am,#list,cores do
  mfcc_filename = list[index]
  collectgarbage("collect")
  -- cargamos el dataset correspondiente a la frase actual
  local tr_filename = test:get_field('phon_filename')[index]
  local basenamestr = remove_extensions(string.basename(tr_filename))
  print ("# Loading transcription: ", tr_filename, index.."/"..#list)
  local tr_string   = io.open(tr_filename):read("*l")
  print ("# Loading frames:        ", mfcc_filename)
  local frames = frames_loader(mfcc_filename)
  local numFrames   = frames:dim()[1]
  local numParams   = frames:dim()[2] -- nCCs+1
  local parameters = {
    patternSize = {step, numParams},
    offset      = {0,0},  -- default value
    stepSize    = {step, 0}, -- default value, second value is not important
    numSteps    = {numFrames/step, 1}
  }
  local actual_ds = dataset.matrix(frames, parameters)
  if feats_mean_and_devs then
    actual_ds:normalize_mean_deviation(feats_mean_and_devs.means,
				       feats_mean_and_devs.devs)
  end
  if filehmmdefs==nil then
    actual_ds = dataset.contextualizer(actual_ds,
				       ann.left_context,
				       ann.right_context)
  end
  
  local segmentation_matrix = matrix(numFrames)
  local mat_full = matrix(numFrames, num_emissions)
  local mat_full_ds = dataset.matrix(mat_full)
  ann_trainer:use_dataset{
    input_dataset  = actual_ds,  -- parametrizacion
    output_dataset = mat_full_ds -- matriz de emisiones
  }

  print("# Building HMM model")
  -- anyadimos los silencios
  if begin_sil then
    tr_string = begin_sil .. " " .. tr_string
  end
  if end_sil then 
    tr_string = tr_string .. " " .. end_sil
  end
  -- filtramos la transcripcion
  if transcription_filter then
    tr_string = transcription_filter(tr_string)
  end
  tr_string = string.gsub(tr_string, begin_sil.." "..begin_sil, begin_sil)
  tr_string = string.gsub(tr_string, end_sil.." "..end_sil, end_sil)
  local themodel
  if phdict then
    -- generamos el modelo a partir del diccionario fonetico, necesita
    -- name mangling
    local tbl = string.tokenize(tr_string)
    for i=1,#tbl do
      if not mangling[tbl[i]] and not hmm.silences[tbl[i]] then
	error("Word without transcription: "..tbl[i])
      end
      tbl[i]=mangling[tbl[i]] or tbl[i]
    end
    themodel=generate_models_from_sequences(tbl,
					    hmm_trainer,nil,hmm.silences)
  else
    -- generamos la secuencia de esta frase
    themodel=generate_models_from_sequences(string.tokenize(tr_string),
					    hmm_trainer,tied,hmm.silences)
  end
  
  print("# Segmentation")
  -- ahora generamos la salida de Viterbi
  local logprob,phon_output = themodel:viterbi{
    input_emission       = mat_full,
    do_expectation       = false,
    output_emission_seq  = segmentation_matrix,
    emission_in_log_base = true,
  }
  print(logprob, phon_output)
  
  local output_filename = dir .. "/" .. remove_extensions(string.basename(mfcc_filename)) .. ".mat"
  print ("# Saving " .. output_filename)
  if io.open(output_filename) and not force_write then
    error(string.format("# Output file '%s' exists, use --force to force overwritten\n",
			output_filename))
  else
    matrix.savefile(segmentation_matrix, output_filename, "binary")
  end
  if phondir then
    output_filename = phondir .. "/" .. remove_extensions(string.basename(mfcc_filename)) .. ".phon"
    print ("# Saving " .. output_filename)
    if io.open(output_filename) and not force_write then
      error(string.format("# Output file '%s' exists, use --force to force overwritten\n",
			  output_filename))
    else
      local f = io.open(output_filename, "w")
      f:write(phon_output .. "\n")
      f:close()
    end
  end
end

if child_pid then
  -- esperamos a los hijos
  util.wait()
end
