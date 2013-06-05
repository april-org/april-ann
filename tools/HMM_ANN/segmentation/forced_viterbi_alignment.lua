-- FIXME: Este SCRIPT solo tiene en cuenta trifonemas dentro de las
-- palabras, pero entre palabras

dir = string.gsub(arg[0], string.basename(arg[0]), "")
dir = (dir~="" and dir) or "./"
loadfile(dir .. "utils.lua")()

cmdOptTest = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "Force Viterbi Alignment with April toolkit",
  {
    index_name="defopt",
    description="Load configuration file (a lua tabla)",
    short="f",
    argument="yes",
  },
  { index_name="n", -- antes filenet
    description = "MLP file",
    short    = "n",
    argument = "yes",
  },
  {
    index_name="m", -- antes filem
    description = "hmms file",
    short = "m",
    argument = "yes",
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
  },
  { index_name="p", -- antes testfile
    description = "File with the corpus mfccs data",
    short = "p",
    argument = "yes",
  },
  { index_name="txt", -- antes txtfile
    description = "File with the corpus transcriptions",
    long = "txt",
    argument = "yes",
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
  },
  {
    index_name  = "context", -- antes ann_context_size
    description = "Size of ann context [default=4]",
    long        = "context",
    argument    = "yes"
  },
  {
    index_name  = "feats_format",
    description = "Format of features mat or mfc [default mat]",
    long        = "feats-format",
    argument    = "yes",
  },
  {
    index_name  = "feats_norm",
    description = "Table with means and devs for features [default nil]",
    long        = "feats-norm",
    argument    = "yes",
  },
  {
    index_name  = "step",
    description = "Dataset step [default 1]",
    long        = "step",
    argument    = "yes",
  },
  {
    index_name = "dir",
    description = "Output dir [default initial_segmentation/]",
    long = "dir",
    argument="yes",
  },
  {
    index_name = "phondir",
    description = "Output dir for phonetic transcription [default none]",
    long = "phondir",
    argument="yes",
  },
  {
    index_name = "cores",
    description = "Number of cores [default = 2]",
    long = "cores",
    argument="yes",
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

optargs = cmdOptTest:parse_args()

if type(optargs) == "string" then error(optargs) end

if optargs.defopt then
  local t = dofile(optargs.defopt)
  for name,value in pairs(t) do
    if not optargs[name] then
      fprintf(io.stderr,"# opt %s = %s\n", name, tostring(value))
      optargs[name] = value
    end
  end
end

filenet     = optargs.n -- Red neuronal
filem       = optargs.m -- Fichero de HMMs
filehmmdefs = optargs.hmmdefs
tiedfile    = optargs.t or error ("Needs a tiedlist") -- fichero con modelos ligados
valfile     = optargs.p or error ("Needs a list of MFCCs")
txtfile     = optargs.txt  or error ("Needs a list of TXTs")
dir         = optargs.dir or "initial_segmentation"
phondir     = optargs.phondir
context     = tonumber(optargs.context or 4)
step        = tonumber(optargs.step or 1)
format      = optargs.feats_format or "mat"
transcription_filter = optargs.transcription_filter
phdict      = optargs.phdict
silences    = string.tokenize(optargs.silences or "")
begin_sil   = optargs.begin_sil
end_sil     = optargs.end_sil
cores       = tonumber(optargs.cores or 2)
if not silences then
  fprintf(io.stderr, "# WARNING!!! NOT SILENCES TABLE DEFINED\n")
end
if transcription_filter then
  transcription_filter = dofile(transcription_filter)
end

feats_mean_and_devs = optargs.feats_norm
if feats_mean_and_devs then
  feats_mean_and_devs = dofile(feats_mean_and_devs)
end

if (not filenet or not filem) and not fileHTK then
  error ("Needs a HMM/ANN or HTK file")
end
if fileHTK then
  error ("Not revised!!! probably not implemented!!!")
end

lared = Mlp.load{ filename = filenet }
func  = lared

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
trainer = HMMTrainer.trainer()

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

num_emissions = func:get_output_size()
emiss_to_hmm  = {}
-- anyadimos al trainer los modelos
num_models = 0
for name,model_info in pairs(hmm.models) do
  model_info.model.trainer = trainer
  for _,t in ipairs(model_info.model.transitions) do
    emiss_to_hmm[t.emission] = name
  end
  trainer:add_to_dict(model_info.model, name)
  num_models = num_models + 1
end
ann.output_dictionary = dataset.identity(num_emissions, 0.0, 1.0)
collectgarbage("collect")

-- si nos dan el diccionario fonetico, generamos modelos para las
-- palabras
local mangling -- sirve para el name mangling de las palabras
if phdict then
  print("IMPORTANTE!!! Verifica que el diccionario tiene " ..
	"los silencios como pronunciaciones alternativas")
  print("# Using dictionary for generate phonetic transcriptions")
  _,mangling=HMMTrainer.utils.dictionary2lextree(io.open(phdict),
						 tied,
						 hmm.silences,
						 trainer)
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

collectgarbage("collect")

list = test:get_field('mfcc_filename')
which_i_am,child_pid = util.split_process(cores)

for index=which_i_am,#list,cores do
  mfcc_filename = list[index]
  collectgarbage("collect")
  -- cargamos el dataset correspondiente a la frase actual
  local tr_filename = test:get_field('phon_filename')[index]
  local basenamestr = remove_extensions(string.basename(tr_filename))
  print ("# Cargando transcripcion: ", tr_filename, index.."/"..#list)
  local tr_string   = io.open(tr_filename):read("*l")
  print ("# Cargando frames:        ", mfcc_filename)
  local frames
  if format == "mat" then
    frames = load_matrix(mfcc_filename)
  else
    frames = load_mfcc(mfcc_filename)
  end
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
  func:use_dataset{
    input_dataset  = actual_ds,   -- parametrizacion
    output_dataset = mat_full_ds        -- matriz de emisiones
  }

  print("# Generando modelo HMM")
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
					    trainer,nil,hmm.silences)
  else
    -- generamos la secuencia de esta frase
    themodel=generate_models_from_sequences(string.tokenize(tr_string),
					    trainer,tied,hmm.silences)
  end
  
  print("# Segmentando")
  -- ahora generamos la salida de Viterbi
  local logprob,phon_output = themodel:viterbi{
    input_emission       = mat_full,
    do_expectation       = false,
    output_emission_seq  = segmentation_matrix,
    -- output_emission      = mat_full,              -- para DEBUG
    emission_in_log_base = false,
  }
  print(logprob, phon_output)
  
  local output_filename = dir .. "/" .. remove_extensions(string.basename(mfcc_filename)) .. ".mat"
  print ("# Salvando " .. output_filename)
  matrix.savefile(segmentation_matrix, output_filename, "binary")
  
  if phondir then
    output_filename = phondir .. "/" .. remove_extensions(string.basename(mfcc_filename)) .. ".phon"
    print ("# Salvando " .. output_filename)
    local f = io.open(output_filename, "w")
    f:write(phon_output .. "\n")
    f:close()
  end
end

if child_pid then
  -- esperamos a los hijos
  util.wait()
end
