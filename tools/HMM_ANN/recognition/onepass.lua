--------------------------------------------------
-- command line options
--------------------------------------------------
opt = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "HMM/ANN recognition with April toolkit new one pass decoder",
  {
    index_name="defopt",
    description="Load configuration file (a lua table)",
    short="f",
    argument="yes",
  },
  ----------------------------------------------------------------------
  {
    index_name  = "recog",
    description = "Recog type: asr, htr [default=asr]",
    long        = "recog",
    argument    = "yes"
  },
  ----------------------------------------------------------------------
  -- relacionados con la red neuronal
  { index_name="n",
    description = "MLP file",
    short    = "n",
    argument = "yes",
  },
  { index_name="bunch_size",
    description="Size of bunch for MLPs [default 96]",
    long="bunch-size",
    argument="yes",
  },
  {
    index_name  = "context",
    description = "Size of ann context [default=8]",
    long        = "context",
    argument    = "yes"
  },
  {
    index_name  = "feats_norm",
    description = "Table with means and devs for features",
    long        = "feats-norm",
    argument    = "yes",
  },
  ----------------------------------------------------------------------
  -- con la carga de los modelos
  {
    index_name="m",
    description = "hmms file",
    short = "m",
    argument = "yes",
  },
  { index_name="t",
    description = "HTK unit's tied list",
    short = "t",
    argument = "yes",
  },
  { index_name="d",
    description = "HTK dictionary",
    short = "d",
    argument = "yes",
  },
  { index_name="pt",
    description = "pronunciation tree model",
    long = "pt",
    argument = "yes",
  },
  { index_name="bpt",
    description = "pronunciation tree model is binary [default true]",
    long = "bpt",
    argument = "yes",
  },
  { index_name="lm",
    description = "Language model [default=dont use LM]",
    long = "lm",
    argument = "yes",
  },
  {
    index_name  = "use_context_cues",
    description = "Use context cues",
    long        = "use-context-cues",
    argument    = "yes"
  },
  ----------------------------------------------------------------------
  -- con los parametros de configuracion
  { index_name="beam_width",
    description = "Beam width [default=400]",
    long = "beam-width",
    argument = "yes",
  },
  { index_name="histogram_size",
    description = "Histogram pruning size [default=5000]",
    long = "histogram-size",
    argument = "yes",
  },
  { index_name="gsf",
    description = "Grammar Scale Factor [default=10]",
    long = "gsf",
    argument = "yes",
  },
  { index_name="pt_gsf",
    description = "Prefix Tree Grammar Scale Factor [default=1]",
    long = "pt-gsf",
    argument = "yes",
  },
  { index_name="wip",
    description = "Word Insertion Penalty [default=0]",
    long = "wip",
    argument = "yes",
  },
  ----------------------------------------------------------------------
  --
  { index_name="p",
    description = "File with the corpus mfccs data",
    short = "p",
    argument = "yes",
  },
  { index_name="prep",
    description = "File with the corpus preprocessed data",
    long = "prep",
    argument = "yes",
  },
  { index_name="txt",
    description = "File with the corpus transcriptions",
    long = "txt",
    argument = "yes",
  },
  {
    index_name  = "filter",
    description = "Fiter output sentence before tasas, is a file that return a LUA function [default=nil]",
    long        = "filter",
    argument    = "yes",
  },
  {
    index_name  = "view_filter",
    description = "Fiter output sentence to show result, is a file that return a LUA function [default=nil]",
    long        = "view-filter",
    argument    = "yes",
  },
  {
    index_name  = "feats_format",
    description = "Format of features (mat, raw, mfc) [default mat]",
    long        = "feats-format",
    argument    = "yes",
  },
  {
    index_name  = "wadvance",
    description = "Window advance in ASR, in milliseconds [default=10] used to compute the real time factor",
    long        = "wadvance",
    argument    = "yes",
  },
  {
    index_name  = "cache_size",
    description = "Size of cache for NNLMs [default=20]",
    long        = "cache-size",
    xargument    = "yes",
  },
  {
    index_name  = "trie_size",
    description = "Size of TrieVector for NNLMs [default=24]",
    long        = "trie-size",
    argument    = "yes",
  },
  { index_name="ose_file",
    description = "Over SEgmenter index file, with one string for each corpus file [optional]",
    long = "ose-file",
    argument = "yes",
  },
  {
    description = "shows this help message",
    short       = "h",
    long        = "help",
    argument    = "no",
    action = function (argument) 
	       print(opt:generate_help()) 
	       os.exit(1)
	     end    
  }
}

optargs = opt:parse_args()
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

dir = string.gsub(arg[0], string.basename(arg[0]), "")
dir = (dir~="" and dir) or "./"
loadfile(dir .. "utils.lua")() -- deprecar

--------------------------------------------------
-- algunas funciones

-- function load_matrix(filename)
--   local f = io.open(filename) or error ("No se ha podido encontrar "..filename)
--   local aux = f:read("*a")
--   local ad  = matrix.fromString(aux)
--   f:close()
--   return ad
-- end

guarda_frames = ASR.VectorFrameConsumer()
vad = ASR.VAD{
  -- beginVoiceDuration =  5,
  -- beginVoiceScore    =  0.3,
  -- stopVoiceDuration  = 10,
  -- stopVoiceScore     =  0.1,
  frameConsumer      = guarda_frames,
}
parametrizer  = ASR.Parametrizer{
  HzSamplingAD        = 8000,
  preemphasis         =    0.95,
  msWindowSize        =   25,
  msWindowAdvance     =   10,
  nparam              =   12,
  decorrelation       = "FF1",
  output              = vad
}

function load_raw(filename)
  guarda_frames:reset()
  parametrizer:applyFile(filename)
  return guarda_frames:getMatrix()
end

--------------------------------------------------
-- log
fprintf(io.stderr,"# HOST:\t %s\n", (io.popen("hostname", "r"):read("*l")))
fprintf(io.stderr,"# DATE:\t %s\n", (io.popen("date", "r"):read("*l")))
fprintf(io.stderr,"# CMD: \t %s %s\n",string.basename(arg[0]), table.concat(arg, " "))

--------------------------------------------------
-- Red neuronal
filename_ann        = optargs.n
-- Fichero de HMMs
filename_hmm        = optargs.m
-- Fichero HTK
filename_hmmdefs    = optargs.hmmdefs
--
filename_tiedlist   = optargs.t  or error ("Needs a tiedlist!!!")
filename_dict       = optargs.d  or error ("Needs a dictionary!!!")
filename_prontree   = optargs.pt or error ("Needs a pronunciation tree!!!")
binary_prefix_tree  = optargs.bpt or true

filename_test       = optargs.p
filename_prep       = optargs.prep
filename_txt        = optargs.txt or fprintf(io.stderr,"Warning!!! running without transcriptions --txt!!!\n")
filename_lm         = optargs.lm
dataset_step        = tonumber(optargs.step or 1)
feats_format        = optargs.feats_format or "mat"
feats_mean_and_devs = optargs.feats_norm
features_filename   = optargs.features
recog_type          = string.lower(optargs.recog or "asr")
wsize               = (optargs.wsize    or 25) / 1000 -- para pasar a segundos
wadvance            = (optargs.wadvance or 10) / 1000 -- para pasar a segundos
beam_width          = tonumber(optargs.beam_width or  "400")
hist_prun_size      = tonumber(optargs.histogram_size  or "5000")
hist_prun_step      = 1000
max_hyps            = 1000000
trll_log2_hash_size = 21
trll_log2_max_nodes = 21
cache_size          = tonumber(optargs.cache_size)
trie_size           = tonumber(optargs.trie_size)
gsf                 = optargs.gsf    or 10.0
pt_gsf              = optargs.pt_gsf or  1.0
wip                 = optargs.wip    or  0.0
ose_file            = optargs.ose_file
if recog_type == 'htr' then
  word_sil            = "@"
else
  word_sil            = "."
end
initial_ngram_word  = '<s>'
final_ngram_word    = '</s>'
unk_ngram_word      = '<unk>'
mfcs_matrix_format  = (feats_format == "mat")
use_context_cues    = ((optargs.use_context_cues or "yes") == "yes")

-----------------------------------------------

if feats_mean_and_devs then
  feats_mean_and_devs = dofile(feats_mean_and_devs)
end

-----------------------------------------------

if optargs.filter then
  filter = dofile(optargs.filter)
else
  filter = function (str) return str end
end

-----------------------------------------------

if optargs.view_filter then
  view_filter = dofile(optargs.view_filter)
else
  view_filter = function (str) return str end
end

-----------------------------------------------
-- tied list
-----------------------------------------------
tied_model = tied_model_manager(io.open(filename_tiedlist))

-----------------------------------------------
-- dictionary
-----------------------------------------------
dictionary = lexClass.load(io.open(filename_dict))

-----------------------------------------------
-- HMMs
-----------------------------------------------
local m      = dofile(filename_hmm)
hmm          = {}
hmm.models   = m[1]
hmm.aprioris = m[2]
hmm.units    = parser.lr_hmm_dictionary()
hmm.unit2id  = {}

for id,phone in ipairs(tied_model.id2name) do
  local model_info = hmm.models[phone]
  local lprobs = {}
  for _,t in ipairs(model_info.model.transitions) do
    if t.from == t.to then
      lprobs[t.from-3] = math.exp(t.lprob)
    end
  end
  hmm.unit2id[phone] = hmm.units:insert_unit {
    emiss_indexes = model_info.emissions,
    loop_probs    = lprobs,
  }
end
-- convertimos el diccionario a formato lrhmm
hmm.lrhmm = parser.new_one_step.LR_HMMs(hmm.units)

-----------------------------------------------
-- language model
-----------------------------------------------
if filename_lm then
  ngram_model = ngram.load_language_model(filename_lm,
					  dictionary,
					  initial_ngram_word,
					  final_ngram_word,
					  -- EXTRA
					  {
					    cache_size = cache_size,
					    trie_size  = trie_size,
					  })
else 
  ngram_model = ngram.lira.loop{
    vocabulary_size = #dictionary:getWordVocabulary(),
    final_word      = dictionary:getWordId("</s>"),
  }
end

-----------------------------------------------
-- pronunciation tree
-----------------------------------------------

pronunciation_tree = parser.new_one_step.treeModel{
  filename    = filename_prontree,
  vocabulary  = dictionary:getWordVocabulary(),
  is_mmaped   = binary_prefix_tree,
  desired_gsf = pt_gsf,
}

----------------------------------------------------------------------
-- construmos el lmtrellis:

if getmetatable(ngram_model).id == ngram.nnlm.meta_instance.id then
  lmtrellis = parser.new_one_step.NNLMPBTrellis{
    lm                     = ngram_model,
    trellis_log2_hash_size = trll_log2_hash_size,
    trellis_log2_max_nodes = trll_log2_max_nodes,
    initial_word           = dictionary:getWordId(initial_ngram_word),
    final_word             = dictionary:getWordId(final_ngram_word),
    gsf                    = gsf,
    wip                    = wip,
  }
else
  lmtrellis = parser.new_one_step.LiraPBTrellis{
    lm                     = ngram_model,
    trellis_log2_hash_size = trll_log2_hash_size,
    trellis_log2_max_nodes = trll_log2_max_nodes,
    initial_word           = dictionary:getWordId(initial_ngram_word),
    final_word             = dictionary:getWordId(final_ngram_word),
    gsf                    = gsf,
    wip                    = wip,
    use_context_cues       = use_context_cues,
  }
end

----------------------------------------------------------------------
-- creamos un heappruner:
pruner = parser.new_one_step.HeapPruner(hist_prun_size, beam_width)

----------------------------------------------------------------------
-- construimos el lexdecoder:
lexdecoder = parser.new_one_step.LexDecoder{
  tree_model     = pronunciation_tree,
  max_hyps       = max_hyps,
  hist_prun_size = hist_prun_size,
  hist_prun_step = hist_prun_step,
  lr_hmms        = hmm.lrhmm,
  pruner         = pruner,
  lm_trellis     = lmtrellis,
  silence_index  = tied_model.name2id[word_sil]-1, -- FIXME hay que restar 1 :( si no se lo resto no va
}

----------------------------------------------------------------------
-- crear el objeto decoder, que es un consumidor:
decoder = parser.new_one_step.decoder{
  lexdecoder = lexdecoder,
  a_prioris  = hmm.aprioris
}

----------------------------------------------------------------------
-- cargamos el MLP
ann               = {}
ann.left_context  = tonumber(optargs.context or 8)
ann.right_context = tonumber(optargs.context or 8)
ann.bunch_size    = tonumber(optargs.bunch_size or 96)
ann.bunched_net   = mlp.load(filename_ann, ann.bunch_size)
ann.emiss_func    = ann.bunched_net:get_function()

----------------------------------------------------------------------
-- los datos de prueba

test = field_manager()
recog.generate_filenames{
  corpus_data_manager = test,
  filename_mfc        = filename_test,
  prepfilename        = filename_prep,
  txtfilename         = filename_txt,
}
--

local frames_source
local frames_normalization
local frames_contextualizer

AUXF = io.open("aux.log", "w")

local total_num_words = 0
local lista_tasas     = {}
local count_gc        = 0

-- numero total de tramas procesadas, sirve para medir al final el
-- tiempo CPU por trama:
local  total_processed_frames = 0
-- total de frases correctas, para calcular el SER
local  correct_sentences      = 0

cronometro = util.stopwatch()

local last_recog_lm_wids = nil
local index = 0
local mfcc_filename

if ose_file then ose_file = io.open(ose_file, "r") end
num_sentences = #test:get_field('mfcc_filename')
for index = 1,num_sentences do
  count_gc = count_gc+1
  if count_gc >= 10 then
    collectgarbage("collect")
    count_gc = 0
  end

  mfcc_filename = test:get_field('mfcc_filename')[index]

  -- cargamos el dataset correspondiente a la frase actual
  local tr_filename
  if filename_txt then
    tr_filename = test:get_field('transcription_filename')[index]
  end
  local prep_filename
  if filename_prep then
    prep_filename = test:get_field('prep_filename')[index]
  end
  local basenamestr = ""
  if mfcc_filename then
    basenamestr = remove_extensions(string.basename(mfcc_filename))
  end
  local tr_string
  if tr_filename then
    fprintf(io.stderr,"# Cargando transcripcion: \t%s\n", tr_filename)
    tr_string = io.open(tr_filename):read("*l")
  end
  local frames
  local actual_ds
  local numFrames = 1
  local numParams = 1

  fprintf(io.stderr,"# Cargando frames:        \t%s\n", mfcc_filename)
  if feats_format == 'mat' then
    frames = load_matrix(mfcc_filename)
  elseif feats_format == 'raw' then
    frames = load_raw(mfcc_filename)
  end
  -- if mfcs_matrix_format then
  --   frames = load_matrix(mfcc_filename)
  -- else
  --   frames = load_mfcc(mfcc_filename)
  -- end
  numFrames = frames:dim()[1]
  numParams = frames:dim()[2]
  local parameters = {
    patternSize = {dataset_step, numParams},
    offset      = {0,0},  -- default value
    stepSize    = {dataset_step, 0}, -- default value, second value is not important
    numSteps    = {numFrames/dataset_step, 1}
  }
  actual_ds = dataset.matrix(frames, parameters)
  if feats_mean_and_devs then
    actual_ds:normalize_mean_deviation(feats_mean_and_devs.means,
				       feats_mean_and_devs.devs)
  end
  actual_ds = dataset.contextualizer(actual_ds,
				     ann.left_context,
				     ann.right_context)
  
  -- cargamos la transcripcion ortografica
  local correcta = tr_string
  
  -- RUN ONE STEP PARSER
  duration = wsize + numFrames*wadvance
  fprintf(io.stderr,"# RUN %4d/%d (%d frames %s secs)\n",
	  index,num_sentences,numFrames,duration)

  t1cpu, t1wall = cronometro:read()
  cronometro:go()
  
  if ose_file then
    -- carga la sobresegmentacion de fichero
    local current_ose_filename = ose_file:read("*l")
    local current_ose = io.open(current_ose_filename)
    local frontiers = current_ose:read("*a")
    current_ose:close()
    -- print("frontiers has lenght",#frontiers)
    decoder:set_frontiers_string(frontiers)
    fprintf(io.stderr, "# Loading frontier %s\n", current_ose_filename)
  end

  ann.emiss_func:calculate_in_pipeline{
    producer    = functions.dataset_producer(actual_ds),
    consumer    = decoder,
    input_size  = ann.emiss_func:get_input_size(),
    output_size = ann.emiss_func:get_output_size(),
  }

  path,logprob=decoder:get_result()

  cronometro:stop()
  t2cpu, t2wall = cronometro:read()

  if path then
    reconocida = table.concat(dictionary:searchWordsSequenceFromWIDs(path)," ")
    print(reconocida)
    fprintf (io.stderr,"###########################################\n")
    fprintf (io.stderr,"# logprob N-grama:  %f\n",logprob)
    fprintf(io.stderr,"# Correcta:            %s\n",(correcta or "N/A"))
    fprintf(io.stderr,"# Reconocida N-grama:  %s\n",view_filter(reconocida))
    if correcta and filter(reconocida or "") == filter(correcta) then
      correct_sentences = correct_sentences + 1
    end
    total_num_words = total_num_words + #string.tokenize(reconocida)
    tWall = t2wall - t1wall
    tCPU  = t2cpu  - t1cpu
    if tWall > 0 then
      framesPerSecond = numFrames/tWall
    else
      framesPerSecond = 0
    end
    if recog_type == "asr" then
      RTfactor = tWall/duration
      fprintf(io.stderr,"# Tiempo: %.2f (cpu) %.2f (wall) %.2f frames/s (%.2f x RT)\n",
	     tCPU, tWall, framesPerSecond, RTfactor)
    else
      fprintf(io.stderr,"# Tiempo: %.2f (cpu) %.2f (wall) %.2f frames/s\n",
	     tCPU, tWall, framesPerSecond)
    end
    fprintf(io.stderr,"###########################################\n")
    total_processed_frames = total_processed_frames + numFrames + 1
    
    ------------------------------------------------------------
    -- para tasas
    if correcta then
      table.insert(lista_tasas,{filter(correcta), filter(reconocida)})
      AUXF:write(filter(correcta).."="..filter(reconocida).."\n")
    end
    ------------------------------------------------
  else
    fprintf(io.stderr,"# Correcta:            %s\n",(correcta or "N/A"))
    fprintf(io.stderr,"# Reconocida N-grama:   \n")
    if correcta then
      table.insert(lista_tasas,{filter(correcta), ""})
      if filter(reconocida or "") == filter(correcta) then
	correct_sentences = correct_sentences + 1
      end
    end
    fprintf(io.stderr,"# Error, no se ha encontrado camino!!\n")
    print("")

  end

  fprintf(io.stderr,"\n")
  io.stdout:flush()
  io.stderr:flush()
  --------------------------------------------------------
end


AUXF:close()
if filename_txt then
  local resul_val = tasas{
    typedata = "pairs_lines",
    -- words_sep valor por defecto
    data = lista_tasas,
    tasa = "ie", -- para calcular wer
  }
  fprintf(io.stderr,"Tasa WER:  %f\n", resul_val.tasa)
  
  local resul_val = tasas{
    typedata = "pairs_lines",
    data = lista_tasas,
    tasa = "ie", -- para calcular wer
    words_width = 1,
  }
  fprintf(io.stderr,"Tasa CER:  %f\n", resul_val.tasa)
  fprintf(io.stderr,"Tasa SER:  %f\n", 100*(1 - correct_sentences/num_sentences))
end

total_cpu_time, total_wall_time = cronometro:read()
fprintf(io.stderr,"Total time: %.2f (cpu) %.2f (wall), cpu_time/sentence %.4f cpu_time/word %.4f\n",
       total_cpu_time, total_wall_time, total_cpu_time/num_sentences,
       total_cpu_time/total_num_words)
fprintf(io.stderr,"wall_time/sentence %.4f wall_time/word %.4f",
     total_wall_time/num_sentences, total_wall_time/total_num_words)
if recog_type == "asr" then
  RTfactor = total_wall_time/(total_processed_frames*wadvance)
  fprintf(io.stderr,", RT factor: %.2f x RT",RTfactor)
end
fprintf(io.stderr,"\n")

