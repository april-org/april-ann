dir = string.gsub(arg[0], string.basename(arg[0]), "")
dir = (dir~="" and dir) or "./"
loadfile(dir .. "utils.lua")()

fprintf(io.stderr,"# HOST:\t %s\tload avg: %s\n",
	(io.popen("hostname", "r"):read("*l")),
	(io.open("/proc/loadavg"):read("*l")))
fprintf(io.stderr,"# DATE:\t %s\n", (io.popen("date", "r"):read("*l")))
fprintf(io.stderr,"# CMD: \t %s %s\n",string.basename(arg[0]), table.concat(arg, " "))

--------------------------------------------------
cmdOptTest = dofile(dir .. "/opt.lua")
optargs    = cmdOptTest:parse_args()
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

if optargs.bunch_size then
  error("Bunch mode not implemented!!!, we are working in it ;)")
end

log10 = math.log(10)

--------------------------------------------------
--optargs.fast_nn_lm = true

server_port         = optargs.server
one_step            = ((optargs.one_step or "no") == "yes")
-- Red neuronal
filenet             = optargs.n
-- Fichero de HMMs
filem               = optargs.m
-- Fichero HTK
filehmmdefs         = optargs.hmmdefs
-- Fichero OSE
ose_file            = optargs.ose_file
--
dictfile            = optargs.d  or error ("Needs a dictionary!!!")
use_optional_sil    = (optargs.optsil == "yes")
use_word_sil        = (optargs.wordsil == "yes")
nested_openmp       = (optargs.nested == "yes")
lattice_tool        = optargs.lattice_tool
output_lm_words     = ((optargs.output_lm_words or "no") == "yes")

save_seg            = (optargs.seg or "no") == "yes"
testfile            = optargs.p
input_lat            = optargs.input_lattice
use_input_lat_lm_score = ((optargs.use_input_lat_lm_score or "no") == "yes")
input_lat_log_base  = optargs.input_lat_log_base
if (input_lat_log_base or use_input_lat_lm_score) and not input_lat then
  error ("Needs --input-lat")
end
if not input_lat and not server_port and not testfile then
  error ("Needs a corpus testfile -p option or a --text-input option!!!")
end
tiedfile            = optargs.t  or (input_lat==nil and error ("Needs a tiedlist!!!"))
prepfile            = optargs.prep
txtfile             = optargs.txt or fprintf(io.stderr,"Warning!!! running without transcriptions --txt!!!\n")
ose_type            = optargs.ose
recog_type          = optargs.recog
langmodel           = optargs.lm
if use_input_lat_lm_score and langmodel then
  error("Forbidden to use a LM when use-input-lat-lm-socre='yes'")
end
inputwg             = optargs.save_input_wg
dataset_step        = tonumber(optargs.step or 1)
max_h_out           = tonumber(optargs.max_h_out or 0)
cache_size          = tonumber(optargs.cache_size)
trie_size           = tonumber(optargs.trie_size)
feats_format        = optargs.feats_format or "mfc"
feats_mean_and_devs = optargs.feats_norm
wg_filename         = optargs.save_nbest_wg
nbest_filename      = optargs.save_nbest
num_nbest           = tonumber(optargs.nbest or 0)
features_filename   = optargs.features
uniq_nbest          = optargs.uniq_nbest or "yes"
wadvance            = (optargs.wadvance or 10) / 1000 -- para pasar a segundos
unk_threshold       = tonumber(optargs.unk or "NONE")
cache_txt           = optargs.cache_txt
use_ref_annotations = ((optargs.use_ref_annotations or "no") == "yes")
right_left          = (string.lower(optargs.right_left or "no") == "yes")
cells_per_frame     = tonumber(optargs.cells_per_frame or 20)
use_bcc             = ((optargs.use_bcc or "yes") == "yes")
use_ecc             = ((optargs.use_ecc or "yes") == "yes")
if cache_txt then cache_txt = io.open(cache_txt, "r") or error ("File not found: " .. cache_txt) end
if server_port then server_port = tonumber(server_port) end
if use_ref_annotations and not txtfile then
  error ("use_ref_annotations needs a txtfile")
end

viterbi_m_size      = optargs.viterbim_size

recog_type = string.lower(recog_type)

if num_nbest > 0 and max_h_out == 0 then
  error("max_h_out must be > 0 when nbest > 0!!!")
end

if right_left and wg_filename then
  error ("Forbidden Output WordGraph and RightLeft feature")
end

if features_filename then
  features_file = io.open(features_filename, "w") or error("Cant create file: " .. features_filename)
end
if nbest_filename then
  nbest_file = io.open(nbest_filename, "w") or error("Cant create file: " .. nbest_filename)
end

read_mfcs_in_matrix_format = (feats_format == "mat")

if feats_mean_and_devs then
  feats_mean_and_devs = dofile(feats_mean_and_devs)
end

if server_port then
  prepfile = nil
  txtfile  = nil
  testfile = nil
end

----------------------------------------------------------------------------------

use_word_dict_probs   = ((optargs.use_word_probs or "yes") == "yes")
apply_gsf_word_probs  = ((optargs.apply_gsf_word_probs or "no") == "yes")

ann = {}
ann.left_context  = tonumber(optargs.context or 4)
ann.right_context = tonumber(optargs.context or 4)

-- Parsing
grammar_scale_factor     = optargs.gsf or 10.0
word_insertion_penalty   = optargs.wip or 0.0
nolm                     = (optargs.lm == nil)
viterbi_type             = "viterbi_m" -- viterbi_m" --"hash_fast_it"
-- viterbi_m, array_swap, active_envelope, hash_array_swap, hash_fast_it
if not one_step then
  if not input_lat then
    wgen_hist_pruning_conf = {
      beam_width = optargs.wgen_beam or 1000,
      size       = optargs.wgen_size or 200,
      max_states = optargs.wgen_nstates or 4000,
    }
    wgen_feedback             = true
  end
else
  if optargs.wgen_beam or optargs.wgen_size or optargs.wgen_nstates then
    fprintf(io.stderr, "# Warning!!! WGEN pruning parameters are ignored when one_step=\"yes\"\n")
  end
end
hist_pruning_conf = {
  beam_width = optargs.ngram_beam or 2000,
  size       = optargs.ngram_size or 400,
  max_states = optargs.ngram_nstates or 10000,
}

-- parametros de NGRAMAS
initial_ngram_word        = '<s>'
final_ngram_word          = '</s>'
unk_ngram_word            = '<unk>'

-- SSE
if ose_type and ose_file then
  error ("ose-file and ose are incompatible")
end
if not ose_type then
  if recog_type == "asr" then ose_type = "dummy"
  elseif recog_type == "htr" then ose_type = "htr"
  elseif recog_type == "online" then ose_type = "online"
  else error("Incorrect recognizer type: " .. recog_type)
  end
end
if ose_type == "dummy" then
  segmenter = nil
  sse_type  = "dummy"
elseif ose_type == "htr" then
  segmenter = DFS_CC_Oversegmenter()
  sse_type  = "from_string" -- dummy, from_string, max_diff
elseif ose_type == "online" then
  segmenter = nil
  sse_type  = "from_string"
elseif ose_type == "maxdiff" then
  segmenter = nil
  sse_type  = "maxdiff"
else
  error("Incorrect Oversegmenter type: " .. ose_type)
end
if ose_file then sse_type = "from_string" end

if optargs.dog then segmenter=nil sse_type = "dummy" end

sse_past           = 5
sse_future         = 5
sse_dist_max_front = 3
sse_distance       = "inf-norm" -- 1-norm, 2-norm, inf-norm

if optargs.filter then
  filter = dofile(optargs.filter)
else
  filter = function (str) return str end
end

if optargs.view_filter then
  view_filter = dofile(optargs.view_filter)
else
  view_filter = function (str) return str end
end

-- objeto con informacion sobre modelos ligados
if tiedfile then
  tied = tied_model_manager(io.open(tiedfile) or error("File not found: " .. tiedfile))
end
-- objeto con el diccionario
fprintf(io.stderr,"# Loading dictionary\n")
dictionary = lexClass.load(io.open(dictfile) or error ("File not found: " .. dictfile))

pairid2wordstr = {}
for i=1,dictionary:pairsTblSize() do
  table.insert(pairid2wordstr, tostring(i))
end

----------
-- HMMs --
----------
hmm = {}
unk_pair_ids = {}
if filem and filenet then
  -- HIBRIDOS
  m = dofile(filem)
  hmm.models       = m[1]
  hmm.aprioris     = m[2]
  ---------------------------------------
  -- cargamos la red neuronal
  lared          = Mlp.load{ filename=filenet }
  emissions_func = lared
  ---------------------------------------
  -- creamos las unidades LR
  hmm.tree         = {}
  hmm.tree.unit2id = {}
  hmm.tree.units   = parser.lr_hmm_dictionary()
  local id = 1
  for name,model_info in pairs(hmm.models) do
    id = id + 1
    local lprobs = {}
    for _,t in ipairs(model_info.model.transitions) do
      if t.from == t.to then
	lprobs[t.from-3] = math.exp(t.lprob)
      end
    end
    if right_left then
      hmm.tree.unit2id[name] = hmm.tree.units:insert_unit {
	emiss_indexes = table.reverse(model_info.emissions),
	loop_probs    = table.reverse(lprobs),
      }
    else
      hmm.tree.unit2id[name] = hmm.tree.units:insert_unit {
	emiss_indexes = model_info.emissions,
	loop_probs    = lprobs,
      }
    end
  end
  -- CREAMOS EL OBJETO LEXICO EN ARBOL
  hmm.tree.lexicon = parser.tree_lexicon()
  -- ahora metemos el diccionario

  -- para cada par (word,outsym)
  for pairid=1,dictionary:pairsTblSize() do
    local wid,oid = dictionary:getPair(pairid)
    local data    = dictionary:getData(pairid)
    local us      = data.units -- la transcripcion
    if data.word == unk_ngram_word then
      table.insert(unk_pair_ids, pairid)
    end
    if #us > 0 then
      -- si units es mayor que 0, entonces lo metemos en el lexico
      local t  = {}      
      -- para cada unidad de la transcripcion
      for j,u in ipairs(tied:search_triphone_sequence(us)) do
	table.insert(t, hmm.tree.unit2id[u])
	if not hmm.tree.unit2id[u] then
	  error("ERROR incorrect hmm model in tiedlist: [ "..u.." ]")
	end
      end
      -- sacamos el outsym index
      local prob = data.prob or 0.0
      if apply_gsf_word_probs then prob = prob * grammar_scale_factor end
      prob = math.exp(prob)
      if not use_word_dict_probs then prob = 1.0 end
      -- insertamos en el tree_lexicon
      if right_left then
	t = table.reverse(t)
      end
      if prob ~= 0 then hmm.tree.lexicon:insert_word(t, pairid, prob) end
      -- else print (wid, paird, prob,
      -- dictionary.oid2outsym[dictionary.pairs[pairid].oid]) end
    end
  end
elseif filehmmdefs then
  -- HTK
  if right_left then
    error('right_left is not currently supported for HTK models')
  end
  hmm.tree = {}
  hmm.models = htk_interface.load_hmm_set(filehmmdefs)
  hmm.tree.units,htk_triphone_table = htk_interface.build_lr_hmm_dictionary(tied,
									    hmm.models)
  hmm.tree.lexicon = htk_interface.build_tree_lexicon(dictionary, tied, htk_triphone_table,
						      forbiden_words)
  emissions_func = functions.join(table.invert(hmm.models.all_mixtures))
  hmm.tree.unit2id = htk_triphone_table
elseif not input_lat then
  error ("arg's -m and -n, or --hmmdefs, are needed!!!")
end

-- Silencios
gsub_sils = ""
if not input_lat then
  if recog_type == "htr" then
    hmm.silences = {
      ["@"] = {
	prob = 1.0,
	output = 0 -- lambda
      }
    }
    begin_sil = "@"
    end_sil   = "@"
    word_sil  = "@"
    gsub_sils = "%@"
  elseif recog_type == "online" then
    hmm.silences = {
    }
    begin_sil = ""
    end_sil   = ""
    word_sil  = ""
    gsub_sils = ""
  else
    hmm.silences = {
      ["."] = {
	prob = 1.0,
	output = 0 -- lambda
      },
      ["("] = {
	prob = 1.0,
	output = 0 -- lambda
      },
      [")"] = {
	prob = 1.0,
	output = 0 -- lambda
      }
    }
    begin_sil = "("
    end_sil   = ")"
    word_sil  = "."
    gsub_sils = "%.%(%)"
  end
end

--------------------------------------
-- Cargamos el modelo de ngramas... --
--------------------------------------
fprintf(io.stderr,"# Cargamos el modelo de ngramas y creamos el parser\n")

if langmodel then
  -- estadistico
  ngram_model = ngram.load_language_model(langmodel,
					  dictionary,
					  initial_ngram_word,
					  final_ngram_word,
					  -- EXTRA
					  {
					    cache_size = cache_size,
					    trie_size  = trie_size,
					  })
else
  -- NO LM
  fprintf(io.stderr, "# Loop LM\n")
  ngram_model = ngram.lira.loop{
    vocabulary_size = #dictionary:getWordVocabulary(),
    final_word      = dictionary:getWordId(final_ngram_word),
  }
end
--ngram_model:sanity_check()

-----------------------
-- CREAMOS EL PARSER --
-----------------------
if not one_step then
  model_parser = ngram.load_parser(ngram_model,
				   dictionary,
				   initial_ngram_word,
				   final_ngram_word,
				   max_h_out)
  if not use_bcc then
    model_parser:set_use_begin_context_cue(false)
  end
  if not use_ecc then
    model_parser:set_use_end_context_cue(false)
  end
end




----------------------------------------------------------------
-- GENERAMOS EL MODELO DE MARKOV  HMM para el lexico en arbol --
----------------------------------------------------------------

if not input_lat then
  if not one_step then
    hmm.tree.lexicon_model   = hmm.tree.lexicon:tree_lr_hmm_model(hmm.tree.units)
    if viterbi_m_size then
      hmm.tree.lexicon_model:set_size_active_state_vector(viterbi_m_size);
    end
    hmm.tree.lexicon_factory = parser.factory[viterbi_type](hmm.tree.lexicon_model)
  else
    hmm.tree.lexicon_model = parser.one_step.treeParserModel(hmm.tree.units, hmm.tree.lexicon)
  end
  
  --hmm.tree.units:print()
  --hmm.tree.lexicon:print()
  
  -- silencios
  if use_word_sil then
    if not one_step then
      hmm.tree.word_sil = parser.tree_lexicon()
      hmm.tree.word_sil:insert_word({hmm.tree.unit2id[tied:search_triphone(nil,
									   word_sil,
									   nil)]
				   },
				    hmm.silences[word_sil].output,
				    hmm.silences[word_sil].prob)
      hmm.tree.word_sil_model = hmm.tree.word_sil:tree_lr_hmm_model(hmm.tree.units)
      hmm.tree.word_sil_factory = parser.factory[viterbi_type](hmm.tree.word_sil_model)
    end
  end
  
  if use_optional_sil then
    hmm.tree.begin_sil = parser.tree_lexicon()
    hmm.tree.end_sil = parser.tree_lexicon()
    hmm.tree.begin_sil:insert_word({hmm.tree.unit2id[tied:search_triphone(nil,
									  begin_sil,
									  nil)]
				  },
				   hmm.silences[begin_sil].output,
				   hmm.silences[begin_sil].prob)
    if not one_step then
      hmm.tree.begin_sil_model = hmm.tree.begin_sil:tree_lr_hmm_model(hmm.tree.units)
      hmm.tree.begin_sil_factory = parser.factory[viterbi_type](hmm.tree.begin_sil_model)
    else
      hmm.tree.begin_sil_model = parser.one_step.treeParserModel(hmm.tree.units, hmm.tree.begin_sil)
    end
    
    hmm.tree.end_sil:insert_word({hmm.tree.unit2id[tied:search_triphone(nil,
									end_sil,
									nil)]
				},
				 hmm.silences[end_sil].output,
				 hmm.silences[end_sil].prob)
    if not one_step then
      hmm.tree.end_sil_model = hmm.tree.end_sil:tree_lr_hmm_model(hmm.tree.units)
      hmm.tree.end_sil_factory = parser.factory[viterbi_type](hmm.tree.end_sil_model)
    else
      hmm.tree.end_sil_model = parser.one_step.treeParserModel(hmm.tree.units, hmm.tree.end_sil)
    end
  end
end

----------------------------------------------
-- CREAMOS UN PARSER_FACTORY_AUTOMATA
if not input_lat then
  if unk_threshold then
    if one_step then
      error ("Not implemented 'unk' option for one step algorithm!!!")
    else
      if #unk_pair_ids > 1 then
	error ("Forbidden more than one " .. unk_ngram_word .. " pair in dictionary!!!")
      end
      local filler_parser_factory = parser.simple_filler(unk_pair_ids[1],
							 hmm.tree.units)
      local lexicon_plus_oov_factory = parser.factory.lexicon_plus_oov{
	lexicon_parser_factory = hmm.tree.lexicon_factory,
	filler_parser_factory  = filler_parser_factory,
      }
      hmm.tree.lexicon_factory = lexicon_plus_oov_factory
    end
  end
  
  if not one_step then
    if use_optional_sil then
      transitions = {
	{from="ini",to="med",prob=1,arc=hmm.tree.begin_sil_factory},
	{from="ini",to="med",prob=1,arc=hmm.tree.lexicon_factory},
	{from="ini",to="fin",prob=1,arc=hmm.tree.lexicon_factory},
	{from="med",to="med",prob=1,arc=hmm.tree.lexicon_factory},
	{from="med",to="fin",prob=1,arc=hmm.tree.end_sil_factory},
	{from="med",to="fin",prob=1,arc=hmm.tree.lexicon_factory},
      }
    else
      transitions = {
	{from="ini",to="med",prob=1,arc=hmm.tree.lexicon_factory},
	{from="ini",to="fin",prob=1,arc=hmm.tree.lexicon_factory},
	{from="med",to="med",prob=1,arc=hmm.tree.lexicon_factory},
	{from="med",to="fin",prob=1,arc=hmm.tree.lexicon_factory},
      }
    end
    
    if use_word_sil then
      table.insert(transitions,
		   {from="med", to="sil", prob=1, arc=hmm.tree.word_sil_factory})
      table.insert(transitions,
		   {from="sil", to="med", prob=1, arc=hmm.tree.lexicon_factory})
    end
    
    hmm.tree.factory_automata = parser.factory.automata{
      initials = {
	{"ini", 1}
      },
      finals   = {
	{"fin", 1},
      },
      transitions=transitions
    }
  end
end

--print(hmm.tree.lexicon_factory)
--print(hmm.tree.factory_automata)
--hmm.tree.factory_automata:print()
--
-------------------------------------

--
--------------
-- DATAFLOW --
--------------

test = field_manager()
if not server_port then
  -- nombres de ficheros
  recog.generate_filenames{
    corpus_data_manager    = test,
    filename_mfc           = testfile,
    input_lat_filename      = input_lat,
    prepfilename           = prepfile,
    txtfilename            = txtfile,
  }
end
--

local frames_source
local frames_normalization
local frames_contextualizer
if not one_step then
  -- dataflow
  the_dataflow = dataflow()
  --
  if inputwg then
    --  iohandler = dataflow.process.iohandler{
    --    dataflow     = the_dataflow,
    --    service_name = "iohandler"
    --  }
    --
    --  df_serializer = dataflow.process.serializer{
    --    dataflow = the_dataflow,
    --    io_service_name = "iohandler",
    --    filename=inputwg,
    --  }
  end
  --
  
  if not server_port then
    if not input_lat then
      -- leemos de dataset
      frames_source = dataflow.process.dataset_source{
	dataflow  = the_dataflow,
      }
    end
  else
    iohandler = dataflow.process.iohandler{
      dataflow     = the_dataflow,
      service_name = "iohandler"
    }
    -- leer de socket y aplicar medias y desviaciones
    frames_source = dataflow.process.deserializer{
      dataflow        = the_dataflow,
      io_service_name = "iohandler",
      host            = "0.0.0.0",
      port            = server_port,
    }
    if feats_mean_and_devs then
      frames_normalization = dataflow.process.mean_and_devs_normalization{
	dataflow = the_dataflow,
	means    = feats_mean_and_devs.means,
	devs     = feats_mean_and_devs.devs,
      }
    end
    local dim = emissions_func:get_input_size() / (ann.left_context + ann.right_context + 1)
    frames_contextualizer = dataflow.process.contextualizer{
      dataflow  = the_dataflow,
      left      = ann.left_context,
      right     = ann.right_context,
      input_dim = dim,
    }
  end

  if not input_lat then
    -- gausianas
    df_emissions = dataflow.process.function_box{
      dataflow     = the_dataflow,
      the_function = emissions_func:get_function(),
    }
    if filehmmdefs==nil then
      logp = dataflow.process.df_logp {
	dataflow    = the_dataflow,
	prior_probs = hmm.aprioris,
	-- en este punto se podria poner un vector de indices para
	-- reordenar las emisiones que llegan a los modelos del
      -- seq_graph_gen
      }
    end
    
    -- sse
    if sse_type == "maxdiff"  then
      sse = dataflow.process.sse_maxdiff_simple{
	dataflow       = the_dataflow,
	past           = sse_past,
	future         = sse_future,
	dist_max_front = sse_dist_max_front,
	distance       = sse_distance,
      }
    elseif sse_type == "dummy" then
      sse = dataflow.process.sse_dummy{
	dataflow = the_dataflow
      }
    elseif sse_type == "from_string" then
      sse = dataflow.process.sse_from_string{
	dataflow = the_dataflow
      }
    end
    
    -- creamos el grafo de palabras
    wgen = dataflow.process.df_seq_graph_gen{
      dataflow       = the_dataflow,
      automata       = hmm.tree.factory_automata,
    }
    wgen:configure_histogram_pruning(wgen_hist_pruning_conf)
  else
    -- en este caso WGEN es una fuente de grafos
    wgen = dataflow.process.word_graph_out{
      dataflow              = the_dataflow,
      force_topologic_order = true,
    }
  end
  --
  -- ngram
  a_parser = dataflow.process.parser{
    dataflow  = the_dataflow,
    parser    = model_parser,
    lex_table = dictionary:getCObject(),
}
  --
  
  -- sse_spy = dataflow.process.filter_spy{
  --   dataflow=the_dataflow,
  --   path="sse2.log",
  --   name="sse_spy"
  -- }
  
  -- sink
  a_sink = dataflow.process.dummy_sink{
    dataflow = the_dataflow,
  }
  
  if inputwg then
    -- WORD GRAPH IN --
    -- creamos una caja para el word_graph_in
    wgraph_in = dataflow.process.word_graph_in(the_dataflow)
    -------------------
  end
  
  ---------------------------------------------
  if not input_lat then
    if not frames_normalization then
      if not frames_contextualizer then
	dataflow.connect ({frames_source,        "output"}, {df_emissions,    "input"})
      else
	dataflow.connect ({frames_source,        "output"}, {frames_contextualizer,    "input"})
	dataflow.connect ({frames_contextualizer,"output"}, {df_emissions,    "input"})
      end
    else
      dataflow.connect ({frames_source,        "output"}, {frames_normalization, "input"})
      if not frames_contextualizer then
	dataflow.connect ({frames_normalization, "output"}, {df_emissions,    "input"})
      else
	dataflow.connect ({frames_normalization, "output"}, {frames_contextualizer,    "input"})
	dataflow.connect ({frames_contextualizer, "output"}, {df_emissions,    "input"})
      end
    end
    if filehmmdefs==nil then
      dataflow.connect ({df_emissions,    "output"}, {logp, "input"})
      dataflow.connect ({logp,    "output"},         {sse, "input"})
    else
      dataflow.connect ({df_emissions,    "output"}, {sse, "input"})
    end
    dataflow.connect ({sse,       "output"},
		      
		      -- 		  {sse_spy, "input"})
		      -- dataflow.connect ({sse_spy, "output"},
		      
		      {wgen, "input1"})
  end
  if inputwg then
    dataflow.connect ({wgen,   "output1"},		  {wgraph_in, "input"})
    dataflow.connect ({wgraph_in, "output"}, {a_parser, "input"})
  else
    if input_lat then
      dataflow.connect ({wgen,   "output"},    {a_parser, "input"})
    else
      dataflow.connect ({wgen,   "output1"},    {a_parser, "input"})
    end
  end
  
  -- BESTPROB FEEDBACK --
  if wgen_feedback and not input_lat then
    dataflow.connect ({a_parser,      "output2"}, {wgen,   "input2"})
  end
  -----------------------
  
  -- NGRAM PARSER --
  dataflow.connect ({a_parser,    "output1"},   {a_sink, "input"})
  ------------------
else
  model_parser = ngram.one_step.load_parser(ngram_model,
					    dictionary,
					    initial_ngram_word,
					    final_ngram_word,
					    hmm.tree.begin_sil_model,
					    hmm.tree.end_sil_model,
					    hmm.tree.lexicon_model)
end

-- configuramos el parser
model_parser:configure_histogram_pruning(hist_pruning_conf)
model_parser:set_grammar_scale_factor(grammar_scale_factor)
model_parser:set_word_insertion_penalty(word_insertion_penalty)

--AUXF = io.open("aux.log", "w")

local total_num_words = 0
local lista_tasas  = {}
local count_gc     = 0

-- numero total de tramas procesadas, sirve para medir al final el
-- tiempo CPU por trama:
local  total_processed_frames = 0
-- total de frases correctas, para calcular el SER
local  correct_sentences = 0

cronometro = util.stopwatch()

local last_recog_lm_wids = nil
local last_correcta = nil
local index = 0
local mfcc_filename
local input_lat_filename
if ose_file then
  local name = ose_file
  ose_file   = io.open(ose_file, "r") or error("Not found file: " .. name)
end
while true do
  index = index + 1
  if not server_port then
    if not testfile then
      if index > #test:get_field('input_lat_filename') then break end
      input_lat_filename = test:get_field('input_lat_filename')[index]
    else
      if index > #test:get_field('mfcc_filename') then break end
      mfcc_filename = test:get_field('mfcc_filename')[index]
    end
  end
  count_gc = count_gc+1
  if count_gc >= 10 or server_port then
    collectgarbage("collect")
    count_gc = 0
  end
  -- cargamos el dataset correspondiente a la frase actual
  local tr_filename
  if txtfile then
    tr_filename = test:get_field('transcription_filename')[index]
  end
  local prep_filename
  if prepfile then
    prep_filename = test:get_field('prep_filename')[index]
  end
  local basenamestr = ""
  if mfcc_filename then
    basenamestr = remove_extensions(string.basename(mfcc_filename))
  end
  local tr_string
  if tr_filename then
    fprintf(io.stderr,"# Cargando transcripcion: \t%s\n", tr_filename)
    local ftr = io.open(tr_filename) or error ("File not found: " .. tr_filename)
    tr_string = ftr:read("*l")
    ftr:close()
  end
  local frames
  local actual_ds
  local numFrames = 1
  local numParams = 1
  if not server_port then
    if not mfcc_filename then
      fprintf(io.stderr,"# Word Graph input:       \t%s\n", input_lat_filename)
      local wg = word_graph.from_htk_lattice(input_lat_filename,
					     dictionary,
					     use_input_lat_lm_score,
					     "!NULL",
					     input_lat_log_base)
      wgen:set_word_graph(wg)
    else
      fprintf(io.stderr,"# Cargando frames:        \t%s\n", mfcc_filename)
      if read_mfcs_in_matrix_format then
	frames = load_matrix(mfcc_filename)
      else
	frames = load_mfcc(mfcc_filename)
      end
      numFrames = frames:dim()[1]
      numParams = frames:dim()[2] -- nCCs+1
      local parameters
      if right_left then
	parameters = {
	  patternSize = {dataset_step, numParams},
	  offset      = {numFrames/dataset_step-1,0},  -- default value
	  stepSize    = {-dataset_step, 0}, -- default value, second value is not important
	  numSteps    = {numFrames/dataset_step, 1}
	}
      else
	parameters = {
	  patternSize = {dataset_step, numParams},
	  offset      = {0,0},  -- default value
	  stepSize    = {dataset_step, 0}, -- default value, second value is not important
	  numSteps    = {numFrames/dataset_step, 1}
	}
      end
      actual_ds = dataset.matrix(frames, parameters)
      if feats_mean_and_devs then
	actual_ds:normalize_mean_deviation(feats_mean_and_devs.means,
					   feats_mean_and_devs.devs)
      end
      if filehmmdefs==nil then
	actual_ds = dataset.contextualizer(actual_ds,
					   ann.left_context,
					   ann.right_context,
					   right_left)
      end
    end
  else
    fprintf(io.stderr,"# Esperando frames en:    \t%d\n", server_port)
  end
  
  -- cargamos la transcripcion ortografica
  local correcta = tr_string
  
  if one_step then
    if server_port then error("ONE STEP forbidden with server port!!!") end
    t1cpu, t1wall = cronometro:read()
    cronometro:go()
    
    local frontiers = nil
    if sse_type == "from_string" and not input_lat_filename then
      if ose_file then
	-- carga la sobresegmentacion de fichero
	local current_ose = io.open(ose_file:read("*l")) or error ("File not found!!")
	frontiers = current_ose:read("*a")
	current_ose:close()
      elseif recog_type == "htr" then
	if not prep_filename then error ("Needs a preprocessed files list --prep options!!!") end
	fprintf(io.stderr,"# Cargando datos prep:    \t%s\n", prep_filename)
	-- generamos la segmentacion a partir de las imagenes preprocesadas
	local img = ImageIO.read(prep_filename):to_grayscale()
	local _,img_height = img:geometry()
	local max_intraword_space = 999999 -- 50 -- frames
	local fatFrontiers = false
	frontiers = segmenter:oversegment(img, cells_per_frame,
					  max_intraword_space, fatFrontiers)
      elseif recog_type == "online" then
	-- genera las fronteras para el online. TODO: pasar a C++ este codigo
	frontiers = { util.ose.compose_frontier{ util.ose.generate_output } }
	for ipat,pat in dataset.matrix(frames):patterns() do
	  if pat[#pat] > 0 or ipat == numFrames then
	    table.insert(frontiers, util.ose.compose_frontier{ util.ose.generate_output })
	  else
	    table.insert(frontiers, util.ose.compose_frontier{ })
	  end
	end
	frontiers = table.concat(frontiers, "")
      end
      if #frontiers ~= numFrames+1 then
	error(string.format("Generated frontiers: %d, number of frames: %d",
			    #frontiers, numFrames))
      end
      --print(string.byte(frontiers,1,#frontiers))
      --print(#frontiers, frontiers)
      if right_left then
	frontiers = util.ose.reverse(frontiers)
      end
    end

    -- RUN ONE STEP PARSER
    fprintf(io.stderr,"# RUN\n")
    
    model_parser:reset()
    model_parser:do_viterbi{
      input_dataset  = actual_ds,
      emissions_func = emissions_func:get_function(),
      a_prioris      = hmm.aprioris,
      sse_string     = frontiers,
      nested_openmp  = nested_openmp,
    }
    
    cronometro:stop()
    t2cpu, t2wall = cronometro:read()
    
  else
    -- RUN AL DATAFLOW
    if inputwg then
      cpp_lex_wg = WordGraph()
      lex_wg     = word_graph(cpp_lex_wg, dictionary)
      wgraph_in:set_word_graph(cpp_lex_wg)
    end
    if not server_port then
      if mfcc_filename then
	frames_source:set_dataset(actual_ds)
      end
    end
    if sse_type == "from_string" and not input_lat_filename then
      local frontiers = nil
      if server_port then
	error("FROM STRING ose is forbidden with server port!!!")
      end
      if ose_file then
	-- carga la sobresegmentacion de fichero
	local current_ose = io.open(ose_file:read("*l")) or error("File not found!!!")
	frontiers = current_ose:read("*a")
	current_ose:close()
      elseif recog_type == "htr" then
	if not prep_filename then error ("Needs a preprocessed files list --prep options!!!") end
	fprintf(io.stderr,"# Cargando datos prep:    \t%s\n", prep_filename)
	-- generamos la segmentacion a partir de las imagenes preprocesadas
	local img = ImageIO.read(prep_filename):to_grayscale()
	local _,img_height = img:geometry()
	local max_intraword_space = 999999 -- 50 -- frames
	local fatFrontiers = false
	frontiers = segmenter:oversegment(img, cells_per_frame,
					  max_intraword_space, fatFrontiers)
      elseif recog_type == "online" then
	-- genera las fronteras para el online. TODO: pasar a C++ este codigo
	frontiers = { util.ose.compose_frontier{ util.ose.generate_output } }
	for ipat,pat in dataset.matrix(frames):patterns() do
	  if pat[#pat] > 0 or ipat == numFrames then
	    table.insert(frontiers, util.ose.compose_frontier{ util.ose.generate_output })
	  else
	    table.insert(frontiers, util.ose.compose_frontier{ })
	  end
	end
	frontiers = table.concat(frontiers, "")
      end
      if #frontiers ~= numFrames+1 then
        error(string.format("Generated frontiers: %d, number of frames: %d",
			    #frontiers, numFrames))
      end
      --print(string.byte(frontiers,1,#frontiers))
      --print(#frontiers, frontiers)
      if right_left then
	frontiers = util.ose.reverse(frontiers)
      end
      sse:set_frontiers_string(frontiers)
    end
    fprintf(io.stderr,"# RUN\n")
    t1cpu, t1wall = cronometro:read()
    if ngram_model:has_cache() then
      ngram_model:restart()
      local wids_tbl = last_recog_lm_wids
      if use_ref_annotations and last_correcta then
	local correcta_tbl = string.tokenize(last_correcta)
	local correcta_wids = dictionary:searchWordIdSequence(correcta_tbl)
	wids_tbl = correcta_wids
	fprintf(io.stderr,"# Processed last %d reference words\n", #correcta_wids)
      elseif last_recog_lm_wids then
	fprintf(io.stderr,"# Processed last %d recog words\n", #last_recog_lm_wids)
      end
      if wids_tbl then
	for i=1,#wids_tbl do
	  ngram_model:cacheWord(wids_tbl[i])
	end
      end
      if cache_txt then
	local cache_txt_line = cache_txt:read("*l")
	local cache_words    = string.tokenize(cache_txt_line)
	for i=1,#cache_words do
	  if cache_words[i] == "<stop>" then
	    ngram_model:clearCache()
	  elseif cache_words[i] ~= "<NULL>" then
	    local w = dictionary:getWordId(cache_words[i]) or dictionary:getWordId(unk_ngram_word)
	    ngram_model:cacheWord(w)
	  end
	end
	fprintf(io.stderr,"# Processed %d extra cache words\n", #cache_words)
      end
      --ngram_model:showCache()
    end
    cronometro:go()
    resul=the_dataflow:run()
    cronometro:stop()
    t2cpu, t2wall = cronometro:read()
  
    if inputwg then
      local dot_str = lex_wg:to_dot()
      local f = io.open(string.format("%s_%s_%05d.dot",
				      inputwg,
				      basenamestr,
				      index), "w")
      f:write(dot_str.."\n")
      f:close()
      cpp_lex_wg = nil
      lex_wg     = nil
    end
    
    if not resul then
      fprintf(io.stderr,"Error al hacer dataflow:run()\n")
    end
  end
  
  -- GET WORDGRAPH
  --  wgraph = wgraph_in:get_word_graph()
  --  local wgf = io.open("grafo.dot", "w")
  --  wgf:write(wgraph:to_dot(dictionary:getVocabulary()))
  --  wgf:close()
  --
  
  -- GET BESTPATH
  if save_seg then
    path,seg,logprob = model_parser:get_best_path_and_segmentation()
    fprintf(io.stderr,"%s\n",table.concat(path, " "))
    fprintf(io.stderr,"%s\n",table.concat(seg, " "))
  else
    path,logprob = model_parser:get_best_path()
  end
  --
  if not server_port then
    if mfcc_filename then
      fprintf(io.stderr,"# %s\t%s\n",test:get_field('mfcc_filename')[index],
	      index .. "/" .. #test:get_field('mfcc_filename'))
    elseif input_lat_filename then
      fprintf(io.stderr,"# %s\t%s\n",test:get_field('input_lat_filename')[index],
	      index .. "/" .. #test:get_field('input_lat_filename'))
    end
  end
  if path then
    if right_left then
      path = table.reverse(path)
    end
    -- print ("# DAG VERTEX:", table.concat(segmentation, ", "))
    fprintf (io.stderr,"###########################################\n")
    fprintf (io.stderr,"# logprob N-grama:  %f\n",logprob)
    local get_tr = function(path)
		     local reconocida
		     if optargs.dog then
		       reconocida = table.concat(dictionary:searchOutSymsSequence(path),
						 "")
		       reconocida = string.gsub(reconocida, "[".. gsub_sils .. "]", " ")
		       reconocida = table.concat(string.tokenize(reconocida), " ")
		     else
		       if output_lm_words then
			 reconocida = table.concat(dictionary:searchWordsSequence(path),
						   " ")
		       else
			 reconocida = table.concat(dictionary:searchOutSymsSequence(path),
						   " ")
		       end
		     end
		     return reconocida
		   end
    local get_lm_tr = function(path)
			local reconocida = table.concat(dictionary:searchWordsSequence(path),
							" ")
			if optargs.dog then
			  reconocida = table.concat(dictionary:searchWordsSequence(path),
						    "")
			  reconocida = string.gsub(reconocida, "[".. gsub_sils .. "]", " ")
			  reconocida = table.concat(string.tokenize(reconocida), " ")
			end
			return reconocida
		      end
    reconocida = get_tr(path)
    last_recog_lm_wids =  dictionary:searchWordIdSequenceFromPairID(path)
    fprintf(io.stderr,"# Correcta:            %s\n",(correcta or "N/A"))
    last_correcta = correcta
    fprintf(io.stderr,"# Reconocida N-grama:  %s\n",view_filter(reconocida))
    print(reconocida)
    if correcta and filter(reconocida or "") == filter(correcta) then
      correct_sentences = correct_sentences + 1
    end
    total_num_words = total_num_words + ((correcta and #string.tokenize(correcta)) or
				         #string.tokenize(reconocida))
    if save_seg then
      fprintf(io.stderr,"# Segmentacion:        %s\n",table.concat(seg, " "))
    end
    tWall = t2wall-t1wall
    tCPU =  t2cpu-t1cpu
    if tWall > 0 then
      framesPerSecond = numFrames/tWall
    else
      framesPerSecond = 0
    end
    if recog_type == "asr" then
      RTfactor = tWall/((numFrames+1)*wadvance) -- sumamos 1 pq windosize es aprox 2 wadvance
      fprintf(io.stderr,"# Tiempo: %.2f (cpu) %.2f (wall) %.2f frames/s (%.2f x RT) %d frames\n",
	     tCPU, tWall, framesPerSecond, RTfactor, numFrames or 0)
    else
      fprintf(io.stderr,"# Tiempo: %.2f (cpu) %.2f (wall) %.2f frames/s %d frames\n",
	     tCPU, tWall, framesPerSecond, numFrames or 0)
    end
    fprintf(io.stderr,"###########################################\n")
    -- OJO: he quitado el +1 para poder medir frames/sec, en lugar de
    -- RT factor. Al final, para calcular el RT factor se le suma
    -- num_sentences para compensar
    total_processed_frames = total_processed_frames + numFrames
    
    --
    -- para tasas
    --
    if correcta then
      table.insert(lista_tasas,{filter(correcta), filter(reconocida)})
      --AUXF:write(filter(correcta).."="..filter(reconocida).."\n")
    end
    --
    -------------------------------------------------
    -- NBEST
    local output_wg = nil
    local wg_file   = nil
    if wg_filename then
      output_wg = WordGraph()
    end
    if nbest_file then

      local sri_filename
      local sri_filename2
      local sri_file
      if lattice_tool then
	local wg     = model_parser:get_trellis_wordgraph()
	local str    = wg:toSLF(pairid2wordstr)
	sri_filename = os.tmpname()
	sri_file     = io.open(sri_filename, "w")
	sri_file:write(str)
	sri_file:close()
	sri_filename2 = os.tmpname()
	os.execute("rm -f ".. sri_filename2)
	sri_filename2 = sri_filename2 .. ".out"
	os.execute("mkdir ".. sri_filename2)
	os.execute(string.format("%s %s -read-htk -in-lattice %s -nbest-decode %s "..
				 "-htk-lmscale 0.000000001 -htk-acscale 1 -htk-wdpenalty 0.000000001 "..
				 "-out-nbest-dir %s -htk-logbase 2.7182818284590451", -- -nbest-viterbi",
			       lattice_tool,
			       (uniq_nbest=="yes" and "-nbest-duplicates 0") or "",
			       sri_filename,
			       num_nbest, sri_filename2))
	os.execute("rm -f " .. sri_filename)
	sri_filename = sri_filename2 .. "/dummy.gz"
	sri_file     = io.open(sri_filename, "r")
	sri_file:read("*l") -- leemos la 1 best
      else
	if output_wg then
	  model_parser:configure_eppstein(output_wg)
	else
	  model_parser:configure_eppstein()
	end
      end
      local reconocida = get_tr(path)

      local output    = {}
      write_features =
	function(path,logprob)
	  if features_file then
	    local wids = dictionary:searchWordIdSequenceFromPairID(path)
	    local lm_score =
	      ngram.get_prob_from_id_tbl(ngram_model,
					 wids,
					 dictionary:getWordId(initial_ngram_word),
					 dictionary:getWordId(final_ngram_word),
					 use_bcc,
					 use_ecc)
	    if apply_gsf_word_probs then
	      for i=1,#path do
		lm_score = lm_score + dictionary:getData(path[i]).prob
	      end
	    end
	    hmm_score = logprob - lm_score*grammar_scale_factor -
	      #wids * word_insertion_penalty
	    table.insert(output, string.format("%f %f %d",hmm_score,lm_score,#wids))
	  end
	end
      write_features(path,logprob)

      local uniq = {}
      fprintf(nbest_file,
	      "%d||| %s ||| %s ||| %f\n",
	      index-1,
	      reconocida,
	      output[#output] or "",
	      logprob)
      if uniq_nbest == "yes" then
	uniq[reconocida] = true
      end
      --epp = model_parser:get_eppstein(nbest_wg_file ~= nil)
      local nbesti = 1
      local num_it = 0
      while nbesti < num_nbest and num_it < num_nbest*20 do
	num_it = num_it + 1
	local path
	local logprob
	if lattice_tool then
	  local line = sri_file:read("*l") -- leemos la siguiente
	  if line then
	    local t = string.tokenize(line)
	    logprob = tonumber(t[1])*log10
	    path = {}
	    for i=5,#t-1 do
	      table.insert(path, tonumber(t[i]))
	    end
	  end
	else
	  path,seg,logprob = model_parser:next_best_path()
	  if right_left then
	    path = table.reverse(path)
	  end
	end
	if not path then break end
	local reconocida = get_tr(path)
	local do_print = true
	if uniq_nbest == "yes" then
	  if not uniq[reconocida] then
	    uniq[reconocida] = true
	  else do_print = false
	  end
	end
	if do_print then
	  write_features(path,logprob)
	  fprintf(nbest_file,
		  "%d||| %s ||| %s ||| %f\n",
		  index-1,
		  reconocida,
		  output[#output] or "",
		  logprob)
	  nbesti = nbesti + 1
	  nbest_file:flush()
	end
      end
      if lattice_tool then sri_file:close() end
      if features_file then
	features_file:write(string.format("FEATURES_TXT_BEGIN_0 %d %d 3 hmm lm wip\n",
					  index-1, nbesti))
	features_file:write(table.concat(output, "\n").."\n")
	features_file:write("FEATURES_TXT_END_0\n")
	features_file:flush()
      end
      if sri_filename then
	os.execute("rm -Rf " .. sri_filename2)
      end
    elseif wg_filename then
      model_parser:configure_eppstein(output_wg)
      model_parser:compute_nbest(num_nbest-1)
    end
    
    if output_wg then
      -- salvar word graph
      local wg_file = io.open(string.format("%s_%s_%05d.slf",
					    wg_filename,
					    basenamestr,
					    index), "w")
      local wg = word_graph(output_wg, dictionary)
      -- TODO: usar GSF y WIP
      --wg_file:write(wg:to_slf(nil,grammar_scale_factor,word_insertion_penalty).."\n")
      wg_file:write(wg:to_slf(nil,0,0).."\n")
      wg_file:close()
    end
    ------------------------------------------------
  else
    fprintf(io.stderr,"# Correcta:            %s\n",(correcta or "N/A"))
    last_correcta = correcta
    fprintf(io.stderr,"# Reconocida N-grama:   \n")
    if correcta then
      table.insert(lista_tasas,{filter(correcta), ""})
      if filter(reconocida or "") == filter(correcta) then
	correct_sentences = correct_sentences + 1
      end
    end
    fprintf(io.stderr,"# Error, no se ha encontrado camino!!\n")
    print("")
    last_recog_lm_wids = {}
    -- NBEST
    if nbest_file then
      if features_file then
	features_file:write("FEATURES_TXT_BEGIN_0 %d 1 3 hmm lm wip\n", index-1)
      end
      fprintf(nbest_file,
	      "%d|||     ||| %f %f %f ||| %f\n",
	      index-1, 0, 0, 0, 0)
      write_features =
	function()
	  if features_file then
	    fprintf(features_file,"%f %f %f\n",0,0,0)
	    features_file:flush()
	  end
	end
      write_features()
      if features_file then
	features_file:write("FEATURES_TXT_END_0\n")
      end
    end
    if nbest_wg_file then
    end
  end

  --printf("%d %d\n",
  --hmm.tree.lexicon_model:get_num_vectors(),
  --hmm.tree.lexicon_model:get_num_states())
  fprintf(io.stderr,"\n")
  io.stdout:flush()
  io.stderr:flush()
  --------------------------------------------------------
end
if nbest_file then nbest_file:close() end
if features_file then features_file:close() end
num_sentences = index

--AUXF:close()
if txtfile then
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
fprintf(io.stderr,"Total time: %.2f (cpu) %.2f (wall), cpu/sentence %.4f cpu/word %.4f ",
       total_cpu_time, total_wall_time, total_cpu_time/num_sentences,
       total_cpu_time/total_num_words)
fprintf(io.stderr,"wall/sentence %.4f wall/word %.4f ",
     total_wall_time/num_sentences, total_wall_time/total_num_words)
fprintf(io.stderr,"frames/cpu %.4f frames/wall %.4f",
	total_processed_frames/total_cpu_time, total_processed_frames/total_wall_time)
if recog_type == "asr" then
  -- sumamos num_sentences porque wsize es casi 2 wadvance
  -- (hay que sumar 1 por cada frase = num_sentences)
  RTfactor = total_wall_time/((total_processed_frames+num_sentences)*wadvance)
  fprintf(io.stderr,", RT factor: %.2f x RT",RTfactor)
end
fprintf(io.stderr,"\n")

