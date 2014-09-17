-- calcula la perplejidad de un conjunto de test

opt = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "Ngram PPL computed with April toolkit",
  { index_name="multi_class",
    description = "Compute a multi class with given number of classes",
    long    = "multi-class",
    argument = "no",
  },
  { index_name="vocabfilename",
    description = "Vocabulary (plain or HTK dictionary)",
    short    = "v",
    argument = "yes",
    mode = "always",
  },
  {
    index_name="langmodel_filename",
    description = "Language model (.lua for NNLMs in Lua script, .lira.gz for lira, "..
                  ".DIR for NNLMs in a DIR structure)",
    short = "l",
    argument = "yes",
    mode = "always",
  },
  {
    index_name="nnlm_conf",
    description = "Configuration of NNLMs training",
    short = "f",
    argument = "yes",
  },
  {
    index_name="word2probs",
    description="Word2probs table, for compute correctly the output of NNLMs",
    short="w",
    argument="yes",
  },
  {
    index_name="test",
    description = "Input text",
    short = "t",
    argument = "yes",
    mode = "always",
  },
  { index_name="debug",
    description = "Debug level (0, 1, 2)",
    short = "d",
    argument = "yes",
    mode = "always",
    default_value = 0,
    filter = tonumber,
  },
  {
    index_name="stream",
    description="The input text is a words stream",
    long="stream",
    argument = "no",
  },
  {
    index_name  = "trie_size",
    description = "Size of TrieVector for NNLMs",
    long        = "trie-size",
    argument    = "yes",
    mode = "always",
    default_value = 24,
    filter = tonumber,
  },
  {
    index_name  = "use_unk",
    description = "Indicate if use or not unk words in PPL computation (all, context, none)",
    long        = "unk",
    argument    = "yes",
    mode = "always",
    default_value = "all",
  },
  {
    index_name  = "train_restriction",
    description = "Indicate the use of <train> for indicate which sentences must be used for compute PPL",
    long        = "train-restriction",
    argument = "no",
  },
  {
    index_name  = "cache_data",
    description = "File with cache-data (use only with .DIR)",
    long        = "cache-data",
    argument    = "yes",
  },
  {
    index_name="cache_stop_token",
    description="Cache stop token",
    long="cache-stop-token",
    argument="yes",
    mode = "always",
    default_value = "<stop>",
  },
  {
    index_name="null_token",
    description="Null token",
    long="null-token",
    argument="yes",
    mode = "always",
    default_value = "<NULL>",
  },
  {
    index_name  = "order",
    description = "Use this Ngram order instead of model's order",
    long        = "order",
    argument    = "yes",
    filter = tonumber,
  },
  {
    index_name  = "no_sos",
    description = "Avoid start-of-sentence",
    long        = "no-sos",
    argument = "no",
  },
  {
    index_name  = "no_eos",
    description = "Avoid end-of-sentence",
    long        = "no-eos",
    argument = "no",
  },
  {
    index_name = "max_softmax_constants",
    description = "Max number of softmax constants",
    long="max-softmax-constants",
    argument="yes",
    mode = "always",
    default_value = 0,
    filter = tonumber,
  },
  {
    index_name = "print_stats",
    description = "Print stats",
    long="print-stats",
    argument = "no",
  },
  {
    description = "shows this help message",
    short = "h",
    long = "help",
    argument = "no",
    action = function (argument) 
      print(opt:generate_help()) 
      os.exit(1)
    end    
  }
}

---------------------------------------------------

local optargs = opt:parse_args(nil,"nnlm_conf",false)
table.unpack_on(optargs, _G)

if word2probs then
  word2probs = ngram.load_word2prob_smooth_factor(word2probs)
end

local vocab = lexClass.load(io.open(vocabfilename))
local model = language_models.load(langmodel_filename,
                                   vocab, "<s>", "</s>",
                                   {
                                     cache_size = 10, -- podria ser 0 ;)
                                     trie_size  = trie_size,
                                     word2probs = word2probs,
                                     cache_data = cache_data,
                                     max_trie_constants = max_softmax_constants
                                   })

--[[
if multi_class then
  local N = #vocab:getWordVocabulary()
  multi_class_table = {}
  for i=1,N do
    table.insert(multi_class_table, i)
  end
  r = random()
  multi_class_table = r:shuffle(multi_class_table)
  num_classes       = math.ceil(math.sqrt(N))
end
]]--

local cronometro = util.stopwatch()
cronometro:reset()
cronometro:go()

local resul = language_models.test_set_ppl{ lm = model,
                                            vocab = vocab,
                                            testset = test,
                                            log_file = io.stdout,
                                            debug_flag = debug,
                                            use_unk = use_unk,
                                            use_cache = false,
                                            train_restriction = train_restriction,
                                            cache_stop_token = cache_stop_token,
                                            null_token = null_token,
                                            use_bcc = not no_sos,
                                            use_ecc = not no_eos, }

fprintf(io.stderr,
        "Time: %f\nTime/token: %f\nTime/sentence: %f\n",
        cronometro:read(),
        cronometro:read()/(resul.numwords + resul.numsentences),
        cronometro:read()/resul.numsentences)

if print_stats then
  model:printStats()
end
