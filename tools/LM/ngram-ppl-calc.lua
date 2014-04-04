-- calcula la perplejidad de un conjunto de test

opt = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "Ngram PPL computed with April toolkit",
  { index_name="multi_class",
    description = "Compute a multi class with given number of classes (by default no)",
    long    = "multi-class",
    argument = "yes",
  },
  { index_name="vocabfilename",
    description = "Vocabulary (plain or HTK dictionary)",
    short    = "v",
    argument = "yes",
  },
  {
    index_name="langmodel_filename",
    description = "Language model (.lua for NNLMs in Lua script, .lira.gz for lira, "..
                  ".DIR for NNLMs in a DIR structure)",
    short = "l",
    argument = "yes",
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
  },
  { index_name="debug",
    description = "Debug level (0, 1, 2)",
    short = "d",
    argument = "yes",
  },
  {
    index_name="stream",
    description="The input text is a words stream [default=no]",
    long="stream",
    argument="yes",
  },
  {
    index_name  = "trie_size",
    description = "Size of TrieVector for NNLMs [default=24]",
    long        = "trie-size",
    argument    = "yes",
  },
  {
    index_name  = "unk",
    description = "Indicate if use or not unk words in PPL computation (all, context, none) [default=all]",
    long        = "unk",
    argument    = "yes",
  },
  {
    index_name  = "train_restriction",
    description = "Indicate the use of <train> for indicate which sentences must be used for compute PPL [default=no]",
    long        = "train-restriction",
    argument    = "yes",
  },
  {
    index_name  = "cache_data",
    description = "File with cache-data (use only with .DIR) [default=nil]",
    long        = "cache-data",
    argument    = "yes",
  },
  {
    index_name="cache_stop_token",
    description="Cache stop token (default <stop>)",
    long="cache-stop-token",
    argument="yes",
  },
  {
    index_name="null_token",
    description="Null token (default <NULL>)",
    long="null-token",
    argument="yes",
  },
  {
    index_name  = "order",
    description = "Use this Ngram order instead of model's order [default=nil]",
    long        = "order",
    argument    = "yes",
  },
  {
    index_name  = "use_bcc",
    description = "Use begin context cue [default=yes]",
    long        = "use-bcc",
    argument    = "yes",
  },
  {
    index_name  = "use_ecc",
    description = "Use begin context cue [default=yes]",
    long        = "use-ecc",
    argument    = "yes",
  },
  {
    index_name = "max_softmax_constants",
    description = "Max number of softmax constants [default=0]",
    long="max-softmax-constants",
    argument="yes",
  },
  {
    index_name = "print_stats",
    description = "Print stats [default='no']",
    long="print-stats",
    argument="yes",
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

optargs    = opt:parse_args()
if type(optargs) == "string" then print(optargs) end


nnlm_conf = {}
if optargs.nnlm_conf then
  local t = dofile(optargs.nnlm_conf)
  for name,value in pairs(t) do
    if not nnlm_conf[name] then
      nnlm_conf[name] = value
    end
  end
end

multi_class        = ((optargs.multi_class or "no")=="yes")
vocabfilename      = optargs.vocabfilename or error("Needs a vocabulary")
langmodel_filename = optargs.langmodel_filename or error("Needs a language model")
test               = optargs.test or error("Needs an input text file")
debug              = tonumber(optargs.debug) or 0
trie_size          = tonumber(optargs.trie_size)
is_stream          = ((optargs.stream or "no") == "yes")
use_unk            = (optargs.unk or "all")
word2probs         = optargs.word2probs
optargs.cache_data = optargs.cache_data or nnlm_conf.cache_data
cache_data         = optargs.cache_data
optargs.null_token = optargs.cache_stop_token or nnlm_conf.cache_stop_token
cache_stop_token   = optargs.cache_stop_token or "<stop>"
optargs.null_token = optargs.null_token or nnlm_conf.null_token
null_token         = optargs.null_token or "<NULL>"
optargs.train_restriction = optargs.train_restriction or nnlm_conf.train_restriction
train_restriction  = ((optargs.train_restriction or "no" ) == "yes")
order              = tonumber(optargs.order)
use_bcc            = ((optargs.use_bcc or "yes" ) == "yes")
use_ecc            = ((optargs.use_ecc or "yes" ) == "yes")
max_softmax_constants = tonumber(optargs.max_softmax_constants or 0) or error ("Needs a number at max_softmax_constants")
print_stats = ((optargs.print_stats or "no") == "yes")

if word2probs then
  word2probs = ngram.load_word2prob_smooth_factor(word2probs)
end

vocab = lexClass.load(io.open(vocabfilename))
model = ngram.load_language_model(langmodel_filename,
				  vocab, "<s>", "</s>",
				  {
				    cache_size = 10, -- podria ser 0 ;)
				    trie_size  = trie_size,
				    word2probs = word2probs,
				    cache_data = cache_data,
				    max_trie_constants = max_softmax_constants
				  })

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

cronometro = util.stopwatch()
cronometro:reset()
cronometro:go()

resul = ngram.test_set_ppl(model,vocab,test,io.stdout,debug,
			   "<unk>","<s>","</s>",is_stream,use_unk,
			   model:has_cache(),
			   train_restriction,
			   cache_stop_token,
			   null_token,
			   order,
			   use_bcc,
			   use_ecc,
			   multi_class_table,
			   num_classes)

fprintf(io.stderr,
	"Time: %f\nTime/token: %f\nTime/sentence: %f\n",
	cronometro:read(),
	cronometro:read()/(resul.numwords + resul.numsentences),
	cronometro:read()/resul.numsentences)

if print_stats then
  model:printStats()
end
