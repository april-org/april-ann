local check = utest.check

local path = arg[0]:get_path()
local vocab = lexClass.load(io.open(path .. "vocab"))
local model = language_models.load(path .. "dihana3gram.lira.gz",
                                   vocab, "<s>", "</s>")
local unk_id = -1

local result = language_models.test_set_ppl{
  lm = model,
  vocab = vocab,
  testset = path .. "frase",
  debug_flag = -1,
  use_bcc = true,
  use_ecc = true
}

check.lt( math.abs(result.ppl - 17.223860768396), 1e-03 )
check.lt( math.abs(result.ppl1 - 26.996595980386), 1e-03 )
check.lt( math.abs(result.logprob + 27.194871135223), 1e-03 )
check.eq( result.numsentences, 3 )
check.eq( result.numunks, 2 )
check.eq( result.numwords, 19 )

-----------------------------------------------------------------------------

local bunch_hashed_lira_model = ngram.lira.bunch_hashed_model{
  lira_model   = model,
  trie_vector  = util.trie_vector(18),
  init_word_id = vocab:getWordId("<s>"),
}

local result2 = language_models.test_set_ppl{
  lm = bunch_hashed_lira_model,
  vocab = vocab,
  testset = path .. "frase",
  debug_flag = -1,
  use_bcc = true,
  use_ecc = true,
}

for i,v in pairs(result) do check.eq( v, result2[i] ) end
