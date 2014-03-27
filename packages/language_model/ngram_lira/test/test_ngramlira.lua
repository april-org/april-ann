local path = arg[0]:get_path()
local vocab = lexClass.load(io.open(path .. "vocab"))
local model = language_models.load(path .. "dihana3gram.lira.gz",
				  vocab, "<s>", "</s>")
local phrase = "quer'ia un tren con destino a barcelona"
local lines_it = iterator(io.lines("frase")):
map( function(line) return iterator(line:gmatch("[^%s]+")) end )
for words_it in lines_it() do
  words_it = words_it:map( function(w) return vocab:getWordId(w) or unk_id end )
  local sum,numwords,numunks =
    language_models.get_sentence_prob{ 
    					    lm = model,
					    words_it = words_it,
					    debug_flag = 2
					  }
end

--print("TEST 2")
--
--model:prepare(key)
--prob,key = model:get(key, vocab:getWordId("quer'ia"))
--model:prepare(key)
--prob,key = model:get(key, vocab:getWordId("un"))
--model:prepare(key)
--prob,key = model:get(key, vocab:getWordId("billete"))
--printf("From %d ", key)
--model:prepare(key)
--prob,key = model:get(key, vocab:getWordId("a"))
--printf("To %d with %d = %f\n", key,
--       vocab:getWordId("a"), prob)
--
--key = model:findKeyFromNgram(vocab:searchWordIdSequence(string.tokenize("<s> quer'ia un billete")))
--model:prepare(key)
--printf("From %d ", key)
--prob,key = model:get(key, vocab:getWordId("a"))
--printf("To %d with %d = %f\n", key,
--       vocab:getWordId("a"), prob)
