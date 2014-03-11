vocab = lexClass.load(io.open("vocab"))
model = ngram.lira.model{
  command="zcat dihana3gram.lira.gz",
  vocabulary=vocab:getWordVocabulary(),
  fan_out_threshold=10
}

 sum,numwords,numunks =
   ngram.get_sentence_prob(model, vocab,
 			  string.tokenize("quer'ia un tren con "..
 					  "destino a barcelona"),
 			  io.stdout, 2,
 			    -1, vocab:getWordId("<s>"),
 			  vocab:getWordId("</s>"))

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
