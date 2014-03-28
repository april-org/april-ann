get_table_from_dotted_string("ngram", true)

-- Auxiliar functions
local function exp10(value)
  return math.pow(10, value)
end

local function log10(value)
  return math.log(value) / math.log(10)
end

local function print_pw(log_file, word_id, p)
  fprintf (log_file, "\tp( %d | ... ) = %f\n", word_id, p)
end

function language_models.get_sentence_prob(params)
  local params = get_table_fields(
    {
      lm         = { mandatory = true },
      words_it   = { mandatory = true, isa_match = iterator },
      log_file   = { mandatory = false, default = io.stdout },
      debug_flag = { mandatory = false, type_match = "number", default = 0 },
      unk_id     = { mandatory = false, type_match = "number", default = -1 },
      use_unk    = { mandatory = false, type_match = "string", default = "all" },
      use_bcc    = { mandatory = false },
      use_ecc    = { mandatory = false }
    }, params)

  local lm = params.lm
  local words_it = params.words_it
  local log_file = params.log_file
  local debug_flag = params.debug_flag
  local unk_id = params.debug_flag
  local final_id = params.final_id
  local use_unk = params.use_unk
  local use_bcc = params.use_bcc
  local use_ecc = params.use_ecc
  local sum = 0
  local numwords = 0
  local numunks = 0
  local lmi = lm:get_interface()
  local ngram_value = lm:ngram_order()
  local lastunk = -ngram_value
  local not_used_words = 0
  local i = 1
  local key
  local p
  local result

  assert(lm:is_deterministic(),
         "Error: Expected a deterministic LM")

  if use_bcc then 
    key = lmi:get_initial_key()
  else 
    key = lmi:get_zero_key()
  end

  for word in words_it() do
    -- If word id is -1, unknown words aren't
    -- considered by the model. We must skip
    -- the addition, store the unknown word
    -- index and set key to zero key
    if word == -1 then
      numunks = numunks + 1
      lastunk = i
      key = lmi:get_zero_key()
    -- If unknown words don't appear on context
    elseif i - lastunk >= ngram_value then
	result = lmi:get(key, word)
	key,p = result:get(1)
	-- If current word is unknown, we store
	-- the index and sum its probability if
	-- we consider all unknown words
	if word == unk_id then
	  numunks = numunks + 1
	  lastunk = i
	  if use_unk == "all" then
	    p   = p / math.log(10)
	    sum = sum + p
	  end
	-- If it's known, we sum its probability
	else
	  p   = p / math.log(10)
	  sum = sum + p
	end
     -- If last unknown word is on context, then
     -- we add its probability if we consider all
     -- or only context unknown words
     else
	result = lmi:get(key, word)
	key,p = result:get(1)
	if use_unk == "none" then
	  not_used_words = not_used_words + 1
	else
	  p   = p / math.log(10)
	  sum = sum + p
	end
    end
    i = i + 1
    print_pw(log_file, word, p)
  end

  numwords = i - 1
  if use_unk ~= "all" then
    numwords = numwords - numunks - not_used_words
  end

  if use_ecc then
    result = lmi:get(key, final_id)
    _,p = result:get(1)
    p   = p / math.log(10)
    sum = sum + p
  end
  
  if debug_flag >= 1 then
    fprintf (log_file, "%d sentences, %d words, %d OOVs\n",
	     1, numwords, numunks)
    fprintf (log_file, "0 zeroprobs, logprob= %.4f ppl= %.3f ppl1= %.3f\n",
	     sum,
	     exp10(-sum/(numwords+ ((use_ecc and 1) or 0) )),
	     exp10(-sum/numwords))
    fprintf (log_file, "\n")
  end
  log_file:flush()

  return sum,numwords,numunks
end

function language_models.get_prob_from_id_tbl(lm, word_ids, init_id, final_id,
				    use_bcc, use_ecc)
  local lmi = lm:get_interface()
  local key = lmi:get_initial_key()
  if not use_bcc then key = lmi:get_zero_key() end
  local sum      = 0
  local result
  local p
  local numwords    = #word_ids
  local ngram_value = lm:ngram_order()
  for i=1,numwords do
    if word ~= 0 then
      result = lmi:get(key, word)
      assert(result:size() == 1,
             "Error: Expected a deterministic LM")
      key,p = result:get(1)
      sum = sum + p
    end
  end
  if use_ecc then
    result = lmi:get(key, final_id)
    assert(result:size() == 1,
           "Error: Expected a deterministic LM")
    _,p = result:get(1)
    sum = sum + p
  end
  return sum
end

function language_models.test_set_ppl(lm, vocab, testset, flog, debug_flag,
			    unk_word, init_word, final_word,
			    is_stream,
			    use_unk,
			    use_cache,
			    train_restriction,
			    cache_stop_token,
			    null_token,
			    order,
			    use_bcc,
			    use_ecc,
			    multi_class_table,
			    num_classes)

  local multi_class_table_inv
  if multi_class_table then
    multi_class_table_inv = table.invert(multi_class_table)
  end
  local totalwords     = 0
  local totalsentences = 0
  local totalunks      = 0
  local totalsum       = 0
  local unk_word       = unk_word or "<unk>"
  local init_word      = init_word or "<s>"
  local final_word     = final_word or "</s>"
  local unk_id         = -1
  if vocab:getWordId(unk_word) then unk_id = vocab:getWordId(unk_word) end
  local init_id  = vocab:getWordId(init_word)
  local final_id = vocab:getWordId(final_word)
  -- lm:restart()
  local count = 0
  local lines_it = iterator(io.lines(testset)):
  map( function(line) return iterator(line:gmatch("[^%s]+")) end )

  for words_it in lines_it() do
    words_it = words_it:map( function(w) return vocab:getWordId(w) or unk_id end )
    local use_sentence = true
    if train_restriction then
      if words[1] ~= "<train>" then use_sentence = false end
      table.remove(words, 1)
    end
    if use_sentence then
      count = count + 1
      --if #sentence > 0 and #words > 0 then
	--if debug_flag >= 1 then fprintf(flog, "%s\n", sentence) end
	local sum,numwords,numunks =
	  ngram.get_sentence_prob(lm, words_it, flog, debug_flag,
				  unk_id, final_id,
				  use_unk, use_bcc, use_ecc)
	totalsum       = totalsum + sum
	totalwords     = totalwords + numwords
	totalunks      = totalunks + numunks
	totalsentences = totalsentences + 1
      --end
      -- if use_cache or math.mod(count, 1000) == 0 then lm:restart() end
      -- if use_cache then lm:restart() end
    end
    --[[
    if use_cache then
      local word_ids
      word_ids = vocab:searchWordIdSequence(words, unk_id)
      for i=1,#word_ids do
	if words[i] == cache_stop_token then
	  lm:clearCache()
	elseif words[i] ~= null_token then
	  lm:cacheWord(word_ids[i])
	end
      end
      --lm:showCache()
    end
    ]]--
  end
  -- lm:restart()
  local entropy  = -totalsum/(totalwords + ((use_ecc and totalsentences) or 0))
  local ppl      = exp10(entropy)
  local entropy1 = -totalsum/totalwords
  local ppl1     = exp10(entropy1)

  if is_stream then ppl = ppl1 entropy = entropy1 end
  
  if debug_flag >= 0 then
    fprintf (flog, "file %s: %d sentences, %d words, %d OOVs\n",
	     testset,
	     totalsentences,
	     totalwords,
	     totalunks)
    flog:flush()
    fprintf (flog,
	     "0 zeroprobs, logprob= %f ppl= %f ppl1= %f\n",
	     totalsum,
	     ppl, ppl1)
    flog:close()
  end
  
  return {
    ppl          = ppl,
    ppl1         = ppl1,
    logprob      = totalsum,
    numwords     = totalwords,
    numunks      = totalunks,
    numsentences = totalsentences,
  }
end
