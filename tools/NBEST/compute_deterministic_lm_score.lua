if #arg < 3 or #arg > 7 then
  print("Sintaxis:")
  printf ("\t%s NBESTFILE NNLM VOCAB [trie_size=24 use_bcc=yes use_ecc=yes cache_extra_set=nil]\n",
	  string.basename(arg[0]))
  os.exit(128)
end
nbestfile            = arg[1]
lm_file              = arg[2]
voc_filename         = arg[3]
trie_size            = tonumber(arg[4] or 24)
use_bcc              = ((arg[5] or "yes") == "yes")
use_ecc              = ((arg[6] or "yes") == "yes")
cache_extra_set      = arg[7]
unk_word             = "<unk>"
begin_word           = "<s>"
end_word             = "</s>"
stop_word            = "<stop>"
null_word            = "<NULL>"
--------------------------------------------------

vocab     = lexClass.load(io.open(voc_filename))
words     = vocab:getWordVocabulary()
inv_words = table.invert(words)
unk_id    = vocab:getWordId(unk_word)
------------------------------------------------------------------------

collectgarbage("collect")

-- cargamos el modelo FAST
lm = ngram.load_language_model(lm_file, vocab, begin_word, end_word,
			       {trie_size=trie_size})
local lmi = lm:get_interface()
collectgarbage("collect")
--use_cache = lm:has_cache()

------------------------------------- GO!

time=util.stopwatch()
time:go()
local scores = {}
local ngrams = {}
local j=1
local prevn = nil
lm:restart()
nbestf = io.open(nbestfile, "r")
cachef = nil
--if cache_extra_set then
--  cachef = io.open(cache_extra_set, "r")
--end
--local last_best_line = nil

assert(lm:is_deterministic(),
       "Error: Expected a deterministic LM")

while true do
  local line = nbestf:read("*l")
  local n
  if line then n = tonumber(string.match(line, "(.*)|||.*|||.*|||")) + 1 end
  if not prevn then prevn = n last_best_line = line end
  if not line or n ~= prevn then
    --[[
    if cachef then
      local line  = cachef:read("*l")
      local words = string.tokenize(line)
      for i=1,#words do
        if words[i] == stop_word then lm:clearCache()
        elseif words[i] ~= null_word then
	        local w = dictionary:getWordId(words[i]) or dictionary:getWordId(unk_word)
          ngram_model:cacheWord(w)
        end
      end
    end
    --]]
    -- compute scores for each collected ngram
    for key,where in pairs(ngrams) do
      lmi:clear_queries()
      local sum=0
      for word,_ in pairs(where) do
        lmi:insert_query(key, word)
      end
      result = get_queries()
      local i = 1
      for word,d in pairs(where)
        _,p = result:get(i)
        for _,id in ipairs(d) do
          scores[id] = scores[id] + p
        end
        sum = sum + #d
      end
      --fprintf(io.stderr, "%d : %d\n", key, sum)
    end
    print(table.concat(scores, "\n"))
    scores    = {}
    ngrams    = {}
    prevn     = n
    --[[
    if use_cache or n and math.mod(n, 100) == 99 then lm:restart() end
    if use_cache and last_best_line then
      local words = string.tokenize(string.match(last_best_line, "|||(.*)|||.*|||"))
      local wids  = vocab:searchWordIdSequence(words, vocab:getWordId(unk_word))
      for i=1,#wids do lm:cacheWord(wids[i]) end
    end
    --]]
    last_best_line = line
  end
  if not line then break end
  
  line = string.match(line, "|||(.*)|||.*|||")
  -- collect all ngrams of this sentence for efficient computation
  local words = string.tokenize(line)
  local wids  = vocab:searchWordIdSequence(words, vocab:getWordId(unk_word))
  table.insert(scores, 0) --  frase actual
  -- begin context
  local context = {}
  local key
  if use_bcc then
    for i=1,lm:ngram_order()-1 do
      table.insert(context, vocab:getWordId(begin_word))
    end
    key = lmi:get_initial_key()
  else
    for i=1,lm:ngram_order()-1 do
      table.insert(context, 0)
    end
    key = lmi:get_zero_key()
  end
  for wpos=1,#wids do
    ngrams[key] = ngrams[key] or {}
    local w = wids[wpos]
    ngrams[key][w] = ngrams[key][w] or {}
    table.insert(ngrams[key][w], #scores)
    -- move one position the context
    table.remove(context, 1)
    table.insert(context, w)
    -- next key
    result = lmi:next_keys(key, w)
    assert(#result == 1)
    key = result:get(1)
  end
  if use_ecc then
    ngrams[key] = ngrams[key] or {}
    local w = vocab:getWordId(end_word)
    ngrams[key][w] = ngrams[key][w] or {}
    table.insert(ngrams[key][w], #scores)
  end
  j=j+1
  if math.mod(j, 250) == 0 then collectgarbage("collect") end
end
nbestf:close()
time:stop()
a,b=time:read()
fprintf(io.stderr, "TIME: cpu %f (wall %f)\n", a, b)
