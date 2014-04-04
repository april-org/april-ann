local path = arg[0]:get_path()
local vocab = lexClass.load(io.open(path .. "vocab"))
local model = language_models.load(path .. "dihana3gram.lira.gz",
                                   vocab, "<s>", "</s>")
local unk_id = -1
local lines_it 

print("----------------------------------------------------------------------")
print("-------------------------- USE_UNK = ALL -----------------------------")
print("----------------------------------------------------------------------")
print("\n")

lines_it = iterator(io.lines(path .. "frase")):
map( function(line) return line,iterator(line:gmatch("[^%s]+")) end )

for line,words_it in lines_it() do
  print(line)
  words_it = words_it:map( function(w) return (vocab:getWordId(w) or unk_id), w end )
  local sum,numwords,numunks =
    language_models.get_sentence_prob{ 
      lm = model,
      words_it = words_it,
      debug_flag = 2,
      unk_id = unk_id,
      use_ecc = true,
      use_unk = "all"
    }
  print("SUM:",sum,"Num words:", numwords, "OOVs:", numunks,"\n")
end

-- forcing unknown word to be "billete" we can reproduce use_unk posibilities
local unk_id = vocab:getWordId("billete")

print("----------------------------------------------------------------------")
print("------------------------ USE_UNK = CONTEXT ---------------------------")
print("----------------------------------------------------------------------")
print("\n")

lines_it = iterator(io.lines(path .. "frase")):
map( function(line) return line,iterator(line:gmatch("[^%s]+")) end )

for line,words_it in lines_it() do
  print(line)
  words_it = words_it:map( function(w) return (vocab:getWordId(w) or unk_id), w end )
  local sum,numwords,numunks =
    language_models.get_sentence_prob{ 
      lm = model,
      words_it = words_it,
      debug_flag = 2,
      unk_id = unk_id,
      use_unk = "context"
    }
  print("SUM:",sum,"Num words:", numwords, "OOVs:", numunks,"\n")
end

print("----------------------------------------------------------------------")
print("------------------------- USE_UNK = NONE -----------------------------")
print("----------------------------------------------------------------------")
print("\n")

lines_it = iterator(io.lines(path .. "frase")):
map( function(line) return line,iterator(line:gmatch("[^%s]+")) end )

for line,words_it in lines_it() do
  print(line)
  words_it = words_it:map( function(w) return (vocab:getWordId(w) or unk_id), w end )
  local sum,numwords,numunks =
    language_models.get_sentence_prob{ 
      lm = model,
      words_it = words_it,
      debug_flag = 2,
      unk_id = unk_id,
      use_unk = "none"
    }
  print("SUM:",sum,"Num words:", numwords, "OOVs:", numunks,"\n")
end
