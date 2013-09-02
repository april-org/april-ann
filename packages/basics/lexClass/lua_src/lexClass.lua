local lexClass_methods,
lexClass_class_table = class("lexClass")

function lexClass_class_table:__call(voc_id2word)
  local obj = {
    cobj  = _internal_lexclass_(),
    units = {}
  }
  obj = class_instance(obj, self, true)
  if voc_id2word then
    for _,word in ipairs(voc_id2word) do
      obj:addPair{
	word = word,
      }
    end
  end
  return obj
end

function lexClass_methods:addPair(t)
  local word   = t.word   or error ("Falta definir word")
  local outsym = t.outsym or word
  local prob   = t.prob or 1.0
  local units  = t.units or {}
  local rid    = self.cobj:addPair{
    word=word,
    outsym=outsym,
    prob=math.log(prob)
  }
  if #units > 0 then self.units[rid] = units end
  return rid
end

function lexClass_methods:getWordFromPairId(pairid)
  return self.cobj:getWordFromPairId(pairid)
end

function lexClass_methods:getWordFromWordId(wid)
  return self.cobj:getWordFromWordId(wid)
end

function lexClass_methods:getOutSymFromPairId(pairid)
  return self.cobj:getOutSymFromPairId(pairid)
end

function lexClass_methods:getOutSymFromOutSymId(oid)
  return self.cobj:getOutSymFromOutSymId(oid)
end

function lexClass_methods:getWordId(word)
  return self.cobj:getWordId(word)
end

function lexClass_methods:getOutSymId(outsym)
  return self.cobj:getOutSymId(outsym)
end

-- Outsym [Word] prob t1 t2 t3 ...
function lexClass.load(dict_file)
  local lex = lexClass()
  for line in dict_file:lines() do

    local l = string.tokenize(line)
    local outsym = l[1]
    local begin_units = 2
    local prob        = 1
    local w           = outsym
    
    if #l>1 and string.sub(l[2],1,1)=="[" and string.sub(l[2],#l[2],#l[2])=="]" then
      w = string.sub(l[2], 2, #l[2]-1)
      begin_units = begin_units + 1
    end
    
    local units = {}
    if tonumber(l[begin_units]) ~= nil then
      prob        = tonumber(l[begin_units])
      begin_units = begin_units + 1
    end
    for i=begin_units,#l do
      table.insert(units, l[i])
    end
    
    -- anyadimos la palabra y toda su informacion
    local rid = lex:addPair{
      word   = w,
      outsym = outsym,
      prob   = prob,
      units  = units,
    }
  end

  return lex
end

-- para guardar en disco
function lexClass_methods:save(f)
--   for pairid=1,self.cobj:size() do
--     local info = self.cobj:getInfo(pairid)
--     --
--     f:write(self.oid2outsym[info.oid] .. " ")
--     f:write("[".. self.wid2word[info.wid] .."] ")
--     f:write(math.exp(info.prob) .." ")
--     f:write(table.concat(info.units, " "))
--     f:write("\n")
--     f:flush()
--   end
--   collectgarbage("collect")
  error("Deprecated!!! :'( hay que actualizar esta funcion")
end

function lexClass_methods:getWordVocabulary()
  local t = {}
  for i=1,self.cobj:wordTblSize() do
    table.insert(t, self.cobj:getWordFromWordId(i))
  end
  return t
end

function lexClass_methods:getOutSymVocabulary()
  local t = {}
  for i=1,self.cobj:outsymTblSize() do
    table.insert(t, self.cobj:getOutSymFromOutSymId(i))
  end
  return t
end

function lexClass_methods:searchWordsSequenceFromWIDs(wordid_seq)
  local words={}
  for i,id in ipairs(wordid_seq) do
    if id ~= 0 then
      table.insert(words, self.cobj:getWordFromWordId(id))
    end
  end
  return words
end

function lexClass_methods:searchWordsSequence(pairid_seq)
  local words={}
  for i,id in ipairs(pairid_seq) do
    local word = self.cobj:getWordFromPairId(id)
    if word then
      if #word ~= 0 then
	table.insert(words, word)
      end
    else
      table.insert(words,"<unk>")
    end
  end
  return words
end

function lexClass_methods:searchOutSymsSequenceFromOIDs(outsym_seq)
  error ("Deprecated!!!!")
  --   local outsyms={}
  --   for i,id in ipairs(outsym_seq) do
  --     if id ~= 0 then table.insert(outsyms, self.oid2outsym[id]) end
  --   end
  --   return outsyms
end

function lexClass_methods:searchOutSymsSequence(pairid_seq)
  local outsyms={}
  for i,id in ipairs(pairid_seq) do
    local word = self.cobj:getOutSymFromPairId(id)
    if word then
      if #word ~= 0 then
	table.insert(outsyms,word)
      end
    else
      table.insert(outsyms, "<unk>")
    end
  end
  return outsyms
end

function lexClass_methods:searchWordIdSequence(words, unk_id)
  local ids={}
  for i,w in ipairs(words) do
    local wid = self.cobj:getWordId(w) or unk_id
    if wid ~= 0 then table.insert(ids, wid) end
  end
  return ids
end

function lexClass_methods:searchWordIdSequenceFromPairID(pairid_seq)
  local ids={}
  for i,pid in ipairs(pairid_seq) do
    local wid = self.cobj:getPairData(pid).word
    if wid ~= 0 then table.insert(ids, wid) end
  end
  return ids
end

function lexClass_methods:searchOutSymIdSequenceFromPairID(pairid_seq)
  local ids={}
  for i,pid in ipairs(pairid_seq) do
    local oid = self.cobj:getPairData(pid).outsym
    if oid ~= 0 then table.insert(ids, oid) end
  end
  return ids
end

function lexClass_methods:wordTblSize()
  return self.cobj:wordTblSize()
end

function lexClass_methods:outsymTblSize()
  return self.cobj:outsymTblSize()
end

function lexClass_methods:pairsTblSize()
  return self.cobj:size()
end

function lexClass_methods:size()
  error("OBSOLOTE: lexClass:size()")
end

function lexClass_methods:getCObject()
  return self.cobj
end

function lexClass_methods:getData(pair_id)
  local t = self.cobj:getPairData(pair_id)
  t.units = self.units[pair_id] or {}
  return t
end

function lexClass_methods:getPair(pair_id)
  local wid,oid
  local t = self.cobj:getPairData(pair_id)
  wid,oid = t.word,t.outsym
  return wid,oid
end
