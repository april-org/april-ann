april_set_doc("rates",
	      {
		class="function",
		summary="Computes error rate measures for HMMs decoding.",
		params={
		  ["data"] = "The data to be analyzed",
		  ["datatype"] = {
		    "The format of the data.",
		    "It could be\n",
		    "- 'lines' => data would be a vector of lines where each",
		    "line contains reference_sequence..separator_char..test_sequence.\n",
		    "- 'pairs_lines' => is a vector of pairs { ref_seq, test_seq }\n",
		    "- 'pairs_lwords' => is a vector of pairs of word lists:\n",
		    "{ { 'he', 'is' }, { 'she', 'is' } }\n",
		    "- 'pairs_int => is a vector of pairs of integer lists.",
		    "NOTE that 0 integer is reserved for internal purposes.\n",
		  },
		  ["words_width"] = {
		    "When datatype is 'lines' or 'pairs_lines' this parameter",
		    "indicates that lines would be converted in words",
		    "taking groups of words_width characters",
		    "(non extended ascii)."
		  },
		  ["words_sep"] = {
		    "When datatype is 'lines' or 'pairs_lines', this parameter",
		    "indicates that lines has words separated by the given",
		    "character. NOTE that this parameter is a Lua string used",
		    "as pattern for string.find function, so the % character",
		    "would be needed in some cases (see Lua reference).",
		  },
		  ["rate"] = {
		    "The type of error rate which you want to compute. It",
		    "could be:\n",
		    "- 'pra' => correct / total_words\n",
		    "- 'pre' => errors / total_words\n",
		    "- 'pa' => ACCURACY\n",
		    "- 'ie' => WER\n",
		    "- 'wer' => WER\n",
		    "- 'psb' => (substitutions + deletes) / total_words\n",
		    "- 'iep' => weighted WER\n",
		    "- 'iap' => weighted ACCURACY\n",
		  },
		  ["p"] = {
		    "A number which indicates the insertion and delete cost",
		    "(substitution is 1, and success is 0) [optional]. By",
		    "default is 1",
		  },
		  ["confusion_matrix"] = {
		    "A boolean value [optional], by default",
		    "is false. It forces to compute a confusion matrix"
		  },
		},
		outputs = {
		  ["ac"] = "Number of success",
		  ["borr"] = "Number of deletions",
		  ["sust"] = "Number of substitution",
		  ["ins"] = "Number of insertions",
		  ["rate"] = "Value of the indicated error rate",
		  ["confusion_matrix"]="The confusion matrix, only when needed",		  
		},
	      })

rates.types = {'pra','pre','pa','ip','ie','psb','iep','iap'}

function rates.lines2pairs_lines(lines,setsep)
  -- transforma una lista de cadenas en una lista de pares de cadenas
  local pair_lines = {}
  setsep = setsep or '%*'
  local pattern = "^(.*)["..setsep.."](.*)$"
  for i,line in ipairs(lines) do
    _,_, correct,test = string.find(line, pattern)
    table.insert(pair_lines,{correct,test})
  end
  return pair_lines
end

function rates.pairs_lines2pairs_lwords(pairs_lines,setsep)
  -- transforma una lista de pares de cadenas en una lista de pares de
  -- secuencias de palabras
  local pairs_lwords = {}
  for i,pair_lines in ipairs(pairs_lines) do
    table.insert(pairs_lwords,{
		   string.tokenize(pair_lines[1],setsep),
		   string.tokenize(pair_lines[2],setsep)
		 })
  end
  return pairs_lwords
end

function rates.pairs_lines2string_width(pairs_lines,width)
  -- transforma una lista de pares de cadenas en una lista de pares de
  -- secuencias de palabras
  local pairs_lwords = {}
  for i,pair_lines in ipairs(pairs_lines) do
    table.insert(pairs_lwords,{
		   string.tokenize_width(pair_lines[1],width),
		   string.tokenize_width(pair_lines[2],width),
		 })
  end
  return pairs_lwords
end

function rates.lwords2int(lwords,dictionary)
  local aux = {}
  for _,word in ipairs(lwords) do
    local iword = dictionary.dir[word]
    if (iword == nil) then
      dictionary.num = dictionary.num+1
      dictionary.dir[word] = dictionary.num
      iword = dictionary.num
      dictionary.inv[dictionary.num] = word
    end
    table.insert(aux,iword)
  end
  return aux
end

function rates.pairs_lwords2pairs_int(pairs_lwords)
  local dictionary = {dir={},inv={},num=0}
  dictionary.inv[0] = ""
  local pairs_int = {}
  for i,pair in ipairs(pairs_lwords) do
    table.insert(pairs_int,{
		   rates.lwords2int(pair[1],dictionary),
		   rates.lwords2int(pair[2],dictionary)
		 })
  end
  return pairs_int,dictionary
end

function rates.rates(tbl)
  local datatype = tbl.datatype or 'lines'
  local data = tbl.data
  local dictionary
  if datatype == "lines" then
    data = rates.lines2pairs_lines(data,tbl.line_sep)
    datatype = "pairs_lines"
  end
  if datatype == "pairs_lines" then
    if tbl.words_width then
      data = rates.pairs_lines2string_width(data,tbl.words_width)
    else
      data = rates.pairs_lines2pairs_lwords(data,tbl.words_sep)
    end
    datatype = "pairs_lwords"
  end
  if datatype == "pairs_lwords" then
    data,dictionary = rates.pairs_lwords2pairs_int(data)
    datatype = "pairs_int"
  end
  if datatype ~= "pairs_int" then
    fprintf(io.stderr,"rates Error: %s is not a valid datatype",
	    datatype)
    os.exit(256)
  end
  local the_rate = tbl.rate
  if the_rate == "wer" then
    the_rate = "ie"
  end
  if the_rate == "raw" then
    return rates.raw{
      int_data = data,
      p        = tbl.p,
    }
  else
    local confusion_matrix_dictionary = nil
    if dictionary then
      confusion_matrix_dictionary = dictionary.inv
    end
    return rates.ints{
      int_data = data,
      rate     = the_rate,
      p        = tbl.p,
      confusion_matrix            = tbl.confusion_matrix,
      confusion_matrix_dictionary = confusion_matrix_dictionary
    }
  end
end

-- hacer que la funcion rates.rates se llame rates
local m = getmetatable(rates)
function m.__call(x,tbl)
  if (tbl == nil) then
    april_help("rates")
  else
    return rates.rates(tbl)
  end
end

