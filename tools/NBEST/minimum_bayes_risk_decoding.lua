-- PAPER: "Minimum Bayes Risk Decoding for BLEU", Nicola Ehling and
-- Richard Zens and Hermann Ney, in ACL 2007

-- REVISAR TAMBIEN: "Task-Specific Minimum Bayes-Risk Decoding using
-- Learned Edit Distance", Izhak Shafran and William Byrne

opt = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "Minimum Bayes Risk Decoding in April toolkig",
  {
    index_name="t",
    description="NBest target file",
    short="t",
    argument="yes",
  },
  {
    index_name="f",
    description="Loss function (WER, SER, CER, BLEU, PER)",
    short="f",
    argument="yes",
  },
  {
    index_name="n",
    description="Take the first N hipothesys for compute risks (default take ALL)",
    short="n",
    argument="yes",
  },
  {
    index_name="k",
    description="Extract the one-best from the K first hipothesis, but using the rest N for compute risks (default takes N previous parameter)",
    short="k",
    argument="yes",
  },
  {
    index_name="l",
    description="L0 value (default 1)",
    short="l",
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

optargs = opt:parse_args()
if type(optargs) == "string" then error(optargs) end

nbest    = optargs.t or error ("Needs a NBest file")
lossf    = optargs.f or error ("Needs a Loss function")
maxnbest = tonumber(optargs.n or -1)
lambda0  = tonumber(optargs.l or  1.0)
maxk     = tonumber(optargs.k or -1)

function logAdd(a, b)
  if a>b then
    return a + math.log(1 + math.exp(b-a))
  else
    return b + math.log(1 + math.exp(a-b))
  end
end

id2voc = {}
voc2id = {}
nextid = 1

f          = io.open(nbest, "r")
data       = {}
uniq       = {}
mem_counts = {}
sum        = 0
currentn   = 0
next_sentence_id = 1

function normalize1(s)
  --s = string.lower(s)
  return table.concat(string.tokenize(s), " ")
end

function normalize2(s)
  s = string.gsub(s, "([^%s])([%'\"%(%)%[%]%,%.:;%-%!%?%_%%%+%-%*%$])", "%1 %2")
  s = string.gsub(s, "([%'\"%(%)%[%]%,%.:;%-%!%?%_%%%+%-%*%$])([^%s])", "%1 %2")
  --s = string.gsub(s, "[%'\"%(%)%[%]%,%.:;%-%!%?%_%%%+%-%*%$]", "")
  --s = string.lower(s)
  return table.concat(string.tokenize(s), " ")
end

function normalize3(s)
  --return string.gsub(s, "%s*", "")
  return table.concat(string.tokenize(s), " ")
end

function normalize4(s)
  s = string.lower(s)
  --s = string.gsub(s, "([^%s])([%'\"%(%)%[%]%,%.:;%-%!%?%_%%%+%-%*%$])", "%1 %2")
  --s = string.gsub(s, "([%'\"%(%)%[%]%,%.:;%-%!%?%_%%%+%-%*%$])([^%s])", "%1 %2")
  s = string.gsub(s, "[%'\"%(%)%[%]%,%.:;%-%!%?%_%%%+%-%*%$]", "")
  return table.concat(string.tokenize(s), " ")
end

function compute_none(s)
  return { s = s }
end

function compute_grams(s)
  s = string.tokenize(s)
  r = {
    [1] = {},
    [2] = {},
    [3] = {},
    [4] = {},
    len = #s,
  }
  local gram2 = {}
  local gram3 = {}
  local gram4 = {}
  for i=1,#s do
    if not voc2id[s[i]] then
      voc2id[s[i]]   = nextid
      id2voc[nextid] = s[i]
      nextid         = nextid + 1
    end
    s[i] = voc2id[s[i]]
    local gram1 = s[i]
    r[1][gram1] = (r[1][gram1] or 0) + 1
    if i-1 > 0 then
      local gram2 = table.concat(s, "#", i-1, i)
      r[2][gram2] = (r[2][gram2] or 0) + 1
      if i-2 > 0 then
	local gram3 = table.concat(s, "#", i-2, i)
	r[3][gram3] = (r[3][gram3] or 0) + 1
	if i-3 > 0 then
	  local gram4 = table.concat(s, "#", i-3, i)
	  r[4][gram4] = (r[4][gram4] or 0) + 1
	end
      end
    end
  end
  return r
end

function compute_words(s)
  s = string.tokenize(s)
  r = {
    words = {},
    len = #s,
  }
  for i=1,#s do
    r.words[s[i]] = (r.words[s[i]] or 0) + 1
  end
  return r
  
end

function compute_wer(hyp, hypp)
  local resul = tasas{
    typedata = "pairs_lines",
    data = { {hyp.auxdata.s, hypp.auxdata.s} },
    tasa = "ie",
  }
  return resul.tasa*0.01
end

function compute_cer(ref, hyp)
  local resul = tasas{
    typedata = "pairs_lines",
    data = { {ref.auxdata.s, hyp.auxdata.s} },
    tasa = "ie",
    words_width = 1,
  }
  return resul.tasa*0.01
end

function compute_ser(ref, hyp)
  if ref == hyp then return 0 end
  return 1
end

function compute_per(ref, hyp)
  local diff1=0
  local diff2=0
  for w,rep in pairs(ref.auxdata.words) do
    diff1 = diff1 + math.abs(rep - (hyp.auxdata.words[w] or 0))
  end
  for w,rep in pairs(hyp.auxdata.words) do
    diff2 = diff2 + math.abs(rep - (ref.auxdata.words[w] or 0))
  end
  return (diff1 + diff2)/ref.auxdata.len
end

-- hyp es la REFERENCIA y hypp es la hipotesis ???
function compute_bleu(hyp, hypp)
  -- hypp and hyp are tables with
  --		   {
  --                 id,
  --		     score    = v,
  --		     sentence = sentence,
  --		     grams    = {
  --                   grams1, grams2, grams3, grams4, len
  --                 }
  --		   }
  local BP = 0
  if hypp.auxdata.len < hyp.auxdata.len then
    BP = 1 - hyp.auxdata.len/hypp.auxdata.len
  end
  local counts
  local prod  = 0
  local memid = hyp.id .. "_" .. hypp.id
  if hypp.id < hyp.id then
    memid = hypp.id .. "_" .. hyp.id
  end
  if mem_counts[memid] then
    counts = mem_counts[memid]
  else
    counts = {0,0,0,0}
    for i=1,4 do
      for gram,c in pairs(hyp.auxdata[i]) do
	if hypp.auxdata[i][gram] then
	  counts[i] = counts[i] + math.min(c, hypp.auxdata[i][gram])
	end
      end
      if i > 1 then
	counts[i] = math.log(counts[i] + 1)
      else
      counts[i] = math.log(counts[i])
      end
    end
    mem_counts[memid] = counts
  end
  prod = prod + (counts[1] - math.log(hypp.auxdata.len))
  for i=2,4 do
    --prod = prod + (counts[i] - math.log(hyp.auxdata.len - i + 1))
    prod = prod + (counts[i] - math.log(hypp.auxdata.len - i + 2))
  end
  prod=prod*0.25
  return 1 - math.exp(prod+BP)
end

if lossf == "BLEU" then
  compute_loss    = compute_bleu
  compute_auxdata = compute_grams
  normalize       = normalize1
elseif lossf == "WER" then
  compute_loss    = compute_wer
  compute_auxdata = compute_none
  normalize       = normalize3
elseif lossf == "CER" then
  compute_loss    = compute_cer
  compute_auxdata = compute_none
  normalize       = normalize3
elseif lossf == "SER" then
  compute_loss    = compute_ser
  compute_auxdata = compute_none
  normalize       = normalize4
elseif lossf == "PER" then
  compute_loss    = compute_per
  compute_auxdata = compute_words
  normalize       = normalize3
else
  error("Incorrect loss function: " .. lossf)
end

while true do
  local line = f:read("*l")
  local n,sentence,feats,score
  n = -1
  if line then
    n,sentence,feats,score = string.match(line, "(.*)|||(.*)|||(.*)|||(.*)")
    n = tonumber(n)
    score = tonumber(score)
  end
  if n ~= currentn then
    -- Compute posteriors
    for i=1,#data do data[i].posterior = math.exp(data[i].score - sum) end
    table.sort(data, function(a,b) return a.posterior > b.posterior end)
    -- compute MBR
    local min  = nil
    local best = 1
    local last = #data
    if maxk ~= -1 then last = math.min(maxk, #data) end
    for i=1,last do
      local score = 0
      for j=1,#data do
	-- si son frases identicas, la funcion de perdida es 0, y por
	-- tanto no hay que entrar en este if
	if j ~= i and data[i].sentence ~= data[j].sentence then
	  -- calculamos la funcion de perdida (1 - BLEU)
	  local lossv = compute_loss(data[i], data[j])
	  --print(lossv)
	  --print(data[i].sentence)
	  --print(data[j].sentence)
	  --print(j, data[j].posterior, data[j].score, lossv, score, min)
	  --print(lossv, math.exp(data[j].score), data[j].posterior, math.exp(sum))
	  
	  -- sumamos a la puntuacion de esta frase la combinacion del
	  -- la probabilidad a posteriori y la funcion de perdida, y
	  -- asi calculamos el riesgo
	  score = score + data[j].posterior*lossv
	end
	-- si el score supera el mejor minimo, paramos
	if min and score > min then break end
      end
      --printf("%f %s\n", score, data[i].sentence)
      if not min or score < min then
	-- nos quedamos con aquella frase que tiene menor riesgo
	min  = score
	best = i
      end
    end
    print(table.concat(string.tokenize(data[best].output), " "))
    fprintf(io.stderr, "%-14g %s (%d)\n", min, data[best].sentence, data[best].id)
    io.stdout:flush()
    io.stderr:flush()
    -- RESET, hemos terminado con esta lista
    data       = {}
    mem_counts = {}
    uniq       = {}
    next_sentence_id = 1
    currentn   = n
    sum        = 0
    collectgarbage("collect")
  end
  if line then
    local realsentence = sentence
    sentence = normalize(sentence)
    if maxnbest == -1 or #data < maxnbest then
      local id = uniq[sentence]
      if not id then
	id               = next_sentence_id
	uniq[sentence]   = next_sentence_id
	next_sentence_id = next_sentence_id + 1
      end
      -- para ajustar la escala
      local v = score*lambda0
      -- sum for posteriors
      sum = logAdd(sum, v)
      if not data[id] then
	data[id] = {
	  id = id,
	  score = v,
	  sentence = sentence,
	  output  = realsentence,
	  auxdata = compute_auxdata(sentence),
	}
      else
	data[id].score = logAdd(data[id].score, v)
      end
    end
  else break
  end
end

