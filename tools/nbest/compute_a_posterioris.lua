opt = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "A posterioris computation in April toolkit",
  {
    index_name="t",
    description="NBest target file",
    short="t",
    argument="yes",
  },
  {
    index_name="l",
    description="L0 value (default 1, in HTR/ASR inverse of GSF)",
    short="l",
    argument="yes",
  },
  {
    index_name="k",
    description="Use only first k-best",
    short="k",
    argument="yes",
  },
  {
    index_name="e",
    description="Show prob instead of logprob (defauly no)",
    short="e",
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
lambda0  = tonumber(optargs.l or  1.0)
maxk     = tonumber(optargs.k or -1)
doexp    = optargs.e

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
sum        = 0
currentn   = 0
next_sentence_id = 1

while true do
  local line = f:read("*l")
  local n,sentence,feats,score
  n = -1
  if line then
    n,sentence,feats,score = string.match(line, "(.*)|||(.*)|||(.*)|||(.*)")
    n = tonumber(n)
    score = tonumber(score)
    sentence = table.concat(string.tokenize(sentence), " ")
  end
  if n ~= currentn then
    -- Compute posteriors
    for i=1,#data do data[i].posterior = data[i].score - sum end
    table.sort(data, function(a,b) return a.posterior > b.posterior end)
    -- Print sentences
    for j=1,#data do
      if doexp == "yes" then data[j].posterior=math.exp(data[j].posterior) end
      printf("%d ||| %s ||| %g\n", currentn, data[j].sentence, data[j].posterior)
    end
    io.stdout:flush()
    io.stderr:flush()
    -- RESET, hemos terminado con esta lista
    data       = {}
    uniq       = {}
    next_sentence_id = 1
    currentn   = n
    sum        = 0
    collectgarbage("collect")
  end
  if line then
    if maxk == -1 or #data < maxk then
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
	data[id]          = {}
	data[id].score    = v
	data[id].sentence = sentence
      else data[id].score=logAdd(data[id].score,v)
      end
    end
  else break
  end
end

