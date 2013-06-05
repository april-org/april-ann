opt = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "WER/CER extractor",
  {
    index_name="type",
    description="WER or CER or SER",
    short="t",
    argument="yes",
  },
  {
    index_name="nbest",
    description="Nbest lists.",
    short="n",
    argument="yes",
  },
  {
    index_name="reference",
    description="Reference transcriptions.",
    short="r",
    argument="yes",
  },
  {
    index_name="feats",
    description="Output features.",
    short="F",
    argument="yes",
  },
  {
    index_name="scores",
    description="Output scores.",
    short="S",
    argument="yes",
  },
  {
    index_name="prev_feats",
    description="Previous features.",
    short="E",
    argument="yes",
  },
  {
    index_name="prev_scores",
    description="Previous scores.",
    short="R",
    argument="yes",
  },
  { index_name="filter",
    description="Lua filter (post-process).",
    short="f",
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

optargs = opt:parse_args(args)

if not optargs.nbest or not optargs.reference then
  error("Needs a target and a reference files!!!")
end

local rf = io.open(optargs.reference)

if optargs.filter then
  filter = dofile(optargs.filter)
else
  filter = function (str) return str end
end

local currentn = 0
local correcta

local type   = optargs.type or "WER"
local nbestf = io.open(optargs.nbest)
lista_tasas  = {}
features     = {}

correcta=rf:read("*l")

scoresf = io.open(optargs.scores, "w")
featsf  = io.open(optargs.feats, "w")

if optargs.prev_scores then
  prev_scoresf = io.open(optargs.prev_scores, "r")
  prev_featsf  = io.open(optargs.prev_feats, "r")
end

local kk = 0
while true do
  nbest_line=nbestf:read("*l")
  if nbest_line then
    n,reconocida,f,p    = string.match(nbest_line,
				       "(.*)|||(.*)|||(.*)|||(.*)")
    if n == nil then
      reconocida = nbest_line
      f = "0"
      p = "0"
      n  = kk
      kk = kk + 1
    end
  end
  if not nbest_line or tonumber(n) ~= currentn then
    fprintf(io.stderr,".")
    io.stdout:flush()
    local output_scores   = {}
    local output_features = {}
    local size = 0
    -- procesamos las NBEST
    local resul_val
    if type == "WER" then
      resul_val = tasas{
	typedata = "pairs_lines",
	data = lista_tasas,
	tasa = "raw",
      }
    elseif type == "CER" then
      resul_val = tasas{
	typedata = "pairs_lines",
	data = lista_tasas,
	tasa = "raw",
	words_width = 1,
      }
    elseif type == "SER" then
      resul_val = {}
      for i=1,#lista_tasas do
	resul_val[i] = { na=0, ns=0, ni=0, nb=0 }
	if lista_tasas[i][1] == lista_tasas[i][2] then
	  resul_val[i].na = 1
	else resul_val[i].ns = 1
	end
      end
    else
      error ("Incorrect type: " .. type)
    end
    for i,j in ipairs(resul_val) do
      -- Aciertos Sustituciones Inserciones Borrados
      table.insert(output_scores, string.format("%d %d %d %d", j.na, j.ns, j.ni, j.nb))
      size = size + 1
    end
    if prev_scoresf then
      local line = prev_scoresf:read("*l")
      while true do
	local line = prev_scoresf:read("*l")
	if line ~= "SCORES_TXT_END_0" then
	  table.insert(output_scores, line)
	    size = size + 1
	else break
	end
      end
    end
    
    fprintf(scoresf,"SCORES_TXT_BEGIN_0 %d %d 4 Acc Sub Ins Del\n", currentn, size)
    fprintf(scoresf, table.concat(output_scores, "\n").."\n")
    fprintf(scoresf,"SCORES_TXT_END_0\n")
    
    fprintf(featsf,"FEATURES_TXT_BEGIN_0 %d %d 3 hmm lm wip\n", currentn, size)
    fprintf(featsf,table.concat(features, "\n").."\n")
    if prev_featsf then
      local line = prev_featsf:read("*l")
      while true do
	local line = prev_featsf:read("*l")
	if line ~= "FEATURES_TXT_END_0" then
	  fprintf(featsf, "%s\n", line)
	else break
	end
      end
    end
    fprintf(featsf,"FEATURES_TXT_END_0\n")
    if not nbest_line then break end
    -- reiniciamos
    lista_tasas = {}
    features    = {}
    correcta=rf:read("*l")
    if not correcta then error("Incorrect number of sentences") end
    size = 0
    currentn = tonumber(n)
  end
  table.insert(lista_tasas, {filter(correcta), filter(reconocida)})
  table.insert(features, f)
end
fprintf(io.stderr,"\n")

nbestf:close()
scoresf:close()
featsf:close()
if prev_featsf then prev_featsf:close() end
if prev_scoresf then prev_scoresf:close() end
