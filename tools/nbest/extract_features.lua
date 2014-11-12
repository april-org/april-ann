opt = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "NBest feature extractor",
  {
    index_name="nbest",
    description="Nbest lists.",
    short="n",
    argument="yes",
  },
  {
    index_name="feats",
    description="Output features.",
    short="F",
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

if not optargs.nbest then
  error("Needs a target nbest file!!!")
end

if optargs.filter then
  filter = dofile(optargs.filter)
else
  filter = function (str) return str end
end

local currentn = 0
local correcta

local nbestf = io.open(optargs.nbest)
features     = {}
featsf       = io.open(optargs.feats, "w")

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
    local output_features = {}
    local size = 0
    -- procesamos las NBEST
    local nfeats = #string.tokenize(features[1])
    fprintf(featsf,"FEATURES_TXT_BEGIN_0 %d %d %d", currentn, #features, nfeats)
    for i=1,nfeats do fprintf(featsf, " f%d", i) end fprintf(featsf, "\n")
    fprintf(featsf,table.concat(features, "\n").."\n")
    fprintf(featsf,"FEATURES_TXT_END_0\n")
    if not nbest_line then break end
    -- reiniciamos
    features    = {}
    size = 0
    currentn = tonumber(n)
  end
  table.insert(features, f)
end
fprintf(io.stderr,"\n")

nbestf:close()
featsf:close()
