opt = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "April lexicon handler",
  {
    index_name="text",
    description="Text file where extract the lexicon (It could be a Lua table with pairs {text,prob} )",
    short="t",
    argument="yes",
  },
  {
    index_name="vocfile",
    description="Take only words from this vocabulary, needs text parameter",
    short="v",
    argument="yes",
  },
  {
    index_name="freqvocfile",
    description="Frequency vocabulary extracted from previous running",
    short="f",
    argument="yes",
  },
  {
    index_name="cutsizes_tbl",
    description="Cut sizes table (absolute values)",
    short="c",
    argument="yes",
  },
  {
    index_name="ksizes_tbl",
    description="K-cut sizes table (abosulte frequency values)",
    short="k",
    argument="yes",
  },
  {
    index_name="addunk",
    description="Add <unk> word (default yes)",
    long="add-unk",
    argument="yes",
  },
  {
    index_name="addnull",
    description="Add <NULL> word (default no)",
    long="add-null",
    argument="yes",
  },
  {
    index_name="addstop",
    description="Add <stop> word (default no)",
    long="add-stop",
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

text    = optargs.text
freqvocfile = optargs.freqvocfile
vocfile = optargs.vocfile
addunk  = ((optargs.addunk or "yes") == "yes")
addnull = ((optargs.addnull or "no") == "yes")
addstop = ((optargs.addstop or "no") == "yes")

cutsizes_tbl = string.tokenize(optargs.cutsizes_tbl or "",",")
ksizes_tbl   = string.tokenize(optargs.ksizes_tbl or "", ",")

if (not text and not freqvocfile) or (text and freqvocfile) then
  error ("Needs a text or-exclusive voc!!!")
end

map    = {}
vocab  = {}
next   = 1
counts = 0

if vocfile then
  if not text then
    error("text parameter must be active when using vocfile")
  end
  count_unks = 0
  for line in io.lines(vocfile) do
    if line ~= "<s>" and line ~= "</s>" and line ~= "<unk>" and line ~= "<stop>" and line ~= "<NULL>" then
      map[line] = next
      vocab[next] = {
	w  = line,
	n  = 0,
	id = next
      }
      next = next + 1
    end
  end
end
min = 1000000
if text then
  if string.match(text, "%.lua") then
    textdata = dofile(text)
  else
    textdata = { { text, 1 } }
  end
  for _,d in ipairs(textdata) do
    local patfile=d[1]
    local prob=d[2]
    print(patfile, prob)
    local jaja=0
    if prob < min then min = prob end
    for line in io.lines(patfile) do
      jaja = jaja + 1
      if math.mod(jaja, 100000) == 0 then printf(".") io.stdout:flush() end
      local words = string.tokenize(line)
      for i,w in ipairs(words) do
	if vocfile then
	  if map[w] then
	    vocab[map[w]].n = vocab[map[w]].n + prob
	    counts = counts + prob
	  else
	    count_unks = count_unks + prob
	  end
	elseif not vocfile then
	  if not map[w] then
	    map[w] = next
	    next   = next + 1
	  end
	  vocab[map[w]] = vocab[map[w]] or {
	    w  = w,
	    id = map[w],
	    n  = 0,
	  }
	  vocab[map[w]].n = vocab[map[w]].n+prob
	  counts = counts + prob
	end
      end
    end
    printf("\n")
  end

  table.sort(vocab, function(a, b)
		      if a.n == b.n then return a.id < b.id end
		      return a.n > b.n
		    end)

  local f = io.open("voc.ALL", "w")
  f:write("<s>\n</s>\n")
  if addunk then
    f:write("<unk>\n")
  end
  if addnull then
    f:write("<NULL>\n")
  end
  if addstop then
    f:write("<stop>\n")
  end
  for i,data in ipairs(vocab) do
    fprintf(f, "%s\n", data.w)
  end
  f:close()
  
  local f = io.open("voc.freq.ALL", "w")
  --local g = io.open("voc.freq2.ALL", "w")
  if count_unks then
    fprintf(f, "%12f\t<unk>\n", count_unks)
  end
  for i,data in ipairs(vocab) do
    if vocfile and data.n == 0 then data.n = min end
    fprintf(f, "%12f\t%s\n", data.n, data.w)
    --  fprintf(g, "%s\n", data.w)
  end
  f:close()
  --g:close()

else
  for line in io.lines(freqvocfile) do
    local t = string.tokenize(line)
    map[t[2]] = next
    vocab[next] = {
      w  = t[2],
      n  = tonumber(t[1]),
      id = next
    }
    counts = counts + vocab[next].n
    next = next + 1
  end
end

for j=1,#cutsizes_tbl do
  local sum=0
  cutsize = tonumber(cutsizes_tbl[j]) - 2
  local g = io.open("voc.".. cutsize, "w")
  g:write("<s>\n</s>\n")
  if addunk then
    g:write("<unk>\n")
    cutsize = cutsize - 1
  end
  if addnull then
    g:write("<NULL>\n")
    cutsize = cutsize - 1
  end
  if addstop then
    g:write("<stop>\n")
    cutsize = cutsize - 1
  end
  for i=1,cutsize do
    fprintf(g, "%s\n", vocab[i].w)
    sum = sum + vocab[i].n
  end
  g:close()
end

for j=1,#ksizes_tbl do
  ksize = tonumber(ksizes_tbl[j])
  local g = io.open("voc.K".. ksize, "w")
  g:write("<s>\n</s>\n")
  if addunk then
    g:write("<unk>\n")
  end
  if addnull then
    g:write("<NULL>\n")
  end
  if addstop then
    g:write("<stop>\n")
  end
  for i=1,#vocab do
    if vocab[i].n <= ksize then break end
    fprintf(g, "%s\n", vocab[i].w)
  end
  g:close()
end
