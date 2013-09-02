cmd = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "Generation of Oversegmenter strings from viterbi forced alignment",
  {
    index_name="a",
    description="Force viterbi alignment file indexes",
    short="a",
    argument="yes",
  },
  {
    index_name="t",
    description="Phone transcription file indexes",
    short="t",
    argument="yes",
  },
  {
    index_name="m",
    description="HMMs models, in LUA format",
    short="m",
    argument="yes",
  },
  {
    index_name="w",
    description="Whitespace/silence symbol",
    short="w",
    argument="yes",
  },
  {
    index_name="mark_before_w",
    description="Mark frontier before whitespace/silence symbol (default no)",
    long="mark-before-w",
    argument="yes",
  },
  {
    index_name="mark_after_w",
    description="Mark frontier before whitespace/silence symbol (default yes)",
    long="mark-after-w",
    argument="yes",
  },
  {
    index_name="s",
    description="Use sure frontier (default no)",
    short="s",
    argument="yes",
  },
  {
    index_name="d",
    description="Dest directory",
    short="d",
    argument="yes",
  },
  {
    index_name="f",
    description="Filter the transcription to generate phon transcription",
    short="f",
    argument="yes",
  },
  {
    description = "shows this help message",
    short = "h",
    long = "help",
    argument = "no",
    action = function (argument) 
	       print(cmd:generate_help()) 
	       os.exit(1)
	     end    
  }
}
optargs = cmd:parse_args()

local mark_before_w       = ((optargs.mark_before_w or "no") == "yes")
local mark_after_w        = ((optargs.mark_after_w or "yes") == "yes")
local vit_index_filename  = optargs.a or error ("Needs -a option!!!")
local phn_index_filename  = optargs.t or error ("Needs -t option!!!")
local dest_dir            = optargs.d or error ("Needs -d option!!!")
local hmms_filename       = optargs.m or error ("Needs -m option!!!")
local white_symbol        = optargs.w or error ("Needs -w option!!!")
local sure_frontier       = ((optargs.s or "no") == "yes")
local filter              = optargs.f
if filter then filter = dofile(filter)
else
  filter = function(str) return str end
end

if not mark_after_w and not mark_before_w then
  error("Is mandatory to mark frontiers before or/and after silences!!!")
end

m          = dofile(hmms_filename)
models     = m[1]
emiss2phon = {}
phon2emiss = {}
for name,model_info in pairs(models) do
  phon2emiss[name] = {}
  for _,e in ipairs(model_info.emissions) do
    emiss2phon[e] = name
    phon2emiss[name][e] = true
  end
end

local vit_file = io.open(vit_index_filename)
local phn_file = io.open(phn_index_filename)

for orig_filename in vit_file:lines() do
  collectgarbage("collect")

  local destname = dest_dir .. "/" .. string.basename(orig_filename)
  destname = string.gsub(destname, "%.mat.*$", ".ose")
  fprintf(io.stderr,"# Generando desde \t%s\ta\t%s\n", orig_filename,destname)
  
  local orig_phn_filename = phn_file:read("*l")
  local phn               = filter(io.open(orig_phn_filename):read("*l"))
  local mat               = matrix.fromFilename(orig_filename)
  local ds                = dataset.matrix(mat)
  phn                     = string.tokenize(phn)
  local words             = { }
  for i=1,#phn do
    if phn[i] == white_symbol then
      if phn[i+1] and phn[i+1] ~= white_symbol then
	table.insert(words, {})
      end
    else
      if #words == 0 then table.insert(words, {}) end
      table.insert(words[#words], phn[i])
    end
  end
  -- words es una tabla que contiene tablas con los fonemas de cada
  -- palabra, eliminando los de silencio
  
  local current_frame = 1
  local current_word  = 1
  local current_phone = 1
  
  -- primera frontera, se marca siempre
  local frontiers    = { util.ose.compose_frontier{ util.ose.generate_output } }
  local last_value   = -100
  if ds:patternSize() ~= 1 then error ("Incorrect segmentation file") end
  function add_void_frontier()
    table.insert(frontiers, util.ose.compose_frontier{})
  end
  function add_frontier()
    if sure_frontier then
      table.insert(frontiers,
		   util.ose.compose_frontier{ util.ose.mark,
					      util.ose.generate_output,
					      util.ose.remove_marked } )
    else
      table.insert(frontiers,
		   util.ose.compose_frontier{ util.ose.generate_output })
    end
  end
  function skip_silence()
    local skipped = false
    -- buscamos el primer frame que no es espacio
    while (current_frame < ds:numPatterns() and
	 emiss2phon[ds:getPattern(current_frame)[1]] == white_symbol) do
      --print(current_frame, #frontiers)
      -- metemos la frontera despues del frame actual
      add_void_frontier()
      -- pasamos al siguiente
      current_frame = current_frame + 1
      skipped = true
    end
    return skipped
  end
  -- buscamos el primer frame que no es espacio
  skip_silence()
  if current_frame ~= 1 then
    -- current_frame apunta al primer frame diferente de silencio,
    -- eliminamos el ultimo y anyadimos frontera
    table.remove(frontiers)
    if mark_after_w then
      add_frontier()
    else
      add_void_frontier()
    end
  else current_frame = 1
  end

  -- a partir de este momento frontiers va siempre un frame por detras
  while current_frame < ds:numPatterns() do

    local last_e = -100
    while (current_frame < ds:numPatterns() and
	 emiss2phon[ds:getPattern(current_frame)[1]] == words[current_word][current_phone] and
       last_e <= ds:getPattern(current_frame)[1]) do
      -- anyadimos la frontera vacia del frame anterior al actual
      add_void_frontier()
      last_e = ds:getPattern(current_frame)[1]
      -- pasamos al siguiente
      current_frame = current_frame + 1
      --print("O", current_frame, #frontiers)
    end
    
    -- anyadimos porque vamos retrasados un frame, por tanto esta va
    -- antes del simbolo
    local marked_before = false
    if mark_before_w and emiss2phon[ds:getPattern(current_frame)[1]] == white_symbol then
      add_frontier()
      marked_before = true
    end    
    
    -- se ha producido el cambio de fonema, evitamos el silencio
    local skipped_silence = skip_silence()
    if skipped_silence and mark_before_w then
      -- eliminamos la ultima, para que vuelva a ir un frame por
      -- detras
      table.remove(frontiers)
    end    

    --print("O", current_frame, #frontiers)
    
    -- pasamos al siguiente fonema
    current_phone = current_phone + 1
    if current_phone > #words[current_word] then
      current_word  = current_word + 1
      current_phone = 1
      -- cambio de palabra, metemos una frontera
      if current_frame < ds:numPatterns() then
	if mark_after_w or not marked_before then
	  add_frontier()
	else
	  add_void_frontier()
	end
      else
	-- ultima frontera, se marca siempre
	if skipped_silence then
	  table.insert(frontiers, util.ose.compose_frontier{ util.ose.generate_output })
	else
	  add_frontier()
	end
      end
      -- corremos un frame
      current_frame = current_frame + 1
    end
  end
  if current_word ~= #words+1 then error ("Incorrect words number") end
  if #frontiers ~= ds:numPatterns() + 1 then
    error ("Incorrect number of frontiers: " .. #frontiers .. " instead of " .. ds:numPatterns()+1)
  end
  local f = io.open(destname, "w")
  if not f then error("File not found!!! " .. destname) end
  f:write(table.concat(frontiers, ""))
  f:close()
end
