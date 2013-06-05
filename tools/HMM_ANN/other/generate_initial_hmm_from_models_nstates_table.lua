if #arg ~= 2 then
  fprintf(io.stderr, "SINTAXIS: %s  TIEDLIST  NSTATESFILE\n", arg[0])
  os.exit(128)
end

tiedfile      = arg[1]
nstates_file  = arg[2]

-- objeto con informacion sobre modelos ligados
tied = tied_model_manager(io.open(tiedfile))

local model2nstates = {}
for line in io.lines(nstates_file) do
  local model_name, nstates = unpack(string.tokenize(line))
  name = tied:get_model(model_name)
  if model2nstates[name] and
    model2nstates[name] ~= tonumber(nstates) then
    error ("Two tied models has different number of states :S " .. model_name .. " " .. name)
  end
  model2nstates[name]       = tonumber(nstates)
  model2nstates[model_name] = tonumber(nstates)
end

local num_models    = 0
local hmms          = {}
local next_emission = 1
local trainer       = HMMTrainer.trainer()
for model_id,name in ipairs(tied.id2name) do
  m = tied.tiedlist[name]
  if name == m then
    local num_states = model2nstates[name]
    if not num_states then
      error ("Inexistent model: " .. name)
    end
    local ploops = {}
    for i=1,num_states do ploops[i] = 0.5 end
    num_models = num_models + 1
    local hmm_emissions = {}
    -- generamos el vector de emisiones
    for i=1,num_states do
      hmm_emissions[i] = next_emission
      next_emission    = next_emission + 1
    end
    -- este es el representante
    local desc = HMMTrainer.utils.generate_lr_hmm_desc(name,
						       hmm_emissions,
						       ploops, {},
						       name)
    local model = trainer:model(desc)
    hmms[name] = {
      model     = model,
      emissions = hmm_emissions,
    }
  end
end

print("return { {")
for name,model_info in pairs(hmms) do
  print("['".. string.gsub(name,"'","\\'") .. "'] = {")
  print("\tmodel=HMMTrainer.model.from_table(")
  printf("%s", model_info.model:to_string())
  print("),")
  print("\temissions={".. table.concat(model_info.emissions,
				       ",") .."}")
  print("},")
end
print("},")
print("{")
num_emissions = next_emission - 1
for i=1,num_emissions do
  printf("%g, ", 1/num_emissions)
end
print("},")
print("}")
