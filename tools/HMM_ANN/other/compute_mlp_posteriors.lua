dir = string.gsub(arg[0], string.basename(arg[0]), "")
dir = (dir~="" and dir) or "./"
loadfile(dir .. "utils.lua")()

cmdOptTest = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "Generate the MLP posterior matrix",
  { index_name="n", -- antes filenet
    description = "MLP file",
    short    = "n",
    argument = "yes",
  },
  { index_name="p", -- antes testfile
    description = "File with the corpus mfccs data",
    short = "p",
    argument = "yes",
  },
  {
    index_name  = "context", -- antes ann_context_size
    description = "Size of ann context [default=4]",
    long        = "context",
    argument    = "yes"
  },
  {
    index_name  = "feats_format",
    description = "Format of features mat or mfc [default mat]",
    long        = "feats-format",
    argument    = "yes",
  },
  {
    index_name  = "feats_norm",
    description = "Table with means and devs for features [default nil]",
    long        = "feats-norm",
    argument    = "yes",
  },
  {
    index_name  = "step",
    description = "Dataset step [default 1]",
    long        = "step",
    argument    = "yes",
  },
  {
    index_name = "dir",
    description = "Output dir",
    long = "dir",
    argument="yes",
  },
  {
    index_name = "cores",
    description = "Number of cores [default = 2]",
    long = "cores",
    argument="yes",
  },
  {
    description = "shows this help message",
    short = "h",
    long = "help",
    argument = "no",
    action = function (argument) 
	       print(cmdOptTest:generate_help()) 
	       os.exit(1)
	     end    
  }
}

optargs = cmdOptTest:parse_args()

if type(optargs) == "string" then error(optargs) end

if optargs.defopt then
  local t = dofile(optargs.defopt)
  for name,value in pairs(t) do
    if not optargs[name] then
      fprintf(io.stderr,"# opt %s = %s\n", name, tostring(value))
      optargs[name] = value
    end
  end
end

filenet     = optargs.n -- Red neuronal
valfile     = optargs.p or error ("Needs a list of MFCCs")
dir         = optargs.dir or "initial_segmentation"
context     = tonumber(optargs.context or 4)
step        = tonumber(optargs.step or 1)
format      = optargs.feats_format or "mat"
cores       = tonumber(optargs.cores or 2)

feats_mean_and_devs = optargs.feats_norm
if feats_mean_and_devs then
  feats_mean_and_devs = dofile(feats_mean_and_devs)
end

if not filenet then
  error ("Needs a MLP")
end

lared = Mlp.load{ filename = filenet }
func  = lared

--------------------
-- parametros RNA --
--------------------
ann = {}
ann.left_context           = context
ann.right_context          = context

num_emissions = func:get_output_size()
collectgarbage("collect")

local mfcc_f = io.open(valfile)
local list = {}
for mfcc_filename in mfcc_f:lines() do
  table.insert(list, mfcc_filename)
end

which_i_am,child_pid = util.split_process(cores)

for index=which_i_am,#list,cores do
  mfcc_filename = list[index]
  collectgarbage("collect")
  -- cargamos el dataset correspondiente a la frase actual
  print ("# Cargando frames:        ", mfcc_filename, index .. "/" .. #list)
  local frames
  if format == "mat" then
    frames = load_matrix(mfcc_filename)
  else
    frames = load_mfcc(mfcc_filename)
  end
  local numFrames   = frames:dim()[1]
  local numParams   = frames:dim()[2] -- nCCs+1
  local parameters = {
    patternSize = {step, numParams},
    offset      = {0,0},  -- default value
    stepSize    = {step, 0}, -- default value, second value is not important
    numSteps    = {numFrames/step, 1}
  }
  local actual_ds = dataset.matrix(frames, parameters)
  if feats_mean_and_devs then
    actual_ds:normalize_mean_deviation(feats_mean_and_devs.means,
				       feats_mean_and_devs.devs)
  end
  actual_ds = dataset.contextualizer(actual_ds,
				     ann.left_context,
				     ann.right_context)
  
  local segmentation_matrix = matrix(numFrames)
  local mat_full = matrix(numFrames, num_emissions)
  local mat_full_ds = dataset.matrix(mat_full)
  func:use_dataset{
    input_dataset  = actual_ds,   -- parametrizacion
    output_dataset = mat_full_ds        -- matriz de emisiones
  }
  local bname = string.remove_extension(string.basename(mfcc_filename))
  if string.match(mfcc_filename, "%.gz") then
    bname = string.remove_extension(bname)
  end
  local outfile = string.format("%s/%s.mat.gz", dir, bname)
  print ("# Guardando MLP output:   ", outfile)
  matrix.savefile(mat_full, outfile, "ascii")
end

if child_pid then
  -- esperamos a los hijos
  util.wait()
end
