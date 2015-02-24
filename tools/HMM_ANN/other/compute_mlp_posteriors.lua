april_print_script_header(arg)
dofile(string.get_path(arg[0]) .. "../utils.lua")

cmdOptTest = cmdOpt{
  program_name = string.basename(arg[0]),
  argument_description = "",
  main_description = "Generate the MLP posterior matrix",
  { index_name="n",
    description = "MLP file",
    short    = "n",
    argument = "yes",
    mode     = "always",
  },
  { index_name="p",
    description = "File with the corpus mfccs data",
    short       = "p",
    argument    = "yes",
    mode        = "always",
  },
  {
    index_name  = "context", -- antes ann_context_size
    description = "Size of ann context",
    long        = "context",
    argument    = "yes",
    mode        = "always",
    default_value=4,
    filter=tonumber,
  },
  {
    index_name  = "feats_format",
    description = "Format of features mat, mfc or png",
    long        = "feats-format",
    argument    = "yes",
    mode        = "always",
    default_value="mat",
  },
  {
    index_name  = "feats_norm",
    description = "Table with means and devs for features",
    long        = "feats-norm",
    argument    = "yes",
    filter      = dofile,
  },
  {
    index_name  = "step",
    description = "Dataset step",
    long        = "step",
    argument    = "yes",
    mode        = "always",
    default_value="1",
    filter=tonumber,
  },
  {
    index_name = "dir",
    description = "Output dir",
    long = "dir",
    argument="yes",
    mode="always",
    default_value="posteriors",
  },
  { index_name="f",
    description = "Force overwritten output files",
    short    = "f",
    argument = "no",
  },
  {
    index_name = "cores",
    description = "Number of cores (processes) to use",
    long = "cores",
    argument="yes",
    mode="always",
    default_value=util.omp_get_num_threads(),
    filter=tonumber,
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

local optargs = cmdOptTest:parse_args()
if type(optargs) == "string" then error(optargs) end

local filenet     = optargs.n
local valfile     = optargs.p
local dir         = optargs.dir
local context     = optargs.context
local step        = optargs.step
local format      = optargs.feats_format
local cores       = optargs.cores
local feats_mean_and_devs = optargs.feats_norm
local force_write = optargs.f

os.execute("mkdir -p " .. dir)

--
local trainer       = trainable.supervised_trainer.load(filenet, nil, 128)
local num_emissions = trainer:get_output_size()
local frames_loader = nil
if format == "mat" then
  frames_loader = load_matrix
elseif format == "mfc" then
  frames_loader = load_mfcc
elseif format == "png" then
  frames_loader = load_png
else
  error("Unknown format " .. format)
end

-----------------------------------------------------------------------------

local feats_f = io.open(valfile)
local list = {}
for feats_filename in feats_f:lines() do
  table.insert(list, feats_filename)
end

local which_i_am,child_pid = util.split_process(cores)

for index=which_i_am,#list,cores do
  collectgarbage("collect")
  local feats_filename = list[index]
  print ("# Loading frames:      ", feats_filename, index .. "/" .. #list)
  local frames      = frames_loader(feats_filename)
  local numFrames   = math.floor(frames:dim(1)/step)
  local numParams   = frames:dim(2)
  local parameters = {
    patternSize = {1, numParams},
    stepSize    = {step, 0},
    numSteps    = {numFrames, 1}
  }
  local current_ds = dataset.matrix(frames, parameters)
  if feats_mean_and_devs then
    current_ds:normalize_mean_deviation(feats_mean_and_devs.means,
					feats_mean_and_devs.devs)
  end
  current_ds = dataset.contextualizer(current_ds, context, context)
  local output_mat = matrix(numFrames, num_emissions)
  local output_ds  = dataset.matrix(output_mat)
  trainer:use_dataset{
    input_dataset  = current_ds,
    output_dataset = output_ds,
  }
  local base_name = string.remove_extension(string.basename(feats_filename))
  if string.match(feats_filename, "%.gz") then
    base_name = string.remove_extension(base_name)
  end
  local outfile = string.format("%s/%s.mat.gz", dir, base_name)
  print ("# Saving MLP output:   ", outfile)
  if io.open(outfile) and not force_write then
    error(string.format("# Output file '%s' exists, use -f to force overwritten\n",
			outfile))
  else
    output_mat:toFilename(outfile, "ascii")
  end
end

if child_pid then
  -- esperamos a los hijos
  util.wait()
end
