get_table_from_dotted_string("ann.autoencoders", true)

----------------------------------------------------------------------

local md = matrix.dict

-- The auto-encoder class (AE) will be a denoising auto-encoder (DAE) when
-- trained with corrupted input, and clean output

local ann_autoencoders_ae,ann_autoencoders_ae_methods =
  class("ann.autoencoders.ae", ann.components.base)
ann.autoencoders.ae = ann_autoencoders_ae

function ann_autoencoders_ae:constructor(t)
  local params = get_table_fields(
    {
      name   = { mandatory=false, type_match="string" },
      index  = { mandatory=true,  type_match="number", default=1 },
      hidden = { mandatory=true,  type_match="table",
		 getter=get_table_fields_recursive{
		   actf = { mandatory=true, type_match="string" },
		   size = { mandatory=true, type_match="number" },
       params = {mandatory=false, type_match="table", default = {}}
		 }, },
      visible = { mandatory=true,  type_match="table",
		  getter=get_table_fields_recursive{
		    actf = { mandatory=true, type_match="string" },
		    size = { mandatory=true, type_match="number" },
        params = {mandatory=false, type_match="table", default = {}}
		  }, },
      encoder = { mandatory=false, isa_match=ann.components.base },
      decoder = { mandatory=false, isa_match=ann.components.base },
    }, t)
  
  assert( ( params.encoder and params.decoder) or
	    ( (not params.encoder) and (not params.decoder) ),
	  "Encoder and decoder must be given together, or not given both")
  
  local name  = params.name or "AE-" .. params.index
  local ae    = ann.components.stack{ name=name }
  
  -- this methods must be defined here because
  -- they overwrite methods of
  -- ann.components.stack
  
  -- the clone function produces a new ae object
  self.clone = function(self)
    local params = self.params
    local obj = ann.autoencoders.ae{
      name    = params.name,
      index   = params.index,
      hidden  = table.deep_copy(params.hidden),
      visible = table.deep_copy(params.visible),
      encoder = (params.encoder and params.encoder:clone()) or nil,
      decoder = (params.decoder and params.decoder:clone()) or nil,
    }
    return obj
  end
  -- serialization methods
  self.ctor_name = function(self)
    return "ann.autoencoders.ae(%s)"
  end
  self.ctor_params = function(self)
    return self.params
  end
  -- the build method 
  self.build = function(self,...)
    local ae      = self.ae
    local result  = table.pack(ae:build(...))
    local encoder = self.encoder
    local decoder = self.decoder
    assert(encoder:get_output_size() == decoder:get_input_size() or
             encoder:get_output_size() == 0 or
             decoder:get_input_size() == 0,
           "Output size of encoder must be equals to input size of decoder")
    return table.unpack(result)
  end
  local obj = class_wrapper(ae,self)
  assert(rawequal(obj, self))
  
  local encoder = params.encoder
  if not encoder then
    encoder = ann.components.stack{ name="enc-" .. params.index }:
    push( ann.components.hyperplane{ name="enc-plane-" .. params.index,
				     input=params.visible.size,
				     output=params.hidden.size,
				     dot_product_name="w-enc" .. params.index,
				     bias_name="b-enc" .. params.index,
				     dot_product_weights="w" .. params.index,
				     bias_weights="b-enc" .. params.index,
				   } ):
    push( ann.components.actf[params.hidden.actf](table.merge({ name="h-enc-" .. params.index}, params.hidden.params)))
  end
  
  local decoder = params.decoder
  if not decoder then
    decoder = ann.components.stack{ name="dec-" .. params.index }:
    push( ann.components.hyperplane{ name="dec-plane-" .. params.index,
				     input=params.hidden.size,
				     output=params.visible.size,
				     dot_product_name="w-dec" .. params.index,
				     bias_name="b-dec" .. params.index,
				     dot_product_weights="w" .. params.index,
				     bias_weights="b-dec" .. params.index,
				     transpose=true
				   } ):
    push( ann.components.actf[params.visible.actf](table.merge({ name="v-enc-" .. params.index}, params.hidden.params)))
  end
  if class.is_a(encoder, ann.components.stack) then
    ae:push(encoder:unroll())
  else
    ae:push(encoder)
  end
  if class.is_a(decoder, ann.components.stack) then
    ae:push(decoder:unroll())
  else
    ae:push(decoder)
  end
  
  self.encoder = encoder
  self.decoder = decoder
  self.ae      = ae
  self.params  = params
  
end

function ann_autoencoders_ae_methods:get_C_AE_component()
  return self.ae
end

function ann_autoencoders_ae_methods:get_C_encoder_component()
  return self.encoder
end 

function ann_autoencoders_ae_methods:get_C_decoder_component()
  return self.decoder
end 

function ann_autoencoders_ae_methods:encode(...)
  return self.encoder:forward(...)
end

function ann_autoencoders_ae_methods:decode(...)
  return self.decoder:forward(...)
end

function ann_autoencoders_ae_methods:get_visible_param()
  return self.params.visible
end

function ann_autoencoders_ae_methods:get_hidden_param()
  return self.params.hidden
end

----------------------------------------------------------------------

april_set_doc(ann.autoencoders,
	      {
		class="namespace",
		summary={"Namespace with utilties for easy training",
			 "of Stacked Denoising Auto-Encoders (SDAEs).",
			 "This namespace contains functions which works",
			 "with ann.components instances, and others with",
			 "a table with weight matrixes and bias vectors.", },
	      })

----------------------------------------------------------------------

-- AUXILIAR LOCAL FUNCTIONS --

-- This function builds a codifier from the weights of the first layer of a
-- restricted autoencoder
local function build_two_layered_codifier(names_prefix,
					  input_size,
					  input_actf,
            input_params,
					  cod_size,
					  cod_actf,
            cod_params)
  local codifier_component = ann.components.stack{ name=names_prefix.."stack" }
  codifier_component:push( ann.components.hyperplane{
			     name                = names_prefix.."c",
			     dot_product_name    = names_prefix.."w",
			     dot_product_weights = names_prefix.."w",
			     bias_name           = names_prefix.."b",
			     bias_weights        = names_prefix.."b",
			     input  = input_size,
			     output = cod_size, } )
  codifier_component:push(ann.components.actf[cod_actf](table.merge({ name=names_prefix.."actf" }, cod_params)))
  return codifier_component
end

-- This function builds an autoencoder of two layers (input-hidden-output) where
-- the hidden-output uses the same weights as input-hidden, so it is
-- simetric.
local function build_two_layered_autoencoder_from_sizes_and_actf(names_prefix,
								 input_size,
								 input_actf,
                 input_params,
								 cod_size,
								 cod_actf,
                 cod_params,
								 weights_random)

  for i, v in pairs(cod_params) do
      print(i,v)
  end
  for i, v in pairs(input_params) do
      print(i,v)
  end
  

  local autoencoder_component = ann.components.stack{ name=names_prefix.."stack" }
  autoencoder_component:push( ann.components.hyperplane{
				name                = names_prefix.."layer1",
				dot_product_name    = names_prefix.."w1",
				dot_product_weights = names_prefix.."w",
				bias_name           = names_prefix.."b1",
				bias_weights        = names_prefix.."b1",
				input  = input_size,
				output = cod_size, } )
  autoencoder_component:push(ann.components.actf[cod_actf](table.merge({ name=names_prefix.."actf1" }, cod_params)))
  autoencoder_component:push( ann.components.hyperplane{
				name                = names_prefix.."layer2",
				dot_product_name    = names_prefix.."w2",
				dot_product_weights = names_prefix.."w",
				bias_name           = names_prefix.."b2",
				bias_weights        = names_prefix.."b2",
				input  = cod_size,
				output = input_size,
				transpose = true} )
  autoencoder_component:push(ann.components.actf[input_actf](table.merge({ name=names_prefix.."actf2" }, input_params)))
  local _,weights_table = autoencoder_component:build()
  for _,wname in ipairs({ names_prefix.."w",
			  names_prefix.."b1",
			  names_prefix.."b2" }) do
    ann.connections.
      randomize_weights(weights_table[wname],
                        {
                          random = weights_random,
                          inf    = -math.sqrt(6 / (input_size + cod_size)),
                          sup    =  math.sqrt(6 / (input_size + cod_size))
      })
  end
  return autoencoder_component
end

-- FAKE DATASET INDEXED
local fake_dataset_indexed, fake_indexed_methods =
  class("fake_dataset_indexed", datasetToken)

function fake_dataset_indexed:constructor(ds, dict)
  assert(class.is_a(ds,dataset), "The first argument must be a dataset")
  self.ds=ds
  self.dict=dict
  self.num_pats=ds:numPatterns()
  local pat_size = 0
  for i=1,#dict do
    pat_size = pat_size + dict[i]:patternSize()
    if class.is_a(dict[i], dataset) then
      self.dict[i] = dataset.token.wrapper(dict[i])
    end
  end
  self.pat_size = pat_size
end
function fake_indexed_methods:patternSize()
  return self.pat_size
end
function fake_indexed_methods:numPatterns()
  return self.num_pats
end
function fake_indexed_methods:getPattern(idx)
  if #self.dict == 1 then
    return self.dict[1]:getPattern(self.ds:getPattern(idx)[1])
  else
    local m = matrix(1, self.pat_size)
    local col_pos=1
    local index = self.ds:getPattern(idx)
    for i=1,#self.dict do
      local current_pat_size = self.dict[i]:patternSize()
      local current_token = self.dict[i]:getPattern(index[i])
      m:slice({1,col_pos},{1,current_pat_size}):copy(current_token)
      col_pos = col_pos + current_pat_size
    end
    return tokens.matrix(m)
  end
end
function fake_indexed_methods:getPatternBunch(idxs)
  if #self.dict == 1 then
    local idxs = table.imap(idxs,
			    function(idx) return self.ds:getPattern(idx)[1] end)
    return self.dict[1]:getPatternBunch(idxs)
  else
    error("NOT IMPLEMTENTED YET")
  end
end
--------------------------------------------------------------------------------

--auxiliar function to generate a replacement
function get_replacement_dataset(randObject, replacementSize, ...)
  local resul = {}
  local arg = table.pack(...)
  if #arg > 0 then
    local mat = matrix(replacementSize)
    local numPat = arg[1]:numPatterns()
    for i=1,replacementSize do
      mat:set(i,randObject:randInt(1,numPat))
    end
    local ds = dataset.matrix(mat)
    for i,v in ipairs(arg) do
      if v ~= nil then
	if v:numPatterns() ~= numPat then
	  error("Datasets have differnet number of patterns")
	end
	table.insert(resul,fake_dataset_indexed(ds,{v}))
      end
    end
  end
  return resul
end


-- Generate the data table for training a two-layered auto-encoder
local function generate_training_table_configuration(current_dataset_params,
						     replacement,
						     shuffle_random,
						     noise_pipeline,
						     output_datasets)
  local data = {}
  if current_dataset_params.input_dataset then
    if output_datasets and #output_datasets > 1 then
      error("Without distribution only one dataset is allowed "..
	      "at output_datasets table")
    end

    data.input_dataset  = current_dataset_params.input_dataset
    data.output_dataset = ((output_datasets or {})[1] or
			     current_dataset_params.input_dataset)
    -- The input is perturbed with noise
    for _,noise_builder in ipairs(noise_pipeline) do
      data.input_dataset = noise_builder(data.input_dataset)
    end
  end -- if params.input_dataset
  if current_dataset_params.distribution then
    data.distribution = {}
    for idx,v in ipairs(current_dataset_params.distribution) do
      local ds = v.input_dataset
      -- The input is perturbed with noise
      for _,noise_builder in ipairs(noise_pipeline) do
	data.input_dataset = noise_builder(data.input_dataset)
      end
      if output_datasets and not output_datasets[idx] then
	error("Incorrect number of output_datasets")
      end
      table.insert(data.distribution, {
		     input_dataset  = ds,
		     output_dataset = (output_datasets or {})[idx] or ds,
		     probability    = v.prob })
    end -- for _,v in ipairs(distribution)
  end -- if distribution
  data.shuffle     = shuffle_random
  data.replacement = replacement
  return data
end



local function
    generate_training_table_configuration_on_the_fly(current_dataset_params,
						     replacement,
						     shuffle_random,
						     noise_pipeline,
						     mlp_final_trainer,
						     output_datasets)
  return function()
    -- Take the original datasets (input, and output)
    local input_dataset  = current_dataset_params.input_dataset
    local output_dataset = (output_datasets or {})[1]
    -- Generate a replacement of each dataset
    local input_repl_ds, output_repl_ds

    if replacement then
      -- if autoencoder, only generate one corpus replacement
      input_repl_ds, output_repl_ds = table.unpack( get_replacement_dataset(shuffle_random, replacement, input_dataset, output_dataset) )
    end
    -- generate the last layer dataset
    local aux_input = input_repl_ds or input_dataset
    local input_layer_dataset = dataset.matrix(matrix(aux_input:numPatterns(),
						      mlp_final_trainer:get_output_size()))
    mlp_final_trainer:use_dataset{
      input_dataset  = aux_input,
      output_dataset = input_layer_dataset,
    }
    -- The output is the same than the input
    output_dataset = (output_repl_ds or output_dataset or input_layer_dataset)
    -- Add the noise
    for _,noise_builder in ipairs(noise_pipeline) do
      input_dataset = noise_builder(input_layer_dataset)
    end
    return {
      input_dataset  = input_dataset,
      output_dataset = output_dataset,
      shuffle        = ( not replacement and shuffle_random ) or nil,
    }
  end
end

-- PUBLIC FUNCTIONS --

-- This functions receives layer sizes and sdae_table with weights and bias
-- arrays. It returns a fully connected stacked denoising autoencoder ANN.
ann.autoencoders.build_full_autoencoder =
  april_doc{
    class="function",
    summary="Function to build full SDAE (encoding and decoding) ",
    description=
      {
        "This function composes an ANN component",
        "from the layers table and the sdae_table",
        "build by other functions of this namespace. It builds",
        "a full auto-encoder, which means that it has the encoding",
        "part and the transposed decoding part of the auto-encoder.",
      },
    params= {
      { "A table with layers info, as this example:",
        "{ { size=..., actf=... }, { size=..., actf=...}, ... }", },
      "An sdae table, returned by other function of this namespace",
      { "A string prefix [optional], used as prefix of the",
        "component names." },
    },
    outputs= {
      {"A component object with the especified ",
       "neural network topology" },
    }
  } ..
  function(layers, sdae_table, names_prefix)
    local names_prefix  = names_prefix or ""
    local weights_mat   = sdae_table.weights
    local bias_mat      = sdae_table.bias
    local sdae          = ann.components.stack{ name=names_prefix.."stack" }
    local prev_size     = layers[1].size
    local weights_table = {}
    local k = 1
    for i=2,#layers do
      local size , actf   = layers[i].size,layers[i].actf
      local actf_params = layers[i].params or {}
      local wname , bname = names_prefix.."w" .. (i-1) , names_prefix.."b" .. k
      local actfname = names_prefix.."actf" .. k
      sdae:push( ann.components.hyperplane{
                   name                = names_prefix.."c"..k,
                   input               = prev_size,
                   output              = size,
                   dot_product_name    = wname,
                   dot_product_weights = wname,
                   bias_name           = bname,
                   bias_weights        = bname })
      
      sdae:push( ann.components.actf[actf](table.merge({name=actfname}, actf_params)))
      --
      weights_table[wname] = ann.connections{ input=prev_size,
                                              output=size,
                                              w=weights_mat[i-1] }
      weights_table[bname] = ann.connections{ input=1,
                                              output=size,
                                              w=bias_mat[i-1][1] }
      prev_size = size
      k = k+1
    end
    for i=#layers-1,1,-1 do
      local size , actf   = layers[i].size, layers[i].actf
      local actf_params = layers[i].params or {}
      local wname , bname = names_prefix.."w" .. i , names_prefix.."b" .. k
      local dname = names_prefix.."w" .. k
      local actfname = names_prefix.."actf" .. k
      sdae:push( ann.components.hyperplane{
                   name                = names_prefix.."c"..k,
                   input               = prev_size,
                   output              = size,
                   transpose           = true,
                   dot_product_weights = wname,
                   dot_product_name    = dname,
                   bias_name           = bname,
                   bias_weights        = bname })

      sdae:push( ann.components.actf[actf](table.merge( {name=actfname}, actf_params)))
      --
      weights_table[bname] = ann.connections{ input=1,
                                              output=size,
                                              w=bias_mat[i][2] }
      prev_size = size
      k = k+1
    end
    sdae:build{ weights=weights_table }
    return sdae
  end

-- Params is a table which could contain:
--   * input_dataset => dataset with input (and output) for AE
--   * distribution => a table which contains a list of {input_dataset=...., prob=....}
--   * replacement => replacement value for training
--   * shuffle_random => random number generator
--   * weights_random => random number generator
--   * perturbation_random => random number generator
--   * var => variance of gaussian noise
--   * layers => table which contains a list of { size=...., actf=....}, being
--               size a number and actf a string = "logistic"|"tanh"|"linear"
--   * supervised_layer => size and actf
--   * output_datasets => a table with output datasets
--   * bunch_size => size of mini-batch
--   * learning_rate
--   * momentum
--   * weight_decay
--   * neuron_squared_length_upper_bound
--   * max_epochs
--   * min_epochs
--   * stopping_criterion => function
--   * pretraining_percentage_stopping_criterion
--
-- This function returns a Stacked Denoising Auto-Encoder parameters table,
-- pretrained following algorithm of:
--
-- [CITE]
--
-- If you train an auto-encoder for a topology of 256 128 64
-- the WHOLE auto-encoder will had this topology:
-- 256 - 128 - 64 - 128 - 256
-- So it has four layers: (1) 256-128, (2) 128-64, (3) 64-128, (4) 128-256
--
-- Two arrays store weights and bias, in this order:
-- bias[1] => 128      bias of layer (1)
-- bias[2] =>  64      bias of layer (2)
-- bias[3] => 128      bias of layer (3)
-- bias[4] => 256      bias of layer (4)
-- weights[1] => 256*128  weights of layer (1)
-- weights[2] => 128*64   weights of layer (2)
ann.autoencoders.greedy_layerwise_pretraining =
  april_doc{
    class="function",
    summary={"MAIN function of this namespace, which implements",
             "the SDAE training", },
    description=
      {
        "This function builds a SDAE following the greedy",
        "layerwise algorithm described at",
        "http://deeplearning.net/tutorial/SdA.html",
        "The function receives a table with a lot of parameters",
        "for training description."
      },
    params= {
      ["optimizer"]   = {"A function which returns an optimizer",
                         "[optional]",},
      ["names_prefix"]   = {"A prefix added to all component names",
                            "[optional]",},
      ["shuffle_random"] = "A random object instance for shuffle",
      ["weights_random"] = {"A random object instance for weights",
                            "initialization"},
      ["layers"] =
        {"A table describing the layers of the SDAE:",
         "{ { size=NUMBER, actf=ACTF_STRING }, ... }.",
         "This table has the sizes of INPUT LAYER, FIRST HIDDEN",
         "LAYER, SECOND HIDDEN LAYER, ..., LAST HIDDEN LAYER",},
      ["bunch_size"] = "A number with the value of the bunch",
      ["training_options"] = {
        "It is a table with options and hyperparameters related to",
        "the training. The table has two main fields, global and",
        "layerwise. Layerwise table is an array of options, so",
        "any option at global table could be overwritten by",
        "layerwise option. Each position of layerwise array is",
        "related to each iteration of the greedy algorithm.",
        "Please, read the wiki for detailed information.",
      },
      ["input_dataset"] = {
        "It is the dataset with input data",
      },
      ["output_datasets"] = {
        "An array with one dataset, the output data [optional]",
      },
      ["replacement"] = "Replacement during training [optional]",
      ["supervised_layer"] = {
        "A table with { size=NUMBER, actf=ACTF_STRING } [optional]",
      },
      ["on_the_fly"] = {
        "A boolean, for large datasets, the greedy algorithm must",
        "compute",
        "encodings on-the-fly, in order to reduce the memory",
        "fingerprint [optional]",
      },
    },
    outputs= {
      {"The sdae_table = { weights = ARRAY OF MATRIXES,",
       "bias = ARRAY OF MATRIXES }",},
      {"An instance of ann.components.stack() with encoding",
       "part of SDAE (or a deep classifier if given",
       "output_datsets" },
    }
  } ..
  april_doc{
    class="function",
    summary={"MAIN function of this namespace, which implements",
             "the SDAE training", },
    description=
      {
        "This function builds a SDAE following the greedy",
        "layerwise algorithm described at",
        "http://deeplearning.net/tutorial/SdA.html",
        "The function receives a table with a lot of parameters",
        "for training description."
      },
    params= {
      ["optimizer"]   = {"A function which returns an optimizer",
                         "[optional]",},
      ["names_prefix"]   = {"A prefix added to all component names",
                            "[optional]",},
      ["shuffle_random"] = "A random object instance for shuffle",
      ["weights_random"] = {"A random object instance for weights",
                            "initialization"},
      ["layers"] =
        {"A table describing the layers of the SDAE:",
         "{ { size=NUMBER, actf=ACTF_STRING }, ... }.",
         "This table has the sizes of INPUT LAYER, FIRST HIDDEN",
         "LAYER, SECOND HIDDEN LAYER, ..., LAST HIDDEN LAYER",},
      ["bunch_size"] = "A number with the value of the bunch",
      ["training_options"] = {
        "It is a table with options and hyperparameters related to",
        "the training. The table has two main fields, global and",
        "layerwise. Layerwise table is an array of options, so",
        "any option at global table could be overwritten by",
        "layerwise option. Each position of layerwise array is",
        "related to each iteration of the greedy algorithm.",
        "Please, read the wiki for detailed information.",
      },
      ["distribution"] = {
        "A distribution table as for",
        "trainable.supervised_trainer.train_dataset.",
      },
      ["output_datasets"] = {
        "An array with one dataset, the output data [optional]",
      },
      ["replacement"] = "Replacement during training [optional]",
      ["supervised_layer"] = {
        "A table with { size=NUMBER, actf=ACTF_STRING } [optional]",
      },
      ["on_the_fly"] = {
        "A boolean, for large datasets, the greedy algorithm must",
        "compute",
        "encodings on-the-fly, in order to reduce the memory",
        "fingerprint [optional]",
      },
    },
    outputs= {
      {"The sdae_table = { weights = ARRAY OF MATRIXES,",
       "bias = ARRAY OF MATRIXES }",},
      {"An instance of ann.components.stack() with encoding",
       "part of SDAE (or a deep classifier if given",
       "output_datsets" },
    }
  }..
  function(t)
    local params = get_table_fields(
      {
        optimizer        = { mandatory=false, type_match="function",
                             default = function() return nil end},
        names_prefix     = { mandatory=false, default="", type_match="string" },
        shuffle_random   = { mandatory=true,  isa_match=random },
        weights_random   = { mandatory=true,  isa_match=random },
        layers           = { mandatory=true, type_match="table",
                             getter=get_table_fields_ipairs{
                               actf = { mandatory=true, type_match="string" },
                               size = { mandatory=true, type_match="number" },
                               params = { mandatory=false, type_match="table", default = {} },
                             }, },
        bunch_size       = { mandatory=true, type_match="number", },
        training_options = { mandatory=true },

        --      , type_match="table",
        --			   getter=get_table_fields_recursive{
        --			     global = { mandatory=true, type_match="table",
        --					
        --			   } },
        --
        
        input_dataset    = { mandatory=false },
        output_datasets  = { mandatory=false, type_match="table", default=nil },
        distribution     = { mandatory=false, type_match="table", default=nil,
                             getter=get_table_fields_ipairs{
                               input_dataset   = { mandatory=true },
                               output_datasets = { mandatory=true },
                               probability     = { mandatory=true },
                             }, },
        replacement      = { mandatory=false, type_match="number", default=nil },
        supervised_layer = { mandatory=false, type_match="table", default=nil,
                             getter=get_table_fields_recursive{
                               actf = { mandatory=true, type_match="string" },
                               size = { mandatory=true, type_match="number" },
                             }, },
        on_the_fly        = { mandatory=false, default=false, type_match="boolean" },
      }, t)
    -- Error checking in params table --
    if params.input_dataset and params.distribution then
      error("The input_dataset and distribution parameters are forbidden together")
    end
    if params.distribution and not params.replacement then
      error("The replacement parameter is mandatary if distribution")
    end
    if (params.supervised_layer~=nil) == (not params.output_datasets) then
      error("Params output_datasets and "..
              "supervised_layer must be present together")
    end
    params.training_options.global    = params.training_options.global    or { ann_options = {} }
    params.training_options.layerwise = params.training_options.layerwise or {}
    params.training_options.global.ann_options = params.training_options.global.ann_options or {}
    --------------------------------------
    -- on the fly. Do not generate all the dataset for each layer
    local on_the_fly = params.on_the_fly
    if params.replacement and not params.on_the_fly then
      fprintf(io.stderr,
              "WARNING!!!! THE REPLACEMENT PARAMETER IS NOT FORCING "..
                "ON_THE_FLY, THEY ARE INDEPENDENT\n")
    end
    if on_the_fly and params.distribution then
      error("On the fly mode is not working with dataset distribution")
    end
    -- copy dataset params to auxiliar table
    local current_dataset_params = {
      input_dataset = params.input_dataset,
      distribution  = params.distribution
    }
    -- output weights and bias matrices
    local weights = {}
    local bias    = {}
    -- incremental mlp
    local mlp_final_weights = {}
    local mlp_final = ann.components.stack{ name=params.names_prefix.."stack" }
    -- loop for each pair of layers
    for i=2,#params.layers do
      local input_size = params.layers[i-1].size
      local cod_size   = params.layers[i].size
      printf("# Training of layer %d--%d--%d (number %d)\n",
             input_size, cod_size, input_size, i-1)
      io.stdout:flush()
      local input_actf = params.layers[i-1].actf
      if input_actf == "logistic" then input_actf = "log_logistic"
      elseif input_actf == "softmax" then input_actf = "log_softmax"
      end
      local input_params = params.layers[i-1].params
      local cod_actf   = params.layers[i].actf
      local cod_params = params.layers[i].params

      local global_options    = table.deep_copy(params.training_options.global)
      local layerwise_options = params.training_options.layerwise[i-1] or {}
      layerwise_options.ann_options = layerwise_options.ann_options or {}
      local lookup = function(name) return layerwise_options[name] or global_options[name] end
      local data
      if not on_the_fly or i == 2 then
        data = generate_training_table_configuration(current_dataset_params,
                                                     params.replacement,
                                                     params.shuffle_random,
                                                     lookup("noise_pipeline") or {},
                                                     nil)
      else
        local mlp_final_trainer = trainable.supervised_trainer(mlp_final:clone(),
                                                               nil,
                                                               params.bunch_size,
                                                               params.optimizer())

        local aux_weights = md.clone( mlp_final_weights )
        mlp_final_trainer:build{ weights=aux_weights }
        data = generate_training_table_configuration_on_the_fly(current_dataset_params,
                                                                params.replacement,
                                                                params.shuffle_random,
                                                                lookup("noise_pipeline") or {},
                                                                mlp_final_trainer,
                                                                nil)
      end
      local dae
      dae = build_two_layered_autoencoder_from_sizes_and_actf(params.names_prefix,
                                                              input_size,
                                                              input_actf,
                                                              input_params,
                                                              cod_size,
                                                              cod_actf,
                                                              cod_params,
                                                              params.weights_random)
      collectgarbage("collect")
      local loss_function
      if input_actf == "log_logistic" then
        loss_function = ann.loss.cross_entropy(input_size)
      elseif input_actf == "log_softmax" then
        loss_function = ann.loss.multi_class_cross_entropy(input_size)
      else
        loss_function = ann.loss.mse(input_size)
      end
      ---------- TRAIN THE AUTOENCODER WO VALIDATION ----------
      local trainer = trainable.supervised_trainer(dae, loss_function,
                                                   params.bunch_size,
                                                   params.optimizer())
      trainer:build()
      for key,value in pairs(global_options.ann_options) do
        if layerwise_options.ann_options[key] == nil then
          trainer:set_option(key, value)
        end
      end
      for key,value in pairs(layerwise_options.ann_options) do
        trainer:set_option(key, value)
      end
      for _,reg in ipairs({ "weight_decay", "L1_norm", "max_norm_penalty" }) do
        if trainer:has_option(reg) then
          trainer:set_layerwise_option(params.names_prefix.."b.*", reg, 0.0)
        end
      end
      if input_actf=="linear" or cod_actf=="linear" then
        if trainer:has_option("learning_rate") then
          -- if activation is linear, the derivative slope is so high, so we use
          -- learning_rate to reduce its impact
          local ratio = 1/math.sqrt(cod_size+input_size)
          trainer:set_option("learning_rate", trainer:get_option("learning_rate")*ratio)
 	end
      end
      -- printf("BEFORE TRAIN %d\n", i)
      local train_func = trainable.train_wo_validation{
        min_epochs     = lookup("min_epochs"),
        max_epochs     = lookup("max_epochs"),
        percentage_stopping_criterion = lookup("pretraining_percentage_stopping_criterion"),      
      }
      while train_func:execute(function()
          local data = data
          if type(data) == "function" then data = data() end
          return trainer,trainer:train_dataset(data)
      end) do
        local t = train_func:get_state_table()
        printf("%4d %10.6f  (improvement %.4f)\n",
               t.current_epoch, t.train_error, t.train_improvement)
        io.stdout:flush()
      end
      local best_net = trainer
      -- printf("AFTER TRAIN %d\n", i)
      ---------------------------------------------------------
      local b1mat = best_net:weights(params.names_prefix.."b1"):clone()
      local b2mat = best_net:weights(params.names_prefix.."b2"):clone()
      local wmat  = best_net:weights(params.names_prefix.."w"):clone()
      table.insert(weights, wmat)
      table.insert(bias, { b1mat, b2mat })
      --
      mlp_final:push( ann.components.hyperplane{
                        input  = input_size,
                        output = cod_size,
                        name   = params.names_prefix.."layer" .. (i-1),
                        dot_product_name    = params.names_prefix.."w" .. (i-1),
                        bias_name           = params.names_prefix.."b" .. (i-1),
                        dot_product_weights = params.names_prefix.."w" .. (i-1),
                        bias_weights        = params.names_prefix.."b" .. (i-1), })
      mlp_final:push( ann.components.actf[cod_actf](table.merge({ name=params.names_prefix.."actf" .. (i-1)}, cod_params)))
      mlp_final_weights[params.names_prefix.."w" .. (i-1)] = wmat
      mlp_final_weights[params.names_prefix.."b"    .. (i-1)] = b1mat
      --
      --insert the information
      if not on_the_fly then
        if i ~= #params.layers or params.supervised_layer then
          -- printf("LAYER %d\n", i)
          -- generation of new input patterns using only the first part of
          -- autoencoder except at last loop iteration (only if not
          -- supervised_layer)
          local codifier
          codifier = build_two_layered_codifier(params.names_prefix,
                                                input_size,
                                                input_actf,
                                                input_params,
                                                cod_size,
                                                cod_actf,
                                                cod_params)
          local cod_trainer = trainable.supervised_trainer(codifier,
                                                           nil,
                                                           params.bunch_size,
                                                           params.optimizer())
          cod_trainer:build()
          -- print("Load bias ", params.names_prefix .. "b")
          cod_trainer:weights(params.names_prefix.."b"):copy(b1mat)
          cod_trainer:weights(params.names_prefix.."w"):copy(wmat)
          if current_dataset_params.distribution then
            -- compute code for each distribution dataset
            for _,v in ipairs(current_dataset_params.distribution) do
              v.input_dataset = cod_trainer:use_dataset{
                input_dataset = v.input_dataset,
                output_dataset = dataset.matrix(matrix(v.input_dataset:numPatterns(),
                                                       cod_trainer:get_output_size())),
              }
            end
          else
            -- compute code for input dataset
            local ds = cod_trainer:use_dataset{
              input_dataset = current_dataset_params.input_dataset,
              output_dataset = dataset.matrix(matrix(current_dataset_params.input_dataset:numPatterns(),
                                                     cod_trainer:get_output_size())),
            }
            current_dataset_params.input_dataset = ds
          end -- if distribution ... else
        end -- if i ~= params.layers
      end -- for i=2,#params.layers
    end -- if not on_the_fly
    local sdae_table = {weights=weights, bias=bias}
    -- Train a supervised layer
    if params.supervised_layer then
      printf("# Training of supervised layer %d--%d (number %d)\n",
             params.layers[#params.layers].size, params.supervised_layer.size,
             #params.layers)
      local input_size     = params.layers[#params.layers].size
      local input_actf     = params.layers[#params.layers].actf
      local output_size    = params.supervised_layer.size
      local output_actf    = params.supervised_layer.actf
      local global_options    = table.deep_copy(params.training_options.global)
      local layerwise_options = (params.training_options.layerwise[#params.layers] or
                                   { ann_options = {} })
      layerwise_options.ann_options = layerwise_options.ann_options or {}
      local lookup = function(name) return layerwise_options[name] or global_options[name] end
      local data

      if not on_the_fly then
        data = generate_training_table_configuration(current_dataset_params,
                                                     params.replacement,
                                                     params.shuffle_random,
                                                     lookup("noise_pipeline") or {},
                                                     params.output_datasets)

      else
        local mlp_final_trainer = trainable.supervised_trainer(mlp_final:clone(),
                                                               nil,
                                                               params.bunch_size,
                                                               params.optimizer())
        local aux_weights = md.clone( mlp_final_weights )
        mlp_final_trainer:build{ weights = aux_weights }
        data = generate_training_table_configuration_on_the_fly(current_dataset_params,
                                                                params.replacement,
                                                                params.shuffle_random,
                                                                lookup("noise_pipeline") or {},
                                                                mlp_final_trainer,
                                                                params.output_datasets)

      end
      local thenet = ann.mlp.all_all.generate(
        string.format("%d inputs %d %s",
                      input_size,
                      output_size,
                      output_actf),
        nil, params.names_prefix)
      local loss_function
      if output_actf == "log_logistic" then
        loss_function = ann.loss.cross_entropy(output_size)
      elseif output_actf == "log_softmax" then
        loss_function = ann.loss.multi_class_cross_entropy(output_size)
      else
        loss_function = ann.loss.mse(output_size)
      end
      local thenet_trainer = trainable.supervised_trainer(thenet,
                                                          loss_function,
                                                          params.bunch_size,
                                                          params.optimizer())
      thenet_trainer:build()
      for key,value in pairs(global_options.ann_options) do
        if layerwise_options.ann_options[key] == nil then
          thenet_trainer:set_option(key, value)
        end
      end
      for key,value in pairs(layerwise_options.ann_options) do
        thenet_trainer:set_option(key, value)
      end
      for _,reg in ipairs({ "weight_decay", "L1_norm", "max_norm_penalty" }) do
        if thenet_trainer:has_option(reg) then
          thenet_trainer:set_layerwise_option(params.names_prefix.."b.*", reg, 0.0)
        end
      end
      thenet_trainer:randomize_weights{
        random     = params.weights_random,
        inf=-math.sqrt(6 / (input_size + output_size)),
        sup= math.sqrt(6 / (input_size + output_size))
      }
      local train_func = trainable.train_wo_validation{
        min_epochs     = lookup("min_epochs"),
        max_epochs     = lookup("max_epochs"),
        percentage_stopping_criterion = lookup("pretraining_percentage_stopping_criterion"),      
      }
      while train_func:execute(function()
          local data = data
          if type(data) == "function" then data = data() end
          return thenet_trainer,thenet_trainer:train_dataset(data)
      end) do
        local t = train_func:get_state_table()
        printf("%4d %10.6f  (improvement %.4f)\n",
               t.current_epoch, t.train_error, t.train_improvement)
        io.stdout:flush()	
      end
      local best_net_trainer = thenet_trainer
      local wobj  = best_net_trainer:weights(params.names_prefix.."w1"):clone()
      local bobj  = best_net_trainer:weights(params.names_prefix.."b1"):clone()
      local lastn = #params.layers
      mlp_final:push( ann.components.hyperplane{
                        input  = input_size,
                        output = output_size,
                        name   = params.names_prefix.."layer" .. lastn,
                        dot_product_name    = params.names_prefix.."w" .. lastn,
                        bias_name           = params.names_prefix.."b" .. lastn,
                        dot_product_weights = params.names_prefix.."w" .. lastn,
                        bias_weights        = params.names_prefix.."b" .. lastn, })
      mlp_final:push( ann.components.actf[output_actf]{
                        name=params.names_prefix.."actf" .. lastn } )
      mlp_final_weights[params.names_prefix.."w" .. lastn] = wobj
      mlp_final_weights[params.names_prefix.."b"    .. lastn] = bobj
    end
    
    mlp_final:build{ weights = mlp_final_weights }
    
    return sdae_table, mlp_final
  end

-- This function returns a MLP formed by the codification part of a full stacked
-- auto encoder
function ann.autoencoders.build_codifier_from_sdae_table(sdae_table,
							 layers)
  local weights_mat   = sdae_table.weights
  local bias_mat      = sdae_table.bias
  local codifier_net  = ann.components.stack{ name="stack" }
  local weights_table = {}
  for i=2,#layers do
    local bname = "b"..(i-1)
    local wname = "w"..(i-1)
    codifier_net:push( ann.components.hyperplane{
			 name   = "c" .. (i-1),
			 input  = layers[i-1].size,
			 output = layers[i].size,
			 dot_product_name    = wname,
			 dot_product_weights = wname,
			 bias_name           = bname,
			 bias_weights        = bname })
    codifier_net:push( ann.components.actf[layers[i].actf]{ name="actf"..(i-1) })
    weights_table[wname] = ann.connections{ input=layers[i-1].size,
					    output=layers[i].size,
					    w = weights_mat[i-1] }
    weights_table[bname] = ann.connections{ input=1, output=layers[i].size,
					    w = bias_mat[i-1][1] }
  end
  codifier_net:build{ weights = weights_table }
  return codifier_net
end

-- -- This function returns a MLP formed by the codification part of a full stacked
-- -- auto encoder
-- function ann.autoencoders.build_codifier_from_sdae(sdae, bunch_size, layers)
--   local sdae_connections = sdae:get_layer_connections_vector()
--   local sdae_activations = sdae:get_layer_activations_vector()
--   local codifier_net = ann.mlp{ bunch_size = bunch_size }
--   local codifier_connections = {}
--   local codifier_activations = {}
--   for i=1,(#layers-1)*2 do
--     table.insert(codifier_connections, sdae_connections[i]:clone(codifier_net))
--   end
--   local layer_type = "inputs"
--   for i=1,#layers-1 do
--     table.insert(codifier_activations, sdae_activations[i]:clone(codifier_net,
-- 								 layer_type))
--     layer_type = "hidden"
--   end
--   table.insert(codifier_activations, sdae_activations[#layers]:clone(codifier_net,
-- 								     "outputs"))
--   local k=1
--   for i=2,#layers do
--     local actf    = ann.activations.from_string(layers[i].actf)
--     local input   = codifier_activations[i-1]
--     local output  = codifier_activations[i]
--     local bias    = codifier_connections[k]
--     local weights = codifier_connections[k+1]
--     codifier_net:push_back_all_all_layer{
--       input   = input,
--       output  = output,
--       bias    = bias,
--       weights = weights,
--       actfunc = actf }
--     k = k + 2
--   end
--   return codifier_net
-- end

-- Returns a dataset with the codification of input dataset patterns  
function ann.autoencoders.compute_encoded_dataset_using_codifier()
  error("Deprecated, use trainable.supervised_trainer.use_dataset")
end

----------------------------------------------------------------------------

ann.autoencoders.iterative_sampling =
  april_doc{
    class="function",
    summary={"This function generates samples from a given autoencoder.", },
    description=
      {
        "This function generates samples from a given autoencoder,",
        "iterating a until convergence. It is possible to indicate",
        "a mask of input positions that will be keep untouched.",
      },
    params= {
      ["model"] = {"An autoencoder ANN component (not a trainer"},
      ["noise"] = {"An ANN component for noise generation (not a trainer"},
      ["mask"]   = {"An array with the input positions which",
                    "will be keep untouched [optional]",},
      ["input"] = {"A matrix with the input values."},
      ["max"] = "Max number of iterations",
      ["stop"] = "Stop when loss difference between iterations is lower than given value",
      ["verbose"] = "Verbosity true or false [optional].",
      ["alpha"] = "A number with the gradient step at each iteration.",
      ["log"] = "A boolean indicating if the output is logarithmic [optional]",
      ["loss"] = "A loss function instance",
    },
    outputs= {
      {"A table with the input array after sampling, in natural",
       "scale (not logarithmic), even if log=true"},
    }
  } ..
  function(t)
    local params = get_table_fields(
      {
        model      = { mandatory = true,  isa_match  = ann.components.base },
        noise      = { mandatory = true,  isa_match  = ann.components.base },
        mask       = { mandatory = false, type_match = "table", default = {}, },
        max        = { mandatory = true,  type_match = "number" },
        stop       = { mandatory = true,  type_match = "number" },
        verbose    = { mandatory = false, type_match = "boolean", default=false },
        log        = { mandatory = false, type_match = "boolean", default=false },
        input      = { mandatory = true,  isa_match  = matrix  },
        loss       = { mandatory = true,  isa_match  = ann.loss },
      }, t)
    assert(params.model:get_input_size() == params.model:get_output_size(),
           "Input and output sizes must be equal!!! (it is an auto-encoder)")
    local L       = 11111111111111111
    local last_L  = 11111111111111111
    local input_rewrapped = params.input:rewrap(1, params.input:size())
    local input   = input_rewrapped:clone()
    local output  = input
    local chain   = {}
    for i=1,params.max do
      params.model:reset()
      output = params.model:forward(input)
      -- restore masked positions
      -- for _,pos in ipairs(params.mask) do output[pos] = params.input[pos] end
      -- compute the loss of current iteration
      params.loss:reset()
      L = params.loss:loss(output, params.model:get_input())
      if params.log then output:exp() end
      -- restore masked positions
      for _,pos in ipairs(params.mask) do
        output:set(1,pos,input_rewrapped:get(1,pos))
      end
      -- insert current output to the chain
      table.insert(chain, output:rewrap(table.unpack(params.input:dim())))
      -- improvement measure
      local imp = math.abs(math.abs(last_L - L)/last_L)
      if params.verbose then printf("%6d %6g :: %6g\n", i, L, imp) end
      -- stop criterion
      if last_L == 0 or imp < params.stop then break end
      last_L = L
      -- sample from noise distribution
      params.noise:reset()
      local input_token = params.noise:forward(output)
      input = input_token
      -- restore masked positions
      for _,pos in ipairs(params.mask) do
        input:set(1,pos,input_rewrapped:get(1,pos))
      end
    end
    return output:rewrap(table.unpack(params.input:dim())),L,chain
  end

----------------------------------------------------------------------------

ann.autoencoders.sgd_sampling =
  april_doc{
    class="function",
    summary={"This function generates samples from a given autoencoder.", },
    description=
      {
        "This function generates samples from a given autoencoder,",
        "using gradient descent a until convergence. It is possible to indicate",
        "a mask of input positions that will be keep untouched.",
      },
    params= {
      ["model"] = {"An autoencoder ANN component (not a trainer"},
      ["noise"] = {"An ANN component for noise generation (not a trainer"},
      ["mask"]   = {"An array with the input positions which",
                    "will be keep untouched [optional]",},
      ["input"] = {"A matrix with the input values."},
      ["max"] = "Max number of iterations",
      ["stop"] = "Stop when loss difference between iterations is lower than given value",
      ["verbose"] = "Verbosity true or false [optional].",
      ["alpha"] = "A number with the gradient step at each iteration.",
      ["beta"]  = "A number for combination with iterative sampling.",
      ["clamp"] = "A function to clamp sample values [optional].",
      ["log"] = "A boolean indicating if the output is logarithmic [optional]",
      ["loss"] = "A loss function instance",
    },
    outputs= {
      {"A table with the input array after sampling, in natural",
       "scale (not logarithmic), even if log=true"},
    }
  } ..
  function(t)
    local params = get_table_fields(
      {
        model      = { mandatory = true,  isa_match  = ann.components.base },
        noise      = { mandatory = true,  isa_match  = ann.components.base },
        mask       = { mandatory = false, type_match = "table", default = {}, },
        input      = { mandatory = true,  isa_match  = matrix },
        max        = { mandatory = true,  type_match = "number" },
        stop       = { mandatory = true,  type_match = "number" },
        verbose    = { mandatory = false, type_match = "boolean", default=false },
        alpha      = { mandatory = true,  type_match = "number" },
        beta       = { mandatory = false, type_match = "number",  default=0.0 },
        clamp      = { mandatory = false, type_match = "function",
                       default = function(v) return v end, },
        log        = { mandatory = false, type_match = "boolean", default=false },
        loss       = { mandatory = true,  isa_match  = ann.loss },
      }, t)
    assert(params.model:get_input_size() == params.model:get_output_size(),
           "Input and output sizes must be equal!!! (it is an auto-encoder)")
    local L        = 11111111111111111
    local last_L   = 11111111111111111
    local min      = 11111111111111111
    local input_rewrapped = params.input:rewrap(1, params.input:size())
    local input    = input_rewrapped:clone()
    local output   = input
    local result   = input
    local chain    = {}
    for i=1,params.max do
      params.model:reset()
      output = params.model:forward(input):clone()
      -- restore masked positions
      -- for _,pos in ipairs(params.mask) do output[pos] = params.input[pos] end
      -- compute the loss of current iteration
      params.loss:reset()
      L = params.loss:loss(output, params.model:get_input())
      if params.log then output:exp() end
      -- restore masked positions
      for _,pos in ipairs(params.mask) do
        output:set(1,pos,input_rewrapped:get(1,pos))
      end
      table.insert(chain, output:rewrap(table.unpack(params.input:dim())))
      local imp = math.abs(math.abs(last_L - L)/last_L)
      if params.verbose then printf("%6d %6g :: %6g", i, L, imp) end
      if i==1 or L <= min then
        min,result = L,output
        if params.verbose then printf(" *") end
      end
      if params.verbose then printf("\n") end
      if last_L == 0 or imp < params.stop then break end
      -- GRADIENT DESCENT UPDATE OF INPUT VECTOR
      --aux = params.noise:forward(input)
      ---- restore masked positions
      --for _,pos in ipairs(params.mask) do
      --aux:set(1,pos,input_rewrapped:get(1,pos))
      --end
      
      local gradient = params.model:backprop(params.loss:gradient(params.model:get_output(),
                                                                  params.model:get_input()))
      -- local g = gradient:clone():rewrap(16,16):pow(2):sqrt():clamp(0,1)
      -- matrix.saveImage(g, string.format("gradient-%04d.pnm", i))
      gradient = gradient
      output   = output
      -- input = (1 - beta)*input + beta*output - alpha*gradient
      input = ( input:clone():
                  scal(1.0 - params.beta):
                  axpy(params.beta, output):
                  axpy(-params.alpha, gradient) )
      params.clamp(input)
      --
      last_L = L
      -- sample from noise distribution
      params.noise:reset()
      input = params.noise:forward(input)
      -- restore masked positions
      for _,pos in ipairs(params.mask) do
        input:set(1,pos,input_rewrapped:get(1,pos))
      end
    end
    return result,min,chain
  end
