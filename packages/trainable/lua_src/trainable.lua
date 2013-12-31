-------------------------------
-- LOCAL AUXILIARY FUNCTIONS --
-------------------------------

local MAX_ITERS_WO_COLLECT_GARBAGE=10000

local function check_dataset_sizes(ds1, ds2)
  assert(ds1:numPatterns() == ds2:numPatterns(),
	 string.format("Different input/output datasets "..
			 "numPatterns found: "..
			 "%d != %d",
		       ds1:numPatterns(),
		       ds2:numPatterns()))
end

-----------------------
-- TRAINABLE CLASSES --
-----------------------

-------------------------------------------------------------------------------

------------------------------
-- SUPERVISED_TRAINER CLASS --
------------------------------
april_set_doc("trainable.supervised_trainer", {
		class       = "class",
		summary     = "Supervised machine learning trainer",
		description ={
		  "This class implements methods useful to",
		  "train, evalute and modify contents of",
		  "ANN components or similar supervised learning",
		  "models.",
		  "IMPORTANT: Any method of this class could be overwritten",
		  "on object instances, and the overwritten methods will be",
		  "copied by clone method. Example:",
		  "t = trainable.supervised_trainer(ann.components.base())",
		  "t.train_step=function() print(\"Hola\") end",
		}, })

local trainable_supervised_trainer_methods,
trainable_supervised_trainer_class_metatable=class("trainable.supervised_trainer")

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.__call", {
		class = "method", summary = "Constructor",
		description ={"Constructor of the supervised_trainer class.",
			      "This class implements methods useful to",
			      "train, evalute and modify contents of",
			      "ANN components or similar supervised learning",
			      "models.",
			      "If the component is in build state, the",
			      "constructed trainer is in build state also.",
		},
		params = {
		  "ANN component or similar supervised learning model",
		  "Loss function [optional]",
		  "Bunch size (mini batch) [optional]",
		  "An optimizer [option], by default is ann.optimizer.sgd",
		  {
		    "A gradient smoothing boolean flag [optional], by default",
		    "it is true.",
		    "This parameter scales the gradients depending in the",
		    "bunch_size and in the number of times the weight was used",
		  },
		},
		outputs = { "Instantiated object" }, })

function trainable_supervised_trainer_class_metatable:__call(...)
  if select('#',...) == 1 and type(select(1,...)) == "table" then
    -- Loading from a previously saved object, the first argument is a table
    local t = select(1,...)
    local model = t.model
    local connections = t.connections
    local loss        = t.loss
    local bunch_size  = t.bunch_size
    local optimizer   = t.optimizer
    local smooth_gradients = t.smooth_gradients or true
    local obj = trainable.supervised_trainer(model, loss, bunch_size, optimizer,
					     smooth_gradients)
    obj:build()
    for wname,cobj in obj:iterate_weights() do
      local w = connections[wname]
      if type(w) == "table" then
	w = w.w
	local oldw = w.oldw
	assert(w ~= nil, "Component " .. wname .. " not found at file")
	assert(connections[wname].input == ann.connections.get_input_size(cobj),
	       string.format("Incorrect input size, expected %d, found %d\n",
			     ann.connections.get_input_size(cobj),
			     connections[wname].input))
	assert(connections[wname].output == ann.connections.get_output_size(cobj),
	       string.format("Incorrect output size, expected %d, found %d\n",
			     ann.connections.get_output_size(cobj),
			     connections[wname].output))
	if obj.optimizer.update then
	  obj.optimizer.update[wname] = obj.optimizer.update[wname] or oldw
	end
      end
      if w:get_major_order() == cobj:get_major_order() then
	cobj:copy(w)
      else
	ann.connections.load(cobj, { w=w })
      end
    end
    return obj
  else
    -- Constructor of a new object
    local ann_component,loss_function,bunch_size,optimizer,smooth_gradients = ...
    local optimizer = optimizer or ann.optimizer.sgd()
    local smooth_gradients = smooth_gradients or true
    if loss_function and not isa(loss_function, ann.loss) then
      error("The second parameter must be an instance of ann.loss")
    end
    if optimizer and not isa(optimizer, ann.optimizer) then
      error("The fourth parameter must be an instance of ann.optimizer")
    end
    if bunch_size and not tonumber(bunch_size) then
      error("The third parameter must be a number")
    end
    local obj = {
      ann_component    = assert(ann_component,"Needs an ANN component object"),
      loss_function    = loss_function or false,
      optimizer        = optimizer,
      smooth_gradients = smooth_gradients,
      weights_table    = {},
      components_table = {},
      component2weights_dict = {},
      weights2component_dict = {},
      weights_order    = {},
      components_order = {},
      bunch_size       = bunch_size or false,
      weight_grads     = {},
    }
    obj = class_instance(obj, self, true)
    return obj
  end
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.get_component", {
		class = "method",
		summary = "Returns an instance of ann.components",
		outputs = { "An instance of ann.components" }, })

function trainable_supervised_trainer_methods:get_component()
  return self.ann_component
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.set_loss_function", {
		class = "method",
		summary = "Modifies the loss function property",
		params = { "Loss function" }, })

function trainable_supervised_trainer_methods:set_loss_function(loss_function)
  assert(isa(loss_function, ann.loss, "Needs an instance of ann.loss"))
  self.loss_function = loss_function
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.set_optimizer", {
		class = "method",
		summary = "Modifies the optimizer property",
		params = { "An instance of ann.optimizer" }, })

function trainable_supervised_trainer_methods:set_optimizer(optimizer)
  assert(isa(optimizer, ann.optimizer, "Needs an instance of ann.optimizer"))
  self.optimizer = optimizer
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.get_optimizer", {
		class = "method",
		summary = "Returns the optimizer property",
		outputs = { "An instance of ann.optimizer" }, })

function trainable_supervised_trainer_methods:get_optimizer()
  return self.optimizer
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.set_option", {
		class = "method",
		summary = "Sets a global option of the optimizer",
		params  = {
		  "An string with the option name",
		  "A number with the value",
		}, })

function trainable_supervised_trainer_methods:set_option(name,value)
  local opt = assert(self.optimizer, "The optimizer has not been defined")
  opt:set_option(name,value)
end

april_set_doc("trainable.supervised_trainer.get_option", {
		class = "method",
		summary = "Gets a global option of the optimizer",
		params  = {
		  "An string with the option name",
		},
		outputs = {
		  "A number with the option value",
		} })

function trainable_supervised_trainer_methods:get_option(name)
  local opt = assert(self.optimizer, "The optimizer has not been defined")
  return opt:get_option(name)
end

april_set_doc("trainable.supervised_trainer.has_option", {
		class = "method",
		summary = "Returns true/false depending on the existence of a global option of the optimizer",
		params  = {
		  "An string with the option name",
		},
		outputs = {
		  "A boolean",
		} })

function trainable_supervised_trainer_methods:has_option(name)
  local opt = assert(self.optimizer, "The optimizer has not been defined")
  return opt:has_option(name)
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.set_layerwise_option", {
		class = "method",
		summary = "Sets an specific weights layer option of the optimizer",
		description = {
		  "Sets the value of an optimizer option specifying the",
		  "layer name by a Lua regular expression, so the option is set",
		  "for all weight layers which match the given name",
		},
		params  = {
		  "An string with the Lua regular expression",
		  "An string with the option name",
		  "A number with the value",
		}, })

function trainable_supervised_trainer_methods:set_layerwise_option(layer_match,
								   name,
								   value)
  local opt = assert(self.optimizer, "The optimizer has not been defined")
  for cnn_name,cnn in self:iterate_weights(layer_match) do
    opt:set_layerwise_option(cnn_name,name,value)
  end
end

april_set_doc("trainable.supervised_trainer.get_option_of", {
		class = "method",
		summary = "Returns the optimizer option value for a given layer name",
		description = {
		  "If the layer has a layer-wise option, it will be returned,",
		  "otherwise the global option value will be returned",
		},
		params  = {
		  "An string with the Lua regular expression",
		  "An string with the option name",
		},
		outputs = { "A number with the value", }, })

function trainable_supervised_trainer_methods:get_option_of(layer_name,name)
  local opt = assert(self.optimizer, "The optimizer has not been defined")
  return opt:get_option_of(layer_name,name)
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.get_input_size", {
		class = "method",
		summary = "Gets the input size of its component",
		outputs = { "The input size (a number)" }, })

function trainable_supervised_trainer_methods:get_input_size()
  return self.ann_component:get_input_size()
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.get_output_size", {
		class = "method",
		summary = "Gets the output size of its component",
		outputs = { "The output size (a number)" }, })

function trainable_supervised_trainer_methods:get_output_size()
  return self.ann_component:get_output_size()
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.size", {
		class = "method",
		summary = "Returns the model size (number of weights)",
		outputs = { "A number" }, })

function trainable_supervised_trainer_methods:size()
  if #self.components_order == 0 then
    error("It is not build")
  end
  local sz = 0
  for wname,cnn in pairs(self.weights_table) do
    sz = sz + cnn:size()
  end
  return sz
end

------------------------------------------------------------------------

function trainable_supervised_trainer_methods:to_lua_string(format)
  assert(#self.components_order > 0, "The component is not built")
  local t = { }
  table.insert(t, "trainable.supervised_trainer{ ")
  table.insert(t, "model=")
  table.insert(t, self.ann_component:to_lua_string(format))
  table.insert(t, ",\n")
  table.insert(t, "connections={")
  for _,wname in ipairs(self.weights_order) do
    local cobj = self.weights_table[wname]
    local w = cobj
    table.insert(t, string.format("\n[%q] = ", wname))
    table.insert(t, w:to_lua_string(format))
    table.insert(t, ",")
  end
  table.insert(t, "\n},\n")
  if self.loss_function then
    table.insert(t, "loss=")
    table.insert(t, self.loss_function:to_lua_string(format))
    table.insert(t, ",\n")
  end
  if self.optimizer then
    table.insert(t, "optimizer=")
    table.insert(t, self.optimizer:to_lua_string(format))
    table.insert(t, ",\n")
  end
  if self.bunch_size then
    table.insert(t, "bunch_size=")
    table.insert(t, self.bunch_size)
    table.insert(t, ",\n")
  end
  if self.smooth_gradients ~= nil then
    table.insert(t, "smooth_gradients=")
    table.insert(t, tostring(self.smooth_gradients))
    table.insert(t, ",\n")
  end
  table.insert(t, "}")
  return table.concat(t, "")
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.save", {
		class = "method",
		summary = "Save the model at a disk file",
		description = {
		  "Save the model and connection weights at",
		  "a disk file.",
		  "Only works after build method is called.",
		},
		params = {
		  "A filename string",
		  { "A string indicating the matrix format: ascii or binary",
		    "[optional]. By default is binary." },
		}, })

function trainable_supervised_trainer_methods:save(filename, format)
  assert(#self.components_order > 0, "The component is not built")
  local f = io.open(filename,"w") or error("Unable to open " .. filename)
  f:write("return ")
  f:write(self:to_lua_string(format))
  f:write("\n")
  f:close()
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.load", {
		class = "function",
		summary = "Load the model and weights from a disk file",
		description = {
		  "Load the model and connection weights stored at",
		  "a disk file. The trainer is loaded at build state.",
		},
		params = {
		  "A filename string",
		  "Loss function [optional]",
		  "Bunch size (mini batch) [optional]",
		  "An optimizer, instance of ann.optimizer [optional]",
		}, })

function trainable.supervised_trainer.load(filename, loss, bunch_size, optimizer)
  local f   = loadfile(filename) or error("Unable to open " .. filename)
  local obj = f() or error("Impossible to load chunk from file " .. filename)
  if type(obj) == "table" then
    -- OLD FORMAT LOADER
    obj = trainable.supervised_trainer(obj)
  end
  obj.bunch_size    = bunch_size or obj.bunch_size
  obj.loss_function = loss       or obj.loss_function
  obj.optimizer     = optimizer  or obj.optimizer
  return obj
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.count_components", {
		class = "method",
		summary = "Count the number of components",
		params = {
		  { "A match string: filter and count only components",
		    "which match [optional], by default is '.*'" },
		}, })

function trainable_supervised_trainer_methods:count_components(match_string)
  local match_string = match_string or ".*"
  if #self.components_order == 0 then
    error("It is not build")
  end
  local count = 0
  for i=1,#self.components_order do
    if self.components_order[i]:match(match_string) then count=count+1 end
  end
  return count
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.count_weights", {
		class = "method",
		summary = "Count the number of connection weight objects",
		params = {
		  { "A match string: filter and count only connections",
		    "which match [optional], by default is '.*'" },
		}, })

function trainable_supervised_trainer_methods:count_weights(match_string)
  local match_string = match_string or ".*"
  if #self.components_order == 0 then
    error("It is not build")
  end
  local count = 0
  for i=1,#self.weights_order do
    if self.weights_order[i]:match(match_string) then count=count+1 end
  end
  return count
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.iterate_components", {
		class = "method",
		summary = "Iterates over components",
		description =
		  {
		    "This method is an iterator function to be used at for",
		    "loops: for name,component in trainer:iterate_components()",
		    "do print(name,component) end",
		  },
		params = {
		  { "A match string: filter and iterates only on components",
		    "which match [optional], by default is '.*'" },
		}, })

function trainable_supervised_trainer_methods:iterate_components(match_string)
  local match_string = match_string or ".*"
  if #self.components_order == 0 then
    error("It is not build")
  end
  local pos = 0
  return function()
    repeat
      pos = pos + 1
      if pos > #self.components_order then
	return nil
      end
    until self.components_order[pos]:match(match_string)
    local name = self.components_order[pos]
    return name,self.components_table[name]
  end
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.iterate_weights", {
		class = "method",
		summary = "Iterates over weight connection objects",
		description = 
		  {
		    "This method is an iterator function to be used at for",
		    "loops: for name,connections in trainer:iterate_weights()",
		    "do print(name,component) end",
		  },
		params = {
		  { "A match string: filter and count only connections",
		    "which match [optional], by default is '.*'" },
		}, })

function trainable_supervised_trainer_methods:iterate_weights(match_string)
  local match_string = match_string or ".*"
  if #self.components_order == 0 then
    error("It is not build")
  end
  local pos = 0
  return function()
    repeat
      pos = pos + 1
      if pos > #self.weights_order then
	return nil
      end
    until self.weights_order[pos]:match(match_string)
    local name = self.weights_order[pos]
    return name,self.weights_table[name]
  end
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.component", {
		class = "method",
		summary = "Returns a component given its name",
		description =
		  {
		    "This method returns a component object",
		    "which name is the given argument.",
		    "This method is forbidden before build method is called.",
		    "If an error is produced, it returns nil."
		  },
		params = { "A string with the component name" },
		outputs = { "A component object" } })

function trainable_supervised_trainer_methods:component(str)
  if #self.components_order == 0 then
    error("Needs execution of build method")
  end
  return self.components_table[str]
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.weights", {
		class = "method",
		summary = "Returns a connections object given its name",
		description =
		  {
		    "This method returns a connections object",
		    "which name is the given argument.",
		    "This method is forbidden before build method is called.",
		    "If an error is produced, returns nil."
		  }, 
		params = { "A string with the connections name" },
		outputs = { "An ann.connections object" } })

function trainable_supervised_trainer_methods:weights(str)
  if #self.components_order == 0 then
    error("Needs execution of build method")
  end
  return self.weights_table[str]
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.get_weights_table", {
		class = "method",
		summary = "Returns the table with all weights",
		outputs = { "A table with all the weights, indexed by names" } })

function trainable_supervised_trainer_methods:get_weights_table()
  if #self.components_order == 0 then
    error("Needs execution of build method")
  end
  return self.weights_table
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.randomize_weights", {
		class = "method",
		summary = "Initializes randomly model weights and biases",
		description =
		  {
		    "This method initialies the weights, following an uniform",
		    "distribution, in the range [c*inf,c*sup].",
		    "Constant c depends on fan-in and/or fan-out fields.",
		    "If fan-in and fan-out are false, then c=1.",
		    "If fan-in=true and fan-out=false, then c=1/sqrt(fanin).",
		    "If fan-in=false and fan-out=true, then c=1/sqrt(fanout).",
		    "If fan-in and fan-out are true, then c=1/sqrt(fanin + fanout).",
		  },
		params = {
		  ["name_match"] = {
		    "A match string [optional], if given, only the connection",
		    "weights which match will be randomized",
		  },
		  ["random"] = "A random object",
		  ["inf"]    = "Range inf value",
		  ["sup"]    = "Range sup value",
		  ["use_fanin"] = "An [optional] boolean, by default false",
		  ["use_fanout"] = "An [optional] boolean, by default false",
		}, })

function trainable_supervised_trainer_methods:randomize_weights(t)
  local params = get_table_fields(
    {
      name_match = { type_match="string", mandatory=false, default = nil },
      random  = { isa_match = random,  mandatory = true },
      inf     = { type_match="number", mandatory = true },
      sup     = { type_match="number", mandatory = true },
      use_fanin  = { type_match="boolean", mandatory = false, default = false },
      use_fanout = { type_match="boolean", mandatory = false, default = false },
    }, t)
  assert(#self.components_order > 0,
	 "Execute build method before randomize_weights")
  for i,wname in ipairs(self.weights_order) do
    if not params.name_match or wname:match(params.name_match) then
      local current_inf = params.inf
      local current_sup = params.sup
      local constant    = 0
      local connection  = self.weights_table[wname]
      if params.use_fanin then
	constant = constant + ann.connections.get_input_size(connection)
      end
      if params.use_fanout then
	constant = constant + ann.connections.get_output_size(connection)
      end
      if constant > 0 then
	current_inf = current_inf / math.sqrt(constant)
	current_sup = current_sup / math.sqrt(constant)
      end
      ann.connections.randomize_weights(connection,
					{ random = params.random,
					  inf    = current_inf,
					  sup    = current_sup })
    end
  end
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.build", {
		class = "method",
		summary = "Executes build method of the component",
		description = 
		  {
		    "This method executes the build method of its",
		    "ann_component property, and weights_order, weights_table,",
		    "components_order and components_table properties are",
		    "also built. The method returns two tables with the",
		    "content of weights_table and components_table, in order",
		    "to provide easy acces to components and connections.",
		  }, 
		params = {
		  ["weights"] = "A dictionary weights_name => ann.connections object [optional]",
		  ["input"]   = "The input size of the component [optional]",
		  ["output"]  = "The output size of the component [optional]",
		},
		outputs = {
		  "Weights table, associates weights_name => ann.connections object",
		  "Components table, associates component_name => ann.components object",
		} })

function trainable_supervised_trainer_methods:build(t)
  local params = get_table_fields(
    {
      weights = { type_match="table",  mandatory = false, default=nil },
      input   = { type_match="number", mandatory = false, default=nil },
      output  = { type_match="number", mandatory = false, default=nil },
    }, t or {})
  self.weight_grads  = {}
  self.weights_table = params.weights or {}
  _,
  self.weights_table,
  self.components_table = self.ann_component:build{
    input   = params.input,
    output  = params.output,
    weights = self.weights_table, }
  self.weights_order = {}
  for name,_ in pairs(self.weights_table) do
    table.insert(self.weights_order, name)
  end
  table.sort(self.weights_order)
  self.components_order = {}
  self.component2weights_dict = {}
  self.weights2component_dict = {}
  for name,c in pairs(self.components_table) do
    table.insert(self.components_order, name)
    if c:has_weights_name() then
      local wname = c:get_weights_name()
      self.component2weights_dict[name]  = c:get_weights_name()
      self.weights2component_dict[wname] = self.weights2component_dict[wname] or {}
      table.insert(self.weights2component_dict[wname], c)
    end
  end
  table.sort(self.components_order)
  return self.weights_table,self.components_table
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.get_weights_of", {
		class = "method",
		summary = "Returns a the object connections related to given component name",
		params = { "A string with the component name" },
		outputs = { "An instance of ann.connections" }, })

function trainable_supervised_trainer_methods:get_weights_of(name)
  return self.weights_table[self.component2weights_dict[name]]
end

april_set_doc("trainable.supervised_trainer.get_components_of", {
		class = "method",
		summary = "Returns a table with the components related to given weights name",
		params = { "A string with the weights name" },
		outputs = { "A table of ann.components instances" }, })


function trainable_supervised_trainer_methods:get_components_of(wname)
  return self.weights2component_dict[wname] or {}
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.train_step", {
		class = "method",
		summary = "Executes one training step",
		description = 
		  {
		    "This method executes one training step of its component",
		    "with the given pair input/target output.",
		    "It returns the loss for the given pair of patterns and",
		    "the gradient computed at component inputs.",
		  }, 
		params = {
		  "A table with one input pattern or a token (with one or more patterns)",
		  "The corresponding target output pattern (table or token)",
		  "The loss function [optional]",
		  "An optimizer [optional]",
		  "The bunch size [optional]",
		  "A smooth gradients boolean [optional]",
		},
		outputs = {
		  "The mean of loss function at current batch",
		  "A matrix with the loss of every given pattern",
		} })

function trainable_supervised_trainer_methods:train_step(input, target, loss,
							 optimizer,
							 bunch_size,
							 smooth_gradients)
  if type(input)  == "table" then input  = matrix.col_major(input)  end
  if type(target) == "table" then target = matrix.col_major(target) end
  local loss       = loss or self.loss_function or error("Needs a loss object")
  local optimizer  = optimizer or self.optimizer or error("Needs an optimizer object")
  local bunch_size = bunch_size or self.bunch_size or 1
  local smooth_gradients = smooth_gradients or self.smooth_gradients
  local tr_loss, _, tr_loss_matrix =
    optimizer:execute(function(it)
			local loss  = loss
			local model = self.ann_component
			local grads = self.weight_grads
			model:reset(it)
			local output = model:forward(input, true)
			local tr_loss,tr_loss_matrix
			tr_loss,tr_loss_matrix = loss:compute_loss(output, target)
			if not tr_loss_matrix then return nil end
			gradient = loss:gradient(output, target)
			gradient = model:backprop(gradient)
			--
			for name,mat in pairs(grads) do mat:zeros() end
			--
			grads = model:compute_gradients(grads)
			self.weight_grads = grads
			-- gradient smoothing
			if smooth_gradients then
			  for name,mat in pairs(grads) do
			    local N = grads[name]:get_shared_count()
			    N       = ( N>0 and N) or 1
			    mat:scal( 1.0/math.sqrt(N * bunch_size) )
			  end
			end
			return 
			  -- the loss
			  tr_loss,
			-- the gradients
			self.weight_grads,
			-- the loss matrix
			tr_loss_matrix
		      end,
		      self.weights_table)
  if tr_loss_matrix then loss:accum_loss(tr_loss_matrix) end
  return tr_loss,tr_loss_matrix
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.validate_step", {
		class = "method",
		summary = "Executes one validate step",
		description = 
		  {
		    "This method performs one forward step and computes",
		    "the loss for the given pair input/target output.",
		  }, 
		params = {
		  "A table with one input pattern or a token (with one or more patterns)",
		  "The corresponding target output pattern (table or token)",
		  "The loss function [optional]",
		},
		outputs = {
		  "The mean of loss function at given batch",
		  "A matrix with the loss of every pattern",
		} })

function trainable_supervised_trainer_methods:validate_step(input, target, loss)
  if type(input)  == "table" then input  = matrix.col_major(input)  end
  if type(target) == "table" then target = matrix.col_major(target) end
  local model = self.ann_component
  local loss  = loss or self.loss_function
  model:reset()
  local output   = model:forward(input)
  local tr_loss,tr_loss_matrix = loss:compute_loss(output, target)
  if tr_loss_matrix then
    loss:accum_loss(tr_loss_matrix)
    return tr_loss,tr_loss_matrix
  end
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.compute_gradients_step", {
		class = "method",
		summary = "Executes one gradients computation step",
		params = {
		  "A table with one input pattern or a token (with one or more patterns)",
		  "The corresponding target output pattern (table or token)",
                  "The loss function [optional].",
		  "A table with matrices where to store the gradients [optional]",
		},
		outputs = {
		  "A table with the gradient matrices.",
		  "The mean of loss function at given batch",
		  "A matrix with the loss of each pattern.",
		} })

function trainable_supervised_trainer_methods:compute_gradients_step(input,
								     target,
								     loss,
								     weight_grads)
  if type(input)  == "table" then input  = matrix.col_major(input)  end
  if type(target) == "table" then target = matrix.col_major(target) end
  local loss         = loss or self.loss_function
  local weight_grads = weight_grads or {}
  local tr_loss,tr_loss_matrix,gradient
  self.ann_component:reset()
  local output = self.ann_component:forward(input, true)
  tr_loss,tr_loss_matrix = loss:compute_loss(output, target)
  if tr_loss_matrix then
    loss:accum_loss(tr_loss_matrix)
    gradient = loss:gradient(output, target)
    gradient = self.ann_component:backprop(gradient)
    --
    iterator(pairs(weight_grads)):
    apply(function(name,mat)mat:zeros()end)
    --
    weight_grads = self.ann_component:compute_gradients(weight_grads)
    return weight_grads,tr_loss,tr_loss_matrix
  end
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.grad_check_step", {
		class = "method",
		summary = "Executes one gradients check step",
		params = {
		  "A table with one input pattern or a token (with one or more patterns)",
		  "The corresponding target output pattern (table or token)",
		  "A boolean, true if you want high verbosity level [optional]",
                  "The loss function [optional].",
		},
		outputs = {
		  "A boolean, true or false if the gradient is correct or not",
		} })

function trainable_supervised_trainer_methods:grad_check_step(input, target, verbose, loss)
  if type(input)  == "table" then input  = matrix.col_major(input)  end
  if type(target) == "table" then target = matrix.col_major(target) end
  local loss = loss or self.loss_function
  self.ann_component:reset()
  loss:reset()
  local output   = self.ann_component:forward(input, true)
  local tr_loss,tr_loss_matrix = loss:compute_loss(output, target)
  if not tr_loss_matrix then return true end
  local gradient = loss:gradient(output, target)
  gradient=self.ann_component:backprop(gradient)
  self.weight_grads = self.ann_component:compute_gradients(self.weight_grads)
  local epsilond = 0.3
  local epsilon  = 1e-03
  local ret      = true
  local bunch_size = tr_loss_matrix:dim(1)
  local it = 1
  for wname,cnn in self:iterate_weights() do
    local w = cnn
    -- The shared parameter has no effect in gradients check, only bunch_size
    local ratio = 1/bunch_size
    local ann_grads = self.weight_grads[wname]
    for i=1,w:size() do
      local orig_w = w:raw_get(i-1)
      w:raw_set(i-1, orig_w - epsilon)
      self.ann_component:reset(it)
      it=it+1
      local loss_a = loss:compute_loss(self.ann_component:forward(input,true),
				       target)
      w:raw_set(i-1, orig_w + epsilon)
      self.ann_component:reset(it)
      it=it+1
      local loss_b = loss:compute_loss(self.ann_component:forward(input,true),
				       target)
      w:raw_set(i-1, orig_w)
      local g = (loss_b - loss_a) / (2*epsilon)
      local ann_g = ann_grads:raw_get(i-1)*ratio
      if verbose then
	fprintf(io.stderr,
		"CHECK GRADIENT %s[%d], found %g, expected %g\n",
		wname, i-1, ann_g, g)
      end
      if ann_g ~= 0 or g ~= 0 then
	local abs_err = math.abs(ann_g - g)
	local err = abs_err/math.abs(ann_g+g)
	if err > epsilond and abs_err > 1e-02 then
	  fprintf(io.stderr,
		  "INCORRECT GRADIENT FOR %s[%d], found %g, expected %g "..
		    "(error %g, abs error %g)\n",
		  wname, i-1, ann_g, g, err, abs_err)
	  ret = false
	end
      end
    end
  end
  return ret
end

  ------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.calculate", {
		class = "method",
		summary = "Executes one forward",
		description = 
		  {
		    "This method performs one forward step and returns",
		    "the computed output for the given input.",
		  }, 
		params = {
		  "A table with one input pattern, a col-major matrix or a token (with one or more patterns)",
		},
		outputs = {
		  "A col-major matrix with the computed output",
		} })

function trainable_supervised_trainer_methods:calculate(input)
  if type(input) == "table" then input = matrix.col_major(input) end
  self.ann_component:reset()
  return self.ann_component:forward(input):get_matrix()
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.train_dataset", {
		class = "method",
		summary = "Executes one training epoch with a given dataset",
		description = 
		  {
		    "This method performs one training epoch with a given",
		    "dataset traversing patterns in order, and returns the",
		    "mean loss of each training step.",
		    "Each training step is performed with bunch_size patterns.",
		  }, 
		params = {
		  ["input_dataset"]  = "A dataset float or dataset token",
		  ["output_dataset"] = "A dataset float or dataset token (target output)",
		  ["loss"]           = "A loss function. It is [optional] if loss given at constructor",
		  ["optimizer"]      = "An optimizer. It is [optional] if optimizer is given at constructor",
		  ["bunch_size"]     = 
		    {
		      "Bunch size (mini-batch). It is [optional] if bunch_size",
		      "was set at constructor, otherwise it is mandatory.",
		    }, 
		},
		outputs = {
		  "A number with the mean loss of each training step",
		  "A number with the sample variance of the loss",
		} })

april_set_doc("trainable.supervised_trainer.train_dataset", {
		class = "method",
		summary = "Executes one training epoch with shuffle",
		description = 
		  {
		    "This method performs one training epoch with a given",
		    "dataset traversing patterns in shuffle order, and returns the",
		    "mean loss of each training step.",
		    "Each training step is performed with bunch_size patterns.",
		  }, 
		params = {
		  ["input_dataset"]  = "A dataset float or dataset token",
		  ["output_dataset"] = "A dataset float or dataset token (target output)",
		  ["shuffle"]        = "A random object used to shuffle patterns before training",
		  ["loss"]           = "A loss function. It is [optional] if loss given at constructor",
		  ["optimizer"]      = "An optimizer. It is [optional] if optimizer is given at constructor",
		  ["bunch_size"]     = 
		    {
		      "Bunch size (mini-batch). It is [optional] if bunch_size",
		      "was set at constructor, otherwise it is mandatory.",
		    }, 
		},
		outputs = {
		  "A number with the mean loss of each training step",
		  "A number with the sample variance of the loss",
		} })

april_set_doc("trainable.supervised_trainer.train_dataset", {
		class = "method",
		summary = "Executes one stochastic training epoch with replacement",
		description = 
		  {
		    "This method performs one stochastic training epoch with a given",
		    "dataset. Patterns are choosed randomly with replacement",
		    "until a given replacement size. The",
		    "mean loss of each training step is returned.",
		    "Each training step is performed with bunch_size patterns.",
		  }, 
		params = {
		  ["input_dataset"]  = "A dataset float or dataset token",
		  ["output_dataset"] = "A dataset float or dataset token (target output)",
		  ["shuffle"]        = "A random object used to shuffle patterns before training",
		  ["replacement"]    = "A number with the size of replacement training",
		  ["loss"]           = "A loss function. It is [optional] if loss given at constructor",
		  ["optimizer"]      = "An optimizer. It is [optional] if optimizer is given at constructor",
		  ["bunch_size"]     = 
		    {
		      "Bunch size (mini-batch). It is [optional] if bunch_size",
		      "was set at constructor, otherwise it is mandatory.",
		    }, 
		},
		outputs = {
		  "A number with the mean loss of each training step",
		  "A number with the sample variance of the loss",
		} })

april_set_doc("trainable.supervised_trainer.train_dataset", {
		class = "method",
		summary = "Executes one stochastic training epoch with distribution",
		description = 
		  {
		    "This method performs one stochastic training epoch with a given",
		    "set of datasets with different a-priory probabilities.",
		    "Patterns are choosed randomly with replacement following",
		    "given a-priori distribution, until a given replacement",
		    "size. The mean loss of each training step is returned.",
		    "Each training step is performed with bunch_size patterns.",
		  }, 
		params = {
		  ["distibution"]    = "An array of tables with input_dataset,"..
		    " output_dataset and probability fields",
		  ["shuffle"]        = "A random object used to shuffle patterns before training",
		  ["replacement"]    = "A number with the size of replacement training",
		  ["loss"]           = "A loss function. It is [optional] if loss given at constructor",
		  ["optimizer"]      = "An optimizer. It is [optional] if optimizer is given at constructor",
		  ["bunch_size"]     = 
		    {
		      "Bunch size (mini-batch). It is [optional] if bunch_size",
		      "was set at constructor, otherwise it is mandatory.",
		    }, 
		  ["smooth_gradients"] = "A smooth gradients boolean [optional]",
		},
		outputs = {
		  "A number with the mean loss of each training step",
		  "A number with the sample variance of the loss",
		} })

function trainable_supervised_trainer_methods:train_dataset(t)
  local params = get_table_fields(
    {
      -- This following commented parameters will be checked by
      -- trainable.dataset_pair_iterator:
      --
      -- shuffle        = { isa_match  = random,   mandatory = false, default=nil },
      -- replacement    = { type_match = "number", mandatory = false, default=nil },
      -- input_dataset  = { mandatory = false, default=nil },
      -- output_dataset = { mandatory = false, default=nil },
      -- distribution   = { type_match="table", mandatory = false, default=nil,
      -- 			 getter = get_table_fields_ipairs{
      -- 			   input_dataset  = { mandatory=true },
      -- 			   output_dataset = { mandatory=true },
      -- 			   probability    = { type_match="number",
      -- 					      mandatory=true },
      -- 			 },
      -- },
      bunch_size     = { type_match = "number",
			 mandatory = (self.bunch_size == false) },
      loss           = { isa_match  = ann.loss,
                         mandatory  = (self.loss_function==false),
			 default=self.loss_function },
      optimizer      = { isa_match  = ann.optimizer,
			 mandatory  = (not self.optimizer),
			 default=self.optimizer },
      smooth_gradients = { type_match = "boolean",
			   mandatory  = false,
			   default=self.smooth_gradients },
    }, t, true)
  local loss       = params.loss
  local optimizer  = params.optimizer
  local smooth_gradients = params.smooth_gradients
  params.loss      = nil
  params.optimizer = nil
  params.smooth_gradients   = nil
  params.bunch_size         = params.bunch_size or self.bunch_size
  params.assert_input_size  = self:get_input_size()
  params.assert_output_size = self:get_output_size()
  -- set to ZERO the accumulated of loss
  loss:reset()
  for input_bunch,output_bunch,bunch_indexes in trainable.dataset_pair_iterator(params) do
    self:train_step(input_bunch, output_bunch, loss, optimizer, #bunch_indexes,
		    smooth_gradients)
  end
  collectgarbage("collect")
  return loss:get_accum_loss()
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.grad_check_dataset", {
		class = "method",
		summary = "Executes gradient check with a given dataset",
		params = {
		  ["input_dataset"]  = "A dataset float or dataset token",
		  ["output_dataset"] = "A dataset float or dataset token (target output)",
		  ["loss"]           = "A loss function. It is [optional] if loss given at constructor",
		  ["bunch_size"]     = 
		    {
		      "Bunch size (mini-batch). It is [optional] if bunch_size",
		      "was set at constructor, otherwise it is mandatory.",
		    },
		  ["max_iterations"] = "Number [optional]",
		  ["verbose"] = "A boolean, true if you want high verbosity [optiona]",
		},
		outputs = {
		  "A boolean",
		} })

function trainable_supervised_trainer_methods:grad_check_dataset(t)
  local params = get_table_fields(
    {
      -- This following commented parameters will be checked by
      -- trainable.dataset_pair_iterator:
      --
      -- input_dataset  = { mandatory = false, default=nil },
      -- output_dataset = { mandatory = false, default=nil },
      bunch_size     = { type_match = "number",
			 mandatory = (self.bunch_size == false) },
      loss           = { isa_match  = ann.loss,
                         mandatory = (self.loss_function==false),
			 default=self.loss_function },
      verbose        = { type_match = "boolean",
			 mandatory = false, default=false },
    }, t, true)
  local loss                = params.loss
  local verbose             = params.verbose
  params.loss               = nil
  params.verbose            = nil
  params.bunch_size         = params.bunch_size or self.bunch_size
  params.assert_input_size  = self:get_input_size()
  params.assert_output_size = self:get_output_size()
  -- set to ZERO the accumulated of loss
  loss:reset()
  local ret = true
  for input,output,bunch_indexes in trainable.dataset_pair_iterator(params) do
    if not self:grad_check_step(input, output, verbose, loss) then
      printf("Error processing pattern bunch: %s\n",
	     table.concat(bunch_indexes, " "))
      ret = false
      break
    end
  end
  collectgarbage("collect")
  return ret
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.validate_dataset", {
		class = "method",
		summary = "Executes one validation epoch with a given dataset",
		description = 
		  {
		    "This method performs one validation epoch with a given",
		    "dataset traversing patterns in order, and returns the",
		    "mean loss of each validate step.",
		    "Each validate step is performed with bunch_size patterns.",
		  }, 
		params = {
		  ["input_dataset"]  = "A dataset float or dataset token",
		  ["output_dataset"] = "A dataset float or dataset token (target output)",
		  ["loss"]           = "A loss function. It is [optional] if loss given at constructor",
		  ["bunch_size"]     = 
		    {
		      "Bunch size (mini-batch). It is [optional] if bunch_size",
		      "was set at constructor, otherwise it is mandatory.",
		    }, 
		},
		outputs = {
		  "A number with the mean loss of each validate step",
		  "A number with the sample variance of the loss",
		} })

april_set_doc("trainable.supervised_trainer.validate_dataset", {
		class = "method",
		summary = "Executes one validation epoch with shuffle",
		description = 
		  {
		    "This method performs one validation epoch with a given",
		    "dataset traversing patterns in shuffle order, and returns the",
		    "mean loss of each validate step.",
		    "Each validate step is performed with bunch_size patterns.",
		  }, 
		params = {
		  ["input_dataset"]  = "A dataset float or dataset token",
		  ["output_dataset"] = "A dataset float or dataset token (target output)",
		  ["shuffle"]        = "A random object used to shuffle patterns before validate",
		  ["loss"]           = "A loss function. It is [optional] if loss given at constructor",
		  ["bunch_size"]     = 
		    {
		      "Bunch size (mini-batch). It is [optional] if bunch_size",
		      "was set at constructor, otherwise it is mandatory.",
		    }, 
		},
		outputs = {
		  "A number with the mean loss of each validate step",
		  "A number with the sample variance of the loss",
		} })

april_set_doc("trainable.supervised_trainer.validate_dataset", {
		class = "method",
		summary = "Executes one stochastic validation epoch with replacement",
		description = 
		  {
		    "This method performs one stochastic validation epoch with a given",
		    "dataset. Patterns are choosed randomly with replacement",
		    "until a given replacement size. The",
		    "mean loss of each validate step is returned.",
		    "Each validate step is performed with bunch_size patterns.",
		  }, 
		params = {
		  ["input_dataset"]  = "A dataset float or dataset token",
		  ["output_dataset"] = "A dataset float or dataset token (target output)",
		  ["shuffle"]        = "A random object used to shuffle patterns before validate",
		  ["replacement"]    = "A number with the size of replacement validate",
		  ["loss"]           = "A loss function. It is [optional] if loss given at constructor",
		  ["bunch_size"]     = 
		    {
		      "Bunch size (mini-batch). It is [optional] if bunch_size",
		      "was set at constructor, otherwise it is mandatory.",
		    }, 
		},
		outputs = {
		  "A number with the mean loss of each validate step",
		  "A number with the sample variance of the loss",
		} })

function trainable_supervised_trainer_methods:validate_dataset(t)
  local params = get_table_fields(
    {
      -- In this case, we check all the given parameters, because all the
      -- dataset iteration schemes are not available for validate_dataset
      input_dataset  = { mandatory = true },
      output_dataset = { mandatory = true },
      bunch_size     = { type_match = "number",
			 mandatory = (self.bunch_size == false) },
      loss           = { isa_match  = ann.loss,
                         mandatory = (self.loss_funcion==false),
			 default=self.loss_function },
      shuffle        = { isa_match  = random, mandatory = false, default=nil },
      replacement    = { type_match = "number", mandatory = false, default=nil },
    }, t)
  -- ERROR CHECKING
  assert(params.input_dataset ~= not params.output_dataset,
	 "input_dataset and output_dataset fields are mandatory together")
  assert(not params.input_dataset or not params.distribution,
	 "input_dataset/output_dataset fields are forbidden with distribution")
  local loss                = params.loss
  params.loss               = nil
  params.bunch_size         = params.bunch_size or self.bunch_size
  params.assert_input_size  = self:get_input_size()
  params.assert_output_size = self:get_output_size()
  -- set to ZERO the accumulated of loss
  loss:reset()
  for input_bunch,output_bunch in trainable.dataset_pair_iterator(params) do
    self:validate_step(input_bunch, output_bunch, loss)
  end
  collectgarbage("collect")
  collectgarbage("collect")
  return loss:get_accum_loss()
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.use_dataset", {
		class = "method",
		summary = "Computes forward with a given dataset "..
		  "provinding output dataset",
		description = 
		  {
		    "This method performs forward with all patterns of the",
		    "given input_dataset, storing outputs at an",
		    "output_dataset with enough space.",
		    "If output_dataset field is given, it must be prepared to",
		    "store all input_dataset:numPatterns().",
		    "If output_dataset field is nil, a new dataset will be",
		    "constructed. The method returns the output_dataset in",
		    "both cases.",
		    "Each forward step is performed with bunch_size patterns.",
		  }, 
		params = {
		  ["input_dataset"]  = "A dataset float or dataset token",
		  ["output_dataset"] = "A dataset float or dataset token [optional].",
		  ["bunch_size"]     = 
		    {
		      "Bunch size (mini-batch). It is [optional] if bunch_size",
		      "was set at constructor, otherwise it is mandatory.",
		    }, 
		},
		outputs = {
		  "The output_dataset with input_dataset:numPatterns().",
		} })

function trainable_supervised_trainer_methods:use_dataset(t)
  local params = get_table_fields(
    {
      -- In this case, we check all the given parameters, because all the
      -- dataset iteration schemes are not available for use_dataset
      input_dataset  = { mandatory = true },
      output_dataset = { mandatory = false, default=nil },
      bunch_size     = { type_match = "number",
			 mandatory = (self.bunch_size == false)  },
    }, t)
  local nump        = params.input_dataset:numPatterns()
  local outsize     = self.ann_component:get_output_size()
  if params.output_dataset then
    if isa(params.output_dataset, dataset) then
      params.output_dataset = dataset.token.wrapper(params.output_dataset)
    end
  elseif isa(params.input_dataset, dataset) then
    params.output_dataset = dataset.matrix(matrix(nump, outsize))
    t.output_dataset      = params.output_dataset
    params.output_dataset = dataset.token.wrapper(params.output_dataset)
  else
    params.output_dataset = dataset.token.vector(outsize)
    t.output_dataset      = params.output_dataset
  end
  if isa(params.input_dataset, dataset) then
    params.input_dataset = dataset.token.wrapper(params.input_dataset)
  end
  local output_dataset        = params.output_dataset
  params.bunch_size           = params.bunch_size or self.bunch_size 
  params.datasets             = { params.input_dataset }
  params.assert_pattern_sizes = { self:get_input_size() }
  params.input_dataset, params.output_dataset = nil, nil
  local ann_component = self.ann_component
  for input_bunch,bunch_indexes in trainable.dataset_multiple_iterator(params) do
    ann_component:reset()
    local output = ann_component:forward(input_bunch)
    output_dataset:putPatternBunch(bunch_indexes,output)
  end  
  return t.output_dataset
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.show_weights", {
		class = "method",
		summary = "Print connection weights (for debug purposes).", })

function trainable_supervised_trainer_methods:show_weights()
  for _,wname in pairs(self.weights_order) do
    local w = self.weights_table[wname]:toTable()
    print(wname, table.concat(w, " "))
  end
end

------------------------------------------------------------------------

april_set_doc("trainable.supervised_trainer.clone",
	      {
		class = "method",
		summary = "Returns a deep-copy of the object.",
		description = {
		  "Returns a deep-copy of the object.",
		  "This method copies user functions on given object instance,",
		  "so user can overwrite any of methods in this class.",
		},
	      })

function trainable_supervised_trainer_methods:clone()
  local obj = trainable.supervised_trainer(self.ann_component:clone(),
					   nil,
					   self.bunch_size,
					   nil)
  if self.loss_function then
    obj:set_loss_function(self.loss_function:clone())
  end
  if self.optimizer then
    obj:set_optimizer(self.optimizer:clone())
  end
  if #self.weights_order > 0 then
    obj:build{ weights = table.map(self.weights_table,
				   function(cnn) return cnn:clone() end) }
  end
  -- add possible user functions
  for i,v in pairs(self) do
    if type(v) == "function" then obj[i] = v end
  end
  return obj
end

april_set_doc("trainable.supervised_trainer.norm2", {
		class = "method",
		summary = "Returns the maximum norm2 of the weights which name matches",
		params={
		  "A connection weights name Lua pattern string [optional], by default it is .*",
		},
		outputs = {
		  "A number with the maximum norm2",
		}, })

function trainable_supervised_trainer_methods:norm2(match_string)
  local norm2 = 0
  for _,cnn in self:iterate_weights(match_string) do
    norm2 = math.max(norm2,
		     reduce(function(a,b)
			      return math.max(a,b:norm2())
			    end, 0, cnn:sliding_window():iterate()))
  end
  return norm2
end

------------------------------------------------------------------------
-- params is a table like this:
-- { training_table   = { input_dataset = ...., output_dataset = ....., .....},
--   validation_table = { input_dataset = ...., output_dataset = ....., .....},
--   validation_function = function( thenet, validation_table ) .... end
--   min_epochs = NUMBER,
--   max_epochs = NUMBER,
--   -- train_params is this table ;)
--   stopping_criterion = function{ current_epoch, best_epoch, best_val_error, train_error, validation_error, train_params } .... return true or false end
--   update_function = function{ current_epoch, best_epoch, best_val_error, 
--   first_epoch = 1
-- }
--
-- and returns this:
--  return { best             = best,
--	     best_epoch       = best_epoch,
--	     best_val_error   = best_val_error,
--	     num_epochs       = epoch,
--	     last_train_error = last_train_error,
--	     last_val_error   = last_val_error }
april_set_doc("trainable.supervised_trainer.train_holdout_validation", {
		class = "method",
		summary = "Trains until convergence with training and "..
		  "validation tables.",
		description = 
		  {
		    "This method trains the component object using",
		    "a training_table checking convergence using a",
		    "validation_table.",
		    "It performs a loop until convergence (stopping criterion)",
		    "running train_dataset(training_table) and",
		    "validate_dataset(validation_table) alternatively.",
		    "Execution of validate_dataset(...) is configurable by",
		    "validation_function parameter. After each epoch",
		    "update_function parameter could be used to do things",
		    "or to show information at screen. It outputs a table.",
		  }, 
		params = {
		  ["training_table"] = "A table for training_dataset method",
		  ["validation_table"] = "A table for validate_dataset method",
		  ["training_function"] = "A function to customize "..
		    "training procedure [optional]. By default it calls "..
		    "train_dataset(training_table) and uses same loss "..
		    "function to train and validate. The function is called "..
		    "as train_function(trainer, training_table). It must" ..
		    "return the loss of training epoch",
		  ["validation_function"] = "A function to customize "..
		    "validation procedure [optional]. By default it calls "..
		    "validate_dataset(validation_table) and uses same loss "..
		    "function to train and validate. The function is called "..
		    "as validation_function(trainer, validation_table)",
		  ["best_function"] = "A function to customize code execution "..
		    "when validation loss is improved [optional]. It is "..
		    "called as best_function(best_trainer_clone,best_error,best_epoch)",
		  ["epochs_wo_validation"] = {
		    "Number of epochs without taking into account validation",
		    "error [optional]. By default is 0",
		  },
		  ["min_epochs"] = "Minimum number of epochs",
		  ["max_epochs"] = "Maximum number of epochs",
		  ["stopping_criterion"] = "A predicate function which "..
		  "returns true if stopping criterion, false otherwise. "..
		    "Some basic criteria are implemented at "..
		    "trainable.stopping_criteria table."..
		    "The criterion function is called as "..
		    "stopping_criterion({ current_epoch=..., best_epoch=..., "..
		    "best_val_error=..., train_error=..., "..
		    "validation_error=..., train_params=THIS_PARAMS }).",
		  ["update_function"] = "Function executed after each "..
		  "epoch (after training and validation), for print "..
		    "purposes (and other stuff). It is called as "..
		    "update_function({ current_epoch=..., "..
		    "best_epoch=..., best_val_error=..., "..
		    "train_error=..., validation_error=..., "..
		    "cpu=CPU_TIME_LAST_EPCOCH, "..
		    "wall=CPU_WALL_TIME_LAST_EPOCH, "..
		    "train_params=THIS_PARAMS}) [optional].",
		  ["first_epoch"] = "Useful to rerun previous stopped "..
		    "experiments",
		},
		outputs = {
		  ["best"] = "The best validation loss trainer clone",
		  ["best_val_error"] = "The best validation loss",
		  ["best_epoch"] = "The best epoch",
		  ["last_epoch"] = "Number of epochs performed",
		  ["last_train_error"] = "The training loss achieved at last epoch",
		  ["last_val_error"] = "The validation loss achieved at last epoch",
		}, })

function trainable_supervised_trainer_methods:train_holdout_validation(t)
  print("DEPRECATED: use the class trainable.train_holdout_validation")
  local params = get_table_fields(
    {
      training_table   = { mandatory=true, type_match="table" },
      training_function = { mandatory=false, type_match="function",
			      default=function(trainer,t)
				return trainer:train_dataset(t)
			      end },
      validation_table = { mandatory=true, type_match="table" },
      validation_function = { mandatory=false, type_match="function",
			      default=function(trainer,t)
				return trainer:validate_dataset(t)
			      end },
      best_function = { mandatory=false, type_match="function",
			default=function(trainer,t) end },
      epochs_wo_validation = { mandatory=false, type_match="number", default=0 },
      min_epochs = { mandatory=true, type_match="number" },
      max_epochs = { mandatory=true, type_match="number" },
      stopping_criterion = { mandatory=true, type_match="function" },
      update_function    = { mandatory=false, type_match="function",
			     default=function(t) return end },
      first_epoch        = { mandatory=false, type_match="number", default=1 },
    }, t)
  local best_epoch       = params.first_epoch-1
  local best             = self:clone()
  local best_val_error   = params.validation_function(self,
						      params.validation_table)
  local last_val_error   = best_val_error
  local last_train_error = 0
  local last_epoch       = 0
  local clock            = util.stopwatch()
  for epoch=params.first_epoch,params.max_epochs do
    collectgarbage("collect")
    clock:reset()
    clock:go()
    local tr_error  = params.training_function(self, params.training_table)
    local val_error = params.validation_function(self, params.validation_table)
    last_train_error,last_val_error,last_epoch = tr_error,val_error,epoch
    clock:stop()
    cpu,wall = clock:read()
    if val_error < best_val_error or epoch <= params.epochs_wo_validation then
      best_epoch     = epoch
      best_val_error = val_error
      best           = self:clone()
      params.best_function(best, best_val_error, best_epoch)
    end
    params.update_function({ current_epoch    = epoch,
			     best_epoch       = best_epoch,
			     best_val_error   = best_val_error,
			     train_error      = tr_error,
			     validation_error = val_error,
			     cpu              = cpu,
			     wall             = wall,
			     train_params     = params })
    if (epoch > params.min_epochs and
	  params.stopping_criterion({ current_epoch    = epoch,
				      best_epoch       = best_epoch,
				      best_val_error   = best_val_error,
				      train_error      = tr_error,
				      validation_error = val_error,
				      train_params     = params })) then
      break						  
    end
  end
  return { best             = best,
	   best_val_error   = best_val_error,
	   best_epoch       = best_epoch,
	   last_epoch       = last_epoch,
	   last_train_error = last_train_error,
	   last_val_error   = last_val_error }
end

---------------------------------------------------------------------------
-- This function trains without validation, it is trained until a maximum of
-- epochs or until the improvement in training error is less than given
-- percentage
--
-- params is a table like this:
-- { training_table   = { input_dataset = ...., output_dataset = ....., .....},
--   min_epochs = NUMBER,
--   max_epochs = NUMBER,
--   update_function = function{ current_epoch, best_epoch, best_val_error,
--   percentage_stopping_criterion = NUMBER (normally 0.01 or 0.001)
-- }
--
-- returns the trained object
april_set_doc("trainable.supervised_trainer.train_wo_validation", {
		class = "method",
		summary = "Trains until a given convergence of training loss, "..
		  "using a given training_table.",
		description = 
		  {
		    "This method trains the component object using",
		    "a training_table checking convergence based on training",
		    "loss.",
		    "It performs a loop until convergence (stopping criterion)",
		    "running train_dataset(training_table).",
		    "After each epoch",
		    "update_function parameter could be used to do things",
		    "or to show information at screen. It outputs a",
		    "trainable.supervised_trainer object.",
		  }, 
		params = {
		  ["training_table"] = "A table for training_dataset method",
		  ["training_function"] = "A function to customize "..
		    "training procedure [optional]. By default it calls "..
		    "train_dataset(training_table). The function is called "..
		    "as train_function(trainer, training_table). It must" ..
		    "return the loss of training epoch",
		  ["min_epochs"] = "Minimum number of epochs",
		  ["max_epochs"] = "Maximum number of epochs",
		  ["percentage_stopping_criterion"] = "A percentage of "..
		    "training loss improvement under which convergence is "..
		    "achieved.",
		  ["update_function"] = "Function executed after each "..
		  "epoch (after training and validation), for print "..
		    "purposes (and other stuff). It is called as "..
		    "update_function({ train_error=..., "..
		    "train_improvement=..., "..
		    "train_params=THIS_PARAMS}) [optional].",
		},
		outputs = {
		  "A trainable.supervised_trainer object"
		}, })

april_set_doc("trainable.supervised_trainer.train_wo_validation", {
		class = "method",
		summary = "Trains until a given convergence of training loss, "..
		  "using a given training_table functor.",
		description = 
		  {
		    "This method trains the component object using",
		    "a training_table checking convergence based on training",
		    "loss. The training_table is a function which after",
		    "it is called, it returns a table for",
		    "training_dataset(...) method, so it is customizable.",
		    "It performs a loop until convergence (stopping criterion)",
		    "running train_dataset(training_table).",
		    "After each epoch",
		    "update_function parameter could be used to do things",
		    "or to show information at screen. It outputs a",
		    "trainable.supervised_trainer object.",
		  }, 
		params = {
		  ["training_table"] = "A function which returns a "..
		    "train_dataset compatible table",
		  ["training_function"] = "A function to customize "..
		    "training procedure [optional]. By default it calls "..
		    "train_dataset(training_table). The function is called "..
		    "as train_function(trainer, training_table). It must" ..
		    "return the loss of training epoch",
		  ["min_epochs"] = "Minimum number of epochs",
		  ["max_epochs"] = "Maximum number of epochs",
		  ["percentage_stopping_criterion"] = "A percentage of "..
		    "training loss improvement under which convergence is "..
		    "achieved.",
		  ["update_function"] = "Function executed after each "..
		  "epoch (after training and validation), for print "..
		    "purposes (and other stuff). It is called as "..
		    "update_function({ current_epoch=..., "..
		    "train_error=..., "..
		    "train_improvement=..., "..
		    "train_params=THIS_PARAMS}) [optional].",
		},
		outputs = {
		  "A trainable.supervised_trainer object"
		}, })

function trainable_supervised_trainer_methods:train_wo_validation(t)
  print("DEPRECATED: use the class trainable.train_wo_validation")
  local params = get_table_fields(
    {
      training_table = { mandatory=true },
      training_function = { mandatory=false, type_match="function",
			      default=function(trainer,t)
				return trainer:train_dataset(t)
			      end },
      min_epochs = { mandatory=true, type_match="number" },
      max_epochs = { mandatory=true, type_match="number" },
      update_function = { mandatory=false, type_match="function",
			  default=function(t) return end },
      percentage_stopping_criterion = { mandatory=false, type_match="number",
					default=0.01 },
    }, t)
  local prev_tr_err = 1111111111111
  local best        = self:clone()
  for epoch=1,params.max_epochs do
    local tr_table = params.training_table
    if type(tr_table) == "function" then tr_table = tr_table() end
    collectgarbage("collect")
    local tr_err        = params.training_function(self, tr_table)
    local tr_improvement = (prev_tr_err - tr_err)/prev_tr_err
    if (epoch > params.min_epochs and
	tr_improvement < params.percentage_stopping_criterion) then
      break
    end
    best = self:clone()
    params.update_function{ current_epoch     = epoch,
			    train_error       = tr_err,
			    train_improvement = tr_improvement,
			    train_params      = params }
    prev_tr_err = tr_err
  end
  return best
end

-------------------------------------------------------------------------------

--------------------------------
-- DATASET_ITERATOR FUNCTIONS --
--------------------------------

april_set_doc("trainable.dataset_pair_iterator", {
		class       = "function",
		summary     = {
		  "A function which returns an iterator which",
		  "does a dataset pair loop (input_dataset and output_dataset)"
		},
		description ={
		  "This class is useful to implement a loop over a dataset.",
		  "It implements different loop schemes, in all the cases",
		  "returning a bunch_size patterns.",
		  "It admits the following traversals: sequential, shuffled,",
		  "shuffled with replacement, shuffled with distribution.",
		}, })

local TOO_LARGE_NUMPATTERNS = 4000000
function trainable.dataset_pair_iterator(t)
  local params = get_table_fields(
    {
      input_dataset  = { mandatory = false, default=nil },
      output_dataset = { mandatory = false, default=nil },
      distribution   = { type_match="table", mandatory = false, default=nil,
			 getter = get_table_fields_ipairs{
			   input_dataset  = { mandatory=true },
			   output_dataset = { mandatory=true },
			   probability    = { type_match="number",
					      mandatory=true },
			 },
      },
      bunch_size     = { type_match = "number", mandatory = true },
      shuffle        = { isa_match  = random,   mandatory = false, default=nil },
      replacement    = { type_match = "number", mandatory = false, default=nil },
      assert_input_size = { type_match = "number", mandatory = false, default=0 },
      assert_output_size = { type_match = "number", mandatory = false, default=0 },
    }, t)
  -- ERROR CHECKING
  assert(params.input_dataset ~= not params.output_dataset,
	 "input_dataset and output_dataset fields are mandatory together")
  assert(not params.input_dataset or not params.distribution,
	 "input_dataset/output_dataset fields are forbidden with distribution")
  assert(params.assert_input_size ~= not params.assert_output_size,
	 "assert_input_size and assert_output_size are mandatory together")
  if params.input_dataset then
    params.datasets = { params.input_dataset, params.output_dataset }
    params.input_dataset, params.output_dataset = nil
  end
  if params.distribution then
    iterator(ipairs(params.distribution)):
    apply(function(_,data)
	    data.datasets = { data.input_dataset, data.output_dataset }
	    data.input_dataset, data.output_dataset = nil, nil
	  end)
  end
  if params.assert_input_size then
    params.assert_pattern_sizes = { params.assert_input_size,
				    params.assert_output_size }
    params.assert_input_size, params.assert_output_size = nil, nil
  end
  return trainable.dataset_multiple_iterator(params)
end

----------------------------

april_set_doc("trainable.dataset_multiple_iterator", {
		class       = "function",
		summary     = {
		  "A function which returns an iterator which",
		  "does a dataset loop through multiple datasets"
		},
		description ={
		  "This function is useful to implement a loop over a dataset.",
		  "It implements different loop schemes, in all the cases",
		  "returning a token with bunch_size patterns.",
		  "It admits the following traversals: sequential, shuffled,",
		  "shuffled with replacement, shuffled with distribution.",
		}, })

local TOO_LARGE_NUMPATTERNS = 4000000
function trainable.dataset_multiple_iterator(t)
  local params = get_table_fields(
    {
      datasets       = { mandatory = false, default=nil },
      distribution   = { type_match="table", mandatory = false, default=nil,
			 getter = get_table_fields_ipairs{
			   datasets    = { mandatory=true },
			   probability = { type_match="number",
					   mandatory=true },
			 },
      },
      bunch_size     = { type_match = "number", mandatory = true },
      shuffle        = { isa_match  = random,   mandatory = false, default=nil },
      replacement    = { type_match = "number", mandatory = false, default=nil },
      bunch_major    = { type_match = "string", mandatory = false, default="row_major" },
      assert_pattern_sizes = { type_match = "table", mandatory = false,
			       default={ } },
    }, t)
  -- ERROR CHECKING
  assert(not params.datasets or not params.distribution,
	 "datasets field is forbidden with distribution")
  assert(params.datasets or params.distribution,
	 "datasets or distribution are needed")
  --
  local bunch_size = params.bunch_size
  --
  local to_dataset_token = function(ds_table)
    local nump
    return iterator(ipairs(ds_table)):
    map(function(_,ds)
	  if nump then
	    assert(nump == ds:numPatterns(),
		   string.format("Incorrect number of patterns, expected "..
				   "%d, found %d", nump, ds:numPatterns()))
	  else nump = ds:numPatterns()
	  end
	  return isa(ds,dataset) and dataset.token.wrapper(ds,
							   params.bunch_major) or ds
	end):
    table(), nump
  end
  -- TRAINING TABLES
  
  -- for each pattern, index in dataset
  local ds_idx_table = {}
  if params.distribution then
    -- Training with distribution: given a table of datasets the patterns are
    -- sampled following the given apriory probability
    assert(params.shuffle,"shuffle is mandatory with distribution")
    assert(params.replacement,"replacement is mandatory with distribution")
    params.datasets = { }
    local aprioris = {}
    local sizes    = {}
    local sums     = { 0 }
    local num_ds
    --
    params.datasets = iterator(ipairs(params.distribution.datasets)):
    map(function() return dataset.token.union() end):
    table()
    --
    for i,v in ipairs(params.distribution) do
      local nump
      if num_ds then
	assert(num_ds == #v.datasets,
	       string.format("Incorrect number of datasets, expected %d, found %d",
			     num_ds, #v.datasets))
      else num_ds = #v.datasets
      end
      v.datasets = to_dataset_token(v.datasets)
      table.insert(aprioris, v.probability)
      table.insert(sizes, nump)
      table.insert(sums, sums[#sums] + nump)
      --
      iterator(ipairs(params.datasets)):
      apply(function(k,union_ds) union_ds:push_back(v.datasets[k]) end)
    end -- for i,v in ipairs(params.distribution)
    -- generate training tables
    local dice = random.dice(aprioris)
    for i=1,params.replacement do
      local whichclass=dice:thrown(params.shuffle)
      local idx=params.shuffle:randInt(1,sizes[whichclass])
      table.insert(ds_idx_table, idx + sums[whichclass])
    end
  else -- if params.distribution then ... else
    --
    local num_patterns
    params.datasets,num_patterns = to_dataset_token(params.datasets)
    assert(num_patterns < TOO_LARGE_NUMPATTERNS or params.replacement,
	   "The number of patterns is too large, use replacement instead")
    -- generate training tables depending on training mode (replacement,
    -- shuffled, or sequential)
    if params.replacement then
      assert(params.shuffle,"shuffle is mandatory with replacement")
      assert(params.replacement < TOO_LARGE_NUMPATTERNS, 
	     "The number of patterns is too large, use a shorter replacement")
      for i=1,params.replacement do
	table.insert(ds_idx_table, params.shuffle:randInt(1,num_patterns))
      end
    elseif params.shuffle then
      ds_idx_table = params.shuffle:shuffle(num_patterns)
    else
      for i=1,num_patterns do table.insert(ds_idx_table, i) end
    end
  end
  -- SANITY CHECK
  iterator(ipairs(params.assert_pattern_sizes)):
  apply(function(k,psize)
	  local ds_psize = params.datasets[k]:patternSize()
	  assert(psize == 0 or psize == ds_psize,
		 string.format("Incorrect patternSize at dataset %d, found %d"..
				 ", expected %d", k, ds_psize, psize))
	end)
  --
  -- ITERATOR USING ds_idx_table
  local k=0
  local bunch_indexes = {}
  local i=1 -- pattern index
  return function()
    if i > #ds_idx_table then
      ds_idx_table=nil
      collectgarbage("collect")
      return nil
    end
    table.clear(bunch_indexes)
    local last = math.min(i+bunch_size-1, #ds_idx_table)
    for j=i,last do table.insert(bunch_indexes, ds_idx_table[j]) end
    local data = iterator(ipairs(params.datasets)):select(2):
    call('getPatternBunch', bunch_indexes):
    table()
    table.insert(data, bunch_indexes)
    k=k+1
    i=i+#bunch_indexes
    if k == MAX_ITERS_WO_COLLECT_GARBAGE then collectgarbage("collect") k=0 end
    return table.unpack(data)
  end
end

-------------------------------------------------------------------------------

------------------------------------
-- TRAIN_HOLDOUT_VALIDATION CLASS --
------------------------------------

april_set_doc("trainable.train_holdout_validation", {
		class       = "class",
		summary     = "Training class using holdout validation",
		description ={
		  "This training class defines a train_func which",
		  "follows a training schedule based on validation error or",
		  "in number of epochs. Method execute receives a function",
		  "which trains one epoch and returns the trainer object,",
		  "the training loss and the validation loss. This method",
		  "returns true in case the training continues, or false if",
		  "the stop criterion is true.",
		}, })

local train_holdout_methods, train_holdout_class_metatable =
  class("trainable.train_holdout_validation")

april_set_doc("trainable.train_holdout_validation.__call", {
		class = "method", summary = "Constructor",
		description ={
		  "Constructor of the train_holdout_validation class.",
		},
		params = {
		  epochs_wo_validation = {
		    "Number of epochs from start where the validation is",
		    "ignored [optional], by default it is 0",
		  },
		  min_epochs = "Minimum number of epochs for training [optional]. By default it is 0",
		  max_epochs = "Maximum number of epochs for training",
		  stopping_criterion = {
		    "A predicate function which",
		    "returns true if stopping criterion, false otherwise.",
		    "Some basic criteria are implemented at",
		    "trainable.stopping_criteria table.",
		    "The criterion function is called as",
		    "stopping_criterion({ current_epoch=..., best_epoch=...,",
		    "best_val_error=..., train_error=...,",
		    "validation_error=... }). [optional] By default it is max_epochs criterion.",
		  },
		  tolerance = {
		    "The tolerance>=0 is the minimum relative difference to",
		    "take the current validation loss as the best [optional],",
		    "by default it is 0.",
		  },
		  first_epoch = "The first epoch number [optional]. By default it is 0.",
		},
		outputs = { "Instantiated object" }, })

function train_holdout_class_metatable:__call(t,saved_state)
  local params = get_table_fields(
    {
      epochs_wo_validation = { mandatory=false, type_match="number", default=0 },
      min_epochs = { mandatory=true, type_match="number", default=0 },
      max_epochs = { mandatory=true, type_match="number" },
      tolerance  = { mandatory=true, type_match="number", default=0 },
      stopping_criterion = { mandatory=true, type_match="function",
			     default = function() return false end },
      first_epoch        = { mandatory=false, type_match="number", default=1 },
    }, t)
  assert(params.tolerance >= 0, "tolerance < 0 is forbidden")
  local saved_state = saved_state or {}
  local obj = {
    params = params,
    state  = {
      current_epoch    = saved_state.current_epoch    or 0,
      train_error      = saved_state.train_error      or math.huge,
      validation_error = saved_state.validation_error or math.huge,
      best_epoch       = saved_state.best_epoch       or 0,
      best_val_error   = saved_state.best_val_error   or math.huge,
      best             = saved_state.best             or nil,
      last             = saved_state.last             or nil,
    },
  }
  return class_instance(obj, self)
end

april_set_doc("trainable.train_holdout_validation.execute", {
		class = "method", summary = "Runs one training epoch",
		description ={
		  "This method executes one training epoch. It receives an",
		  "epoch function which where the user trains the model,",
		  "and which returns the trained model, the train loss, and the",
		  "validation loss.",
		},
		params = {
		  {
		    "A function which trains a model and returns the trained model,",
		    "the training loss and the validation loss",
		  },
		  first_epoch = "The first epoch number",
		},
		outputs = { "True or false, indicating if the training continues or not" }, })

function train_holdout_methods:execute(epoch_function)
  local params = self.params
  local state  = self.state
  -- check max epochs
  if state.current_epoch >= params.max_epochs then
    return false
  end
  -- check stopping criterion
  if ( state.current_epoch > params.min_epochs and
       params.stopping_criterion(state) ) then
    return false
  end
  -- compute one training step by using epoch_function
  state.current_epoch = state.current_epoch + 1
  collectgarbage("collect")
  state.last, state.train_error, state.validation_error = epoch_function()
  assert(state.last and state.train_error and state.validation_error,
	 "Needs a function which returns three values: "..
	   "a model, training error and validation error")
  -- update with the best model
  if ( state.validation_error < state.best_val_error or
       state.current_epoch <= params.epochs_wo_validation ) then
    local abs_error = state.best_val_error - state.validation_error
    local rel_error = abs_error / state.best_val_error
    if state.best_val_error == math.huge or rel_error > params.tolerance then
      state.best_epoch     = state.current_epoch
      state.best_val_error = state.validation_error
      state.best           = state.last:clone()
    end
  end
  return true
end

april_set_doc("trainable.train_holdout_validation.set_param", {
		class = "method",
		summary =
		  "Modifies one parameter of which was given at construction",
		params = {
		  "The parameter name",
		  "The parameter value",
		}, })

function train_holdout_methods:set_param(name,value)
  assert(self.params[name], "Param  " .. name .. " not found")
  self.params[name] = value
end

april_set_doc("trainable.train_holdout_validation.get_param", {
		class = "method",
		summary =
		  "Returns the value of a param",
		params = {
		  "The parameter name",
		},
		outputs = { "The paremter value" }, })

function train_holdout_methods:get_param(name)
  return self.params[name]
end

april_set_doc("trainable.train_holdout_validation.get_state", {
		class = "method",
		summary =
		  "Returns the state of the training",
		outputs = {
		  "Current epoch",
		  "Train loss",
		  "Validation loss",
		  "Best epoch",
		  "Best epoch validation loss",
		  "Best trained model",
		  "Last trained model",
		}, })

function train_holdout_methods:get_state()
  local state = self.state
  return state.current_epoch, state.train_error, state.validation_error,
  state.best_epoch, state.best_val_error, state.best, state.last
end

april_set_doc("trainable.train_holdout_validation.get_state_table", {
		class = "method",
		summary =
		  "Returns the state table of the training",
		outputs = {
		  current_epoch = "Current epoch",
		  train_error = "Train loss",
		  validation_error = "Validation loss",
		  best_epoch = "Best epoch",
		  best_val_error = "Best epoch validation loss",
		  best = "Best trained model",
		  last = "Last trained model",
		}, })

function train_holdout_methods:get_state_table()
  return self.state
end

april_set_doc("trainable.train_holdout_validation.get_state_string", {
		class = "method",
		summary =
		"Returns the state of the training in string format, for printing",
		outputs = {
		  { "A string with the format ('%5d %.6f %.6f    %5d %.6f',",
		    "current_epoch,train_error,validation_error,best_epoch,best_val_error)" }
		}, })

function train_holdout_methods:get_state_string()
  local state = self.state
  return string.format("%5d %.6f %.6f    %5d %.6f",
		       state.current_epoch,
		       state.train_error,
		       state.validation_error,
		       state.best_epoch,
		       state.best_val_error)
end

function train_holdout_methods:to_lua_string(format)
  local t = { }
  table.insert(t, "trainable.train_holdout_validation(")
  --
  table.insert(t, "\n\t")
  table.insert(t, table.tostring(self.params, format))
  table.insert(t, ",")
  table.insert(t, "\n\t")
  table.insert(t, table.tostring(self.state, format))
  table.insert(t, "\n)")
  return table.concat(t, "")
end

april_set_doc("trainable.train_holdout_validation.save", {
		class = "method",
		summary = "Saves the training in a filename",
		description = {
		  "Saves the training in a filename.",
		  "If the filename exists, it is renamed as filename.bak",
		},
		params={
		  "The filename",
		  { "The format for matrix data ('ascii' or 'binary'),",
		    "[optional] by default 'binary'", },
		  {
		    "Extra data dictionary (a table) [optional].",
		    "It is useful to store random objects.",
		    "The serialization of this objects is automatic",
		    "if they has a 'to_lua_string(format)' method, or",
		    "if they are Lua standard types (number, string, table).",
		  },
		}, })

function train_holdout_methods:save(filename,format,extra)
  assert(format==nil or luatype(format)=="string",
	 "Second argument is a string with the format: 'binary' or 'ascii'")
  local f = io.open(filename, "r")
  if f then
    f:close()
    os.execute(string.format("mv -f %s %s.bak", filename, filename))
  end
  local f = io.open(filename, "w") or error("Unable to open " .. filename)
  f:write("return ")
  f:write(self:to_lua_string(format))
  if extra then
    f:write(",\n")
    f:write(table.tostring(extra, format))
  end
  f:write("\n")
  f:close()
end

april_set_doc("trainable.train_holdout_validation.load", {
		class = "method",
		summary = "Loads the training from a filename",
		params={
		  "The filename",
		},
		outputs = {
		  "A train_holdout_methods instance",
		  "A table with extra saved data or nil if not given when saving",
		}, })

function trainable.train_holdout_validation.load(filename)
  local f = loadfile(filename) or error("Unable to open " .. filename)
  local obj,extra = f()
  assert(obj, "Impossible to load chunk from file " .. filename)
  return obj,extra
end

-------------------------------------------------------------------------------

------------------------------------
-- TRAIN_HOLDOUT_VALIDATION CLASS --
------------------------------------

april_set_doc("trainable.train_wo_validation", {
		class       = "class",
		summary     = "Training class without validation",
		description ={
		  "This training class defines a train_func which",
		  "follows a training schedule based on training error or",
		  "in number of epochs. Method execute receives a function",
		  "which trains one epoch and returns the trainer object and",
		  "the training loss. This method",
		  "returns true in case the training continues, or false if",
		  "the stop criterion is true.",
		}, })

local train_wo_validation_methods, train_wo_validation_class_metatable =
  class("trainable.train_wo_validation")

april_set_doc("trainable.train_wo_validation.__call", {
		class = "method", summary = "Constructor",
		description ={
		  "Constructor of the train_wo_validation class.",
		},
		params = {
		  min_epochs = "Minimum number of epochs for training [optional]. By default it is 0,",
		  max_epochs = "Maximum number of epochs for training",
		  percentage_stopping_criterion = "A number [optional]. By default it is 0.01",
		  first_epoch = "The first epoch number [optional]. By default it is 1",
		},
		outputs = { "Instantiated object" }, })

function train_wo_validation_class_metatable:__call(t,saved_state)
  local params = get_table_fields(
    {
      min_epochs = { mandatory=true, type_match="number", default=0 },
      max_epochs = { mandatory=true, type_match="number" },
      percentage_stopping_criterion = { mandatory=true, type_match="number",
					default = 0.01 },
      first_epoch        = { mandatory=false, type_match="number", default=1 },
    }, t)
  local saved_state = saved_state or {}
  local obj = {
    params = params,
    state  = {
      current_epoch     = saved_state.current_epoch     or 0,
      train_error       = saved_state.train_error       or math.huge,
      train_improvement = saved_state.train_improvement or math.huge,
      last              = saved_state.last              or nil,
    },
  }
  return class_instance(obj, self)
end

april_set_doc("trainable.train_wo_validation.execute", {
		class = "method", summary = "Runs one training epoch",
		description ={
		  "This method executes one training epoch. It receives an",
		  "epoch function which where the user trains the model,",
		  "and which returns the trained model and the train loss.",
		},
		params = {
		  {
		    "A function which trains a model and returns the trained model",
		    "and the training loss",
		  },
		  first_epoch = "The first epoch number",
		},
		outputs = { "True or false, indicating if the training continues or not" }, })

function train_wo_validation_methods:execute(epoch_function)
  local params = self.params
  local state  = self.state
  -- stopping criterion
  if (state.current_epoch > params.min_epochs and
      state.train_improvement < params.percentage_stopping_criterion) then
    return false
  end
  -- check max epochs
  if state.current_epoch >= params.max_epochs then
    return false
  end
  -- compute one training step by using epoch_function
  state.current_epoch = state.current_epoch + 1
  collectgarbage("collect")
  local model,tr_err = epoch_function()
  assert(model and tr_err,
	 "Needs a function which returns two values: "..
	   "a model and training error")
  --
  local prev_tr_err    = state.train_error
  local tr_improvement
  if state.current_epoch > 1 then
    tr_improvement = (prev_tr_err - tr_err)/prev_tr_err
  else
    tr_improvement = math.huge
  end
  
  state.train_error       = tr_err
  state.train_improvement = tr_improvement
  state.last              = model
  
  return true
end

april_set_doc("trainable.train_wo_validation.set_param", {
		class = "method",
		summary =
		  "Modifies one parameter of which was given at construction",
		params = {
		  "The parameter name",
		  "The parameter value",
		}, })

function train_wo_validation_methods:set_param(name,value)
  assert(self.params[name], "Param  " .. name .. " not found")
  self.params[name] = value
end

april_set_doc("trainable.train_wo_validation.get_param", {
		class = "method",
		summary =
		  "Returns the value of a param",
		params = {
		  "The parameter name",
		},
		outputs = { "The paremter value" }, })

function train_wo_validation_methods:get_param(name)
  return self.params[name]
end

april_set_doc("trainable.train_wo_validation.get_state", {
		class = "method",
		summary =
		  "Returns the state of the training",
		outputs = {
		  "Current epoch",
		  "Train loss",
		  "Train relative improvement",
		  "Last trained model",
		}, })

function train_wo_validation_methods:get_state()
  local state = self.state
  return state.current_epoch, state.train_error,
  state.train_improvement,state.last
end

april_set_doc("trainable.train_wo_validation.get_state_table", {
		class = "method",
		summary =
		  "Returns the state table of the training",
		outputs = {
		  current_epoch = "Current epoch",
		  train_error = "Train loss",
		  train_improvement = "Train relative improvement",
		  last = "Last trained model",
		}, })

function train_wo_validation_methods:get_state_table()
  return self.state
end

april_set_doc("trainable.train_wo_validation.get_state_string", {
		class = "method",
		summary =
		"Returns the state of the training in string format, for printing",
		outputs = {
		  { "A string with the format ('%5d %.6f    %.6f',",
		    "current_epoch,train_error,train_improvement)" }
		}, })

function train_wo_validation_methods:get_state_string()
  local state = self.state
  return string.format("%5d %.6f    %.6f",
		       state.current_epoch,
		       state.train_error,
		       state.train_improvement)
end

function train_wo_validation_methods:to_lua_string(format)
  local t = { }
  table.insert(t, "trainable.train_wo_validation(")
  --
  table.insert(t, "\n\t")
  table.insert(t, table.tostring(self.params, format))
  table.insert(t, ",")
  table.insert(t, "\n\t")
  table.insert(t, table.tostring(self.state, format))
  table.insert(t, "\n)")
  return table.concat(t, "")
end

april_set_doc("trainable.train_wo_validation.save", {
		class = "method",
		summary = "Saves the training in a filename",
		description = {
		  "Saves the training in a filename.",
		  "If the filename exists, it is renamed as filename.bak",
		},
		params={
		  "The filename",
		  { "The format for matrix data ('ascii' or 'binary'),",
		    "[optional] by default 'binary'", },
		  {
		    "Extra data dictionary (a table) [optional].",
		    "It is useful to store random objects.",
		    "The serialization of this objects is automatic",
		    "if they has a 'to_lua_string(format)' method, or",
		    "if they are Lua standard types (number, string, table).",
		  },
		}, })
train_wo_validation_methods.save = train_holdout_methods.save

april_set_doc("trainable.train_wo_validation.load", {
		class = "method",
		summary = "Loads the training from a filename",
		params={
		  "The filename",
		},
		outputs = {
		  "A train_wo_validation_methods instance",
		  "A table with extra saved data or nil if not given when saving",
		}, })
trainable.train_wo_validation.load = trainable.train_holdout_validation.load

-------------------------------------------------------------------------------

-------------------------
-- STOPPING CRITERIA --
-------------------------
april_set_doc("trainable.stopping_criteria", {
		class       = "namespace",
		summary     = "Table with built-in stopping criteria", })

trainable.stopping_criteria = trainable.stopping_criteria or {}

--------------------------------------------------------------------------

april_set_doc("trainable.stopping_criteria.make_max_epochs_wo_imp_absolute", {
		class       = "function",
		summary     = "Returns a stopping criterion based on absolute loss.",
		description = 
		  {
		    "This function returns a stopping criterion function",
		    "which returns is true if current_epoch - best_epoch >= abs_max."
		  }, 
		params = { "Absolute maximum difference (abs_max)" },
		outputs = { "A stopping criterion function" }, })

function trainable.stopping_criteria.make_max_epochs_wo_imp_absolute(abs_max)
  local f = function(params)
    return (params.current_epoch - params.best_epoch) >= abs_max
  end
  return f
end

--------------------------------------------------------------------------

april_set_doc("trainable.stopping_criteria.make_max_epochs_wo_imp_relative", {
		class       = "function",
		summary     = "Returns a stopping criterion based on relative loss.",
		description = 
		  {
		    "This function returns a stopping criterion function",
		    "which returns is true if not",
		    "current_epoch/best_epoch < rel_max."
		  }, 
		params = { "Relative maximum difference (rel_max)" },
		outputs = { "A stopping criterion function" }, })

function trainable.stopping_criteria.make_max_epochs_wo_imp_relative(rel_max)
  local f = function(params)
    return not (params.current_epoch/params.best_epoch < rel_max)
  end
  return f
end
