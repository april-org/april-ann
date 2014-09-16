-------------------------------
-- LOCAL AUXILIARY FUNCTIONS --
-------------------------------

local math = math
local table = table
local string = string
--
local ipairs = ipairs
local pairs = pairs
local assert = assert
--
local type = type
local is_a = class.is_a
local iterator = iterator
local get_table_fields = get_table_fields
local april_assert = april_assert

-----------------------------------------

local wrap_matrices = matrix.dict.wrap_matrices

------------------------------
-- SUPERVISED_TRAINER CLASS --
------------------------------

local trainable_supervised_trainer,trainable_supervised_trainer_methods =
  class("trainable.supervised_trainer")
trainable = trainable or {} -- global environment
trainable.supervised_trainer = trainable_supervised_trainer -- global environment

april_set_doc(trainable.supervised_trainer, {
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

------------------------------------------------------------------------

trainable_supervised_trainer.constructor =
  april_doc{
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
    outputs = { "Instantiated object" },
  } ..
  function (self, ...)
    if select('#',...) == 1 and type(select(1,...)) == "table" then
      -- Loading from a previously saved object, the first argument is a table
      local t = select(1,...)
      local model = t.model
      local connections = t.connections
      local loss        = t.loss
      local bunch_size  = t.bunch_size
      local optimizer   = t.optimizer
      local smooth_gradients = t.smooth_gradients or true
      trainable.supervised_trainer.constructor(self,
                                               model, loss, bunch_size,
                                               optimizer, smooth_gradients)
      local weight = connections
      if not is_a(connections, matrix.dict) then
        weights = matrix.dict()
        for name,wdata in pairs(connections) do
          local m = wdata
          if not is_a(m, matrix) then
            m = wdata.w:rewrap(wdata.output,wdata.input)
          end
          if m:get_major_order() == "col_major" then
            weights:insert(name,m)
          else
            weights:insert(name,m:clone("col_major"))
          end
        end
      end
      self:build{ weights = weights }
    else
      -- Constructor of a new object
      local ann_component,loss_function,bunch_size,optimizer,smooth_gradients = ...
      local optimizer = optimizer or ann.optimizer.sgd()
      local smooth_gradients = smooth_gradients or true
      if loss_function and not is_a(loss_function, ann.loss) then
        error("The second parameter must be an instance of ann.loss")
      end
      if optimizer and not is_a(optimizer, ann.optimizer) then
        error("The fourth parameter must be an instance of ann.optimizer")
      end
      if bunch_size and not tonumber(bunch_size) then
        error("The third parameter must be a number")
      end
      self.ann_component    = assert(ann_component,"Needs an ANN component object")
      self.loss_function    = loss_function or false
      self.optimizer        = optimizer
      self.smooth_gradients = smooth_gradients
      self.weights_table    = matrix.dict()
      self.components_table = {}
      self.component2weights_dict = {}
      self.weights2component_dict = {}
      self.weights_order    = {}
      self.components_order = {}
      self.bunch_size       = bunch_size or false
      self.weight_grads     = matrix.dict()
    end
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.get_component =
  april_doc{
    class = "method",
    summary = "Returns an instance of ann.components",
    outputs = { "An instance of ann.components" },
  } ..
  function(self)
    return self.ann_component
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.set_loss_function =
  april_doc{
    class = "method",
    summary = "Modifies the loss function property",
    params = { "Loss function" },
  } ..
  function(self, loss_function)
    assert(is_a(loss_function, ann.loss), "Needs an instance of ann.loss")
    self.loss_function = loss_function
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.get_loss_function =
  april_doc{
    class = "method",
    summary = "Modifies the loss function property",
    params = { "Loss function" },
  } ..
  function(self)
    return self.loss_function
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.set_optimizer =
  april_doc{
    class = "method",
    summary = "Modifies the optimizer property",
    params = { "An instance of ann.optimizer" },
  } ..
  function(self, optimizer)
    assert(is_a(optimizer, ann.optimizer), "Needs an instance of ann.optimizer")
    self.optimizer = optimizer
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.get_optimizer =
  april_doc{
    class = "method",
    summary = "Returns the optimizer property",
    outputs = { "An instance of ann.optimizer" },
  } ..
  function (self)
    return self.optimizer
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.set_option =
  april_doc{
    class = "method",
    summary = "Sets a global option of the optimizer",
    params  = {
      "An string with the option name",
      "A number with the value",
    },
  } ..
  function(self, name,value)
    local opt = assert(self.optimizer, "The optimizer has not been defined")
    opt:set_option(name,value)
  end

trainable_supervised_trainer_methods.get_option =
  april_doc{
    class = "method",
    summary = "Gets a global option of the optimizer",
    params  = {
      "An string with the option name",
    },
    outputs = {
      "A number with the option value",
    },
  } ..
  function(self, name)
    local opt = assert(self.optimizer, "The optimizer has not been defined")
    return opt:get_option(name)
  end

trainable_supervised_trainer_methods.has_option =
  april_doc{
    class = "method",
    summary = "Returns true/false depending on the existence of a global option of the optimizer",
    params  = {
      "An string with the option name",
    },
    outputs = {
      "A boolean",
    },
  } ..
  function(self, name)
    local opt = assert(self.optimizer, "The optimizer has not been defined")
    return opt:has_option(name)
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.set_layerwise_option =
  april_doc{
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
    },
  } ..
  function(self, layer_match, name, value)
    local N=0
    local opt = assert(self.optimizer, "The optimizer has not been defined")
    for cnn_name,cnn in self:iterate_weights(layer_match) do
      opt:set_layerwise_option(cnn_name,name,value)
      N=N+1
    end
    april_assert(N>0, "0 layers match the given layer_match= %s", layer_match)
  end

trainable_supervised_trainer_methods.get_option_of =
  april_doc{
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
    outputs = { "A number with the value", },
  } ..
  function(self, layer_name, name)
    local opt = assert(self.optimizer, "The optimizer has not been defined")
    return opt:get_option_of(layer_name,name)
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.get_input_size =
  april_doc{
    class = "method",
    summary = "Gets the input size of its component",
    outputs = { "The input size (a number)" },
  } ..
  function(self)
    return self.ann_component:get_input_size()
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.get_output_size =
  april_doc{
    class = "method",
    summary = "Gets the output size of its component",
    outputs = { "The output size (a number)" },
  } ..
  function(self)
    return self.ann_component:get_output_size()
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.size =
  april_doc{
    class = "method",
    summary = "Returns the model size (number of weights)",
    outputs = { "A number" },
  } ..
  function(self)
    if not self.is_built then
      error("It is not build")
    end
    return self.weights_table:size()
  end

------------------------------------------------------------------------

function trainable_supervised_trainer_methods:to_lua_string(format)
  assert(self.is_built, "The component is not built")
  local t = { }
  table.insert(t, "trainable.supervised_trainer{ ")
  table.insert(t, "model=")
  table.insert(t, self.ann_component:to_lua_string(format))
  table.insert(t, ",\n")
  table.insert(t, "connections={")
  for _,wname in ipairs(self.weights_order) do
    local cobj = self.weights_table(wname)
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

trainable_supervised_trainer_methods.save =
  april_doc{
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
    },
  } ..
  function(self, filename, format)
    assert(self.is_built, "The component is not built")
    local f = io.open(filename,"w") or error("Unable to open " .. filename)
    f:write("return ")
    f:write(self:to_lua_string(format))
    f:write("\n")
    f:close()
  end

------------------------------------------------------------------------

trainable.supervised_trainer.load =
  april_doc{
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
    },
  } ..
  function(filename, loss, bunch_size, optimizer)
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

trainable_supervised_trainer_methods.count_components =
  april_doc{
    class = "method",
    summary = "Count the number of components",
    params = {
      { "A match string: filter and count only components",
        "which match [optional], by default is '.*'" },
    }, 
  } ..
  function(self, match_string)
    local match_string = match_string or ".*"
    if not self.is_built then
      error("It is not build")
    end
    local count = 0
    for i=1,#self.components_order do
      if self.components_order[i]:match(match_string) then count=count+1 end
    end
    return count
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.count_weights =
  april_doc{
    class = "method",
    summary = "Count the number of connection weight objects",
    params = {
      { "A match string: filter and count only connections",
        "which match [optional], by default is '.*'" },
    },
  } ..
  function(self, match_string)
    local match_string = match_string or ".*"
    if not self.is_built then
      error("It is not build")
    end
    local count = 0
    for i=1,#self.weights_order do
      if self.weights_order[i]:match(match_string) then count=count+1 end
    end
    return count
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.iterate_components =
  april_doc{
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
    },
  } ..
  function(self, match_string)
    local match_string = match_string or ".*"
    if not self.is_built then
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

trainable_supervised_trainer_methods.iterate_weights =
  april_doc{
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
    },
  } ..
  function(self, match_string)
    local match_string = match_string or ".*"
    if not self.is_built then
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
      return name,self.weights_table(name)
    end
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.component =
  april_doc{
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
    outputs = { "A component object" },
  } ..
  function(self, str)
    if not self.is_built then
      error("Needs execution of build method")
    end
    return self.components_table[str]
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.weights =
  april_doc{
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
    outputs = { "An ann.connections object" },
  } ..
  function(self, str)
    if not self.is_built then
      error("Needs execution of build method")
    end
    return self.weights_table(str)
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.get_weights_table =
  april_doc{
    class = "method",
    summary = "Returns the table with all weights",
    outputs = { "A table with all the weights, indexed by names" },
  } ..
  function(self)
    if not self.is_built then
      error("Needs execution of build method")
    end
    return self.weights_table
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.randomize_weights =
  april_doc{
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
    },
  } ..
  function(self, t)
    local params = get_table_fields(
      {
        name_match = { type_match="string", mandatory=false, default = nil },
        random  = { isa_match = random,  mandatory = true },
        inf     = { type_match="number", mandatory = true },
        sup     = { type_match="number", mandatory = true },
        use_fanin  = { type_match="boolean", mandatory = false, default = false },
        use_fanout = { type_match="boolean", mandatory = false, default = false },
      }, t)
    assert(self.is_built,
           "Execute build method before randomize_weights")
    for i,wname in ipairs(self.weights_order) do
      if not params.name_match or wname:match(params.name_match) then
        local current_inf = params.inf
        local current_sup = params.sup
        local constant    = 0
        local connection  = self.weights_table(wname)
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

trainable_supervised_trainer_methods.build =
  april_doc{
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
      "The caller object",
      "Weights table, associates weights_name => ann.connections object",
      "Components table, associates component_name => ann.components object",
    },
  } ..
  function(self, t)
    local params = get_table_fields(
      {
        weights = { mandatory = false, default=nil },
        input   = { type_match="number", mandatory = false, default=nil },
        output  = { type_match="number", mandatory = false, default=nil },
      }, t or {})
    self.weight_grads  = matrix.dict()
    self.weights_table = wrap_matrices(params.weights or matrix.dict())
    -- BUILD CALL
    _,
    self.weights_table,
    self.components_table = self.ann_component:build{
      input   = params.input,
      output  = params.output,
      weights = self.weights_table, }
    --
    self.weights_order = self.weights_table:keys()
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
    self.is_built = true
    return self,self.weights_table,self.components_table
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.get_weights_of =
  april_doc{
    class = "method",
    summary = "Returns a the object connections related to given component name",
    params = { "A string with the component name" },
    outputs = { "An instance of ann.connections" },
  } ..
  function(self, name)
    return self.weights_table(self.component2weights_dict[name])
  end

trainable_supervised_trainer_methods.get_components_of =
  april_doc{
    class = "method",
    summary = "Returns a table with the components related to given weights name",
    params = { "A string with the weights name" },
    outputs = { "A table of ann.components instances" },
  } ..
  function(self, wname)
  return self.weights2component_dict[wname] or {}
end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.train_step =
  april_doc{
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
      "A mask [optional]",
    },
    outputs = {
      "The mean of loss function at current batch",
      "A matrix with the loss of every given pattern",
    },
  } ..
  function(self, input, target, loss, optimizer,
           bunch_size, smooth_gradients, mask)
    if type(input)  == "table" then input  = matrix.col_major(input)  end
    if type(target) == "table" then target = matrix.col_major(target) end
    if type(mask)   == "table" then mask   = matrix.col_major(mask) end
    local loss       = loss or self.loss_function or error("Needs a loss object")
    local optimizer  = optimizer or self.optimizer or error("Needs an optimizer object")
    local bunch_size = bunch_size or self.bunch_size or 1
    local smooth_gradients = smooth_gradients or self.smooth_gradients
    if mask then
      if not is_a(mask,matrix) then mask = mask:get_matrix() end
      if not is_a(target,matrix) then
        target = target:get_matrix()
      end
      target = target:clone():cmul(mask)
    end
    local needs_gradient = optimizer:needs_property("gradient")
    local tr_loss, _, tr_loss_matrix =
      optimizer:execute(function(weights, it)
          if weights ~= self.weights_table then
            self:build{ weights = weights }
          end
          local self   = self
          local loss   = loss
          local model  = self.ann_component
          local grads  = self.weight_grads
          local target = target
          local mask   = mask
          model:reset(it)
          local output = model:forward(input, true)
          if mask then
            if not is_a(output,matrix) then
              output = output:get_matrix()
            end
            if not is_a(target,matrix) then
              target = target:get_matrix()
            end
            output = output:clone():cmul(mask)
          end
          local tr_loss,tr_loss_matrix
          tr_loss,tr_loss_matrix = loss:compute_loss(output, target)
          if not tr_loss_matrix then return nil end
          if needs_gradient then
            local gradient=model:backprop(loss:gradient(output,target))
            --
            grads:zeros()
            --
            local grads = model:compute_gradients(grads)
            self.weight_grads = grads
            -- gradient smoothing
            if smooth_gradients then
              for name,mat in pairs(grads) do
                local N = mat:get_shared_count()
                N       = ( N>0 and N) or 1
                mat:scal( 1.0/math.sqrt(N * bunch_size) )
              end
            end
            -- the loss, the gradients, and the loss matrix
            return tr_loss, grads, tr_loss_matrix
          else
            return tr_loss, nil, tr_loss_matrix
          end
                        end,
        self.weights_table)
    if tr_loss_matrix then loss:accum_loss(tr_loss_matrix) end
    return tr_loss,tr_loss_matrix
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.validate_step =
  april_doc{
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
    },
  } ..
  function(self, input, target, loss, mask)
    if type(input)  == "table" then input  = matrix.col_major(input)  end
    if type(target) == "table" then target = matrix.col_major(target) end
    if type(mask)   == "table" then mask   = matrix.col_major(mask) end
    local model = self.ann_component
    local loss  = loss or self.loss_function
    model:reset()
    local output = model:forward(input)
    if mask then
      if not is_a(mask,matrix) then mask = mask:get_matrix() end
      if not is_a(output,matrix) then
        output = output:get_matrix()
      end
      if not is_a(target,matrix) then
        target = target:get_matrix()
      end
      output = output:clone():cmul(mask)
      target = target:clone():cmul(mask)
    end
    local tr_loss,tr_loss_matrix = loss:compute_loss(output, target)
    if tr_loss_matrix then
      loss:accum_loss(tr_loss_matrix)
      return tr_loss,tr_loss_matrix
    end
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.compute_gradients_step =
  april_doc{
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
    },
  } ..
  function(self, input, target, loss, weight_grads)
    if type(input)  == "table" then input  = matrix.col_major(input)  end
    if type(target) == "table" then target = matrix.col_major(target) end
    local loss         = loss or self.loss_function
    local weight_grads = weight_grads or matrix.dict()
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

trainable_supervised_trainer_methods.grad_check_step =
  april_doc{
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
    },
  } ..
  function(self, input, target, verbose, loss)
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
    local epsilond = 0.10 -- 10% relative error
    local epsilon  = 1e-03
    local ret      = true
    local bunch_size = tr_loss_matrix:dim(1)
    local it = 1
    for wname,cnn in self:iterate_weights() do
      collectgarbage("collect")
      local w = cnn
      -- The shared parameter has no effect in gradients check, only bunch_size
      local ratio = 1/bunch_size
      local ann_grads = self.weight_grads(wname)
      assert(w:is_contiguous(),
             "Unable to check grads of non-contiguous matrices")
      for i=1,w:size() do
        collectgarbage("collect")
        local orig_w = w:raw_get(w:offset() + i-1)
        w:raw_set(w:offset() + i-1, orig_w - epsilon)
        self.ann_component:reset(it)
        it=it+1
        local loss_a = loss:compute_loss(self.ann_component:forward(input,true),
                                         target)
        w:raw_set(w:offset() + i-1, orig_w + epsilon)
        self.ann_component:reset(it)
        it=it+1
        local loss_b = loss:compute_loss(self.ann_component:forward(input,true),
                                         target)
        w:raw_set(w:offset() + i-1, orig_w)
        w:update()
        local g = (loss_b - loss_a) / (2*epsilon)
        local ann_g = ann_grads:raw_get(i-1)*ratio
        if verbose then
          fprintf(io.stderr,
                  "CHECK GRADIENT %s[%d], found %g, expected %g\n",
                  wname, i-1, ann_g, g)
        end
        if math.abs(ann_g) > 2*epsilon or math.abs(g) > 2*epsilon then
          local abs_err = math.abs(ann_g - g)
          local err = 2*abs_err/(math.abs(ann_g) + math.abs(g))
          if err > epsilond and abs_err > 2*epsilon then
            -- force backprop step
            self.ann_component:reset(it)
            loss:reset()
            local output   = self.ann_component:forward(input, true)
            local tr_loss,tr_loss_matrix = loss:compute_loss(output, target)
            assert(tr_loss_matrix)
            local gradient = loss:gradient(output, target)
            self.ann_component:backprop(gradient)
            --
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

trainable_supervised_trainer_methods.calculate =
  april_doc{
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
    },
  } ..
  function(self,input)
    if type(input) == "table" then input = matrix.col_major(input) end
    self.ann_component:reset()
    return self.ann_component:forward(input):get_matrix()
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.train_dataset =
  april_doc{
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
      ["mask_dataset"] = "A dataset float or dataset token (mask target output)",
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
    },
  } ..
  april_doc{
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
      ["mask_dataset"] = "A dataset float or dataset token (mask target output)",
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
    },
  } ..
  april_doc{
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
      ["mask_dataset"] = "A dataset float or dataset token (mask target output)",
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
    },
  } ..
  april_doc{
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
    },
  } ..
  function(self, t)
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
    assert(self.is_built,
           "Execute build method before call this method")
    local loss       = params.loss
    local optimizer  = params.optimizer
    local smooth_gradients = params.smooth_gradients
    local mask_dataset = params.mask_dataset
    params.loss      = nil
    params.optimizer = nil
    params.smooth_gradients   = nil
    params.mask_dataset       = nil
    params.bunch_size         = params.bunch_size or self.bunch_size
    -- set to ZERO the accumulated of loss
    loss:reset()
    if not mask_dataset then
      params.assert_input_size  = self:get_input_size()
      params.assert_output_size = self:get_output_size()
      for input_bunch,output_bunch,bunch_indexes in trainable.dataset_pair_iterator(params) do
        self:train_step(input_bunch, output_bunch, loss, optimizer, #bunch_indexes,
                        smooth_gradients)
      end
    else
      params.datasets = { params.input_dataset,
                          params.output_dataset,
                          mask_dataset }
      params.input_dataset  = nil
      params.output_dataset = nil
      params.assert_pattern_sizes = { self:get_input_size(),
                                      self:get_output_size(),
                                      self:get_output_size() }
      for input_bunch,output_bunch,mask_bunch,bunch_indexes in trainable.dataset_multiple_iterator(params) do
        self:train_step(input_bunch, output_bunch, loss, optimizer, #bunch_indexes,
                        smooth_gradients, mask_bunch)
      end
    end
    return loss:get_accum_loss()
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.grad_check_dataset =
  april_doc{
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
    },
  } ..
  function(self, t)
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
    assert(self.is_built,
           "Execute build method before call this method")
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
    return ret
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.validate_dataset =
  april_doc{
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
      ["mask_dataset"] = "A dataset float or dataset token (mask target output)",
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
    },
  } ..
  april_doc{
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
      ["mask_dataset"] = "A dataset float or dataset token (mask target output)",
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
    },
  } ..
  april_doc{
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
      ["mask_dataset"] = "A dataset float or dataset token (mask target output)",
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
    },
  } ..
  function(self, t)
    local params = get_table_fields(
      {
        -- In this case, we check all the given parameters, because all the
        -- dataset iteration schemes are not available for validate_dataset
        input_dataset  = { mandatory = true },
        output_dataset = { mandatory = true },
        mask_dataset   = { mandatory = false },
        bunch_size     = { type_match = "number",
                           mandatory = (self.bunch_size == false) },
        loss           = { isa_match  = ann.loss,
                           mandatory = (self.loss_funcion==false),
                           default=self.loss_function },
        shuffle        = { isa_match  = random, mandatory = false, default=nil },
        replacement    = { type_match = "number", mandatory = false, default=nil },
      }, t)
    assert(self.is_built,
           "Execute build method before call this method")
    -- ERROR CHECKING
    assert(params.input_dataset ~= not params.output_dataset,
           "input_dataset and output_dataset fields are mandatory together")
    assert(not params.input_dataset or not params.distribution,
           "input_dataset/output_dataset fields are forbidden with distribution")
    local loss                = params.loss
    local mask_dataset        = params.mask_dataset
    params.loss               = nil
    params.mask_dataset       = nil
    params.bunch_size         = params.bunch_size or self.bunch_size
    -- set to ZERO the accumulated of loss
    loss:reset()
    if not mask_dataset then
      params.assert_input_size  = self:get_input_size()
      params.assert_output_size = self:get_output_size()
      for input_bunch,output_bunch in trainable.dataset_pair_iterator(params) do
        self:validate_step(input_bunch, output_bunch, loss)
      end
    else
      params.datasets = { params.input_dataset,
                          params.output_dataset,
                          mask_dataset }
      params.input_dataset  = nil
      params.output_dataset = nil
      params.assert_pattern_sizes = { self:get_input_size(),
                                      self:get_output_size(),
                                      self:get_output_size() }
      for input_bunch,output_bunch in trainable.dataset_multiple_iterator(params) do
        self:validate_step(input_bunch, output_bunch, loss)
      end
    end
    return loss:get_accum_loss()
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.use_dataset = 
  april_doc{
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
    },
  } ..
  function(self, t)
    local params = get_table_fields(
      {
        -- In this case, we check all the given parameters, because all the
        -- dataset iteration schemes are not available for use_dataset
        input_dataset  = { mandatory = true },
        output_dataset = { mandatory = false, default=nil },
        bunch_size     = { type_match = "number",
                           mandatory = (self.bunch_size == false)  },
      }, t)
    assert(self.is_built,
           "Execute build method before call this method")
    local nump        = params.input_dataset:numPatterns()
    local outsize     = self.ann_component:get_output_size()
    if params.output_dataset then
      if is_a(params.output_dataset, dataset) then
        params.output_dataset = dataset.token.wrapper(params.output_dataset)
      end
    elseif is_a(params.input_dataset, dataset) then
      params.output_dataset = dataset.matrix(matrix(nump, outsize))
      t.output_dataset      = params.output_dataset
      params.output_dataset = dataset.token.wrapper(params.output_dataset)
    else
      params.output_dataset = dataset.token.vector(outsize)
      t.output_dataset      = params.output_dataset
    end
    if is_a(params.input_dataset, dataset) then
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

trainable_supervised_trainer_methods.show_weights =
  april_doc{
    class = "method",
    summary = "Print connection weights (for debug purposes).",
  } ..
  function(self)
    for _,wname in pairs(self.weights_order) do
      local w = self.weights_table(wname):toTable()
      print(wname, table.concat(w, " "))
    end
  end

------------------------------------------------------------------------

trainable_supervised_trainer_methods.clone =
  april_doc{
    class = "method",
    summary = "Returns a deep-copy of the object.",
    description = {
      "Returns a deep-copy of the object.",
      "This method copies user functions on given object instance,",
      "so user can overwrite any of methods in this class.",
    },
  } ..
  function(self)
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
      obj:build{ weights = self.weights_table:clone() }
    end
    -- add possible user functions
    for i,v in pairs(self) do
      if type(v) == "function" then obj[i] = v end
    end
    return obj
  end

trainable_supervised_trainer_methods.norm2 =
  april_doc{
    class = "method",
    summary = "Returns the maximum norm2 of the weights which name matches",
    params={
      "A connection weights name Lua pattern string [optional], by default it is .*",
    },
    outputs = {
      "A number with the maximum norm2",
    },
  } ..
  function(self, match_string)
    local norm2 = 0
    for _,cnn in self:iterate_weights(match_string) do
      norm2 = math.max(norm2,
                       reduce(function(a,b)
                           return math.max(a,b:norm2())
                              end, 0, cnn:sliding_window():iterate()))
    end
    return norm2
  end

function trainable_supervised_trainer_methods:train_holdout_validation(t)
  error("DEPRECATED: use the class trainable.train_holdout_validation")
end

function trainable_supervised_trainer_methods:train_wo_validation(t)
  error("DEPRECATED: use the class trainable.train_wo_validation")
end
