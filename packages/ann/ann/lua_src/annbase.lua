get_table_from_dotted_string("ann.mlp.all_all", true)
get_table_from_dotted_string("ann.components", true)

----------------------------------------------------------------------

class.extend(ann.components.base, "get_is_recurrent",
             function() return false end)

----------------------------------------------------------------------

ann.components.const_mul = function(t)
  local params = get_table_fields(
    {
      data    = { mandatory=true },
      weights = { type_match="string" },
      name    = { type_match="string" },
    }, t)
  local m = params.data
  params.data = nil
  if tonumber(m) then
    m = matrix{ m }
  elseif type(m) == "table" then
    m = matrix(m)
  else
    assert(class.is_a(m, matrix), "Needs a matrix, table or number in 'data' field")
  end
  assert(m:num_dim() == 1, "Needs a one-dimensional matrix")
  params.scalar = (m:size() == 1)
  if not params.scalar then params.size = m:size() end
  local c = ann.components.mul(params)
  local wname = c:get_weights_name()
  c:build{ weights = { [wname] = m } }
  local const_name = c:get_name() .. "_const"
  return ann.components.const{ component = c, name = const_name }
end

----------------------------------------------------------------------

ann.components.left_probabilistic_matrix = function(t)
  t.side = "left"
  return ann.components.probabilistic_matrix(t)
end

ann.components.right_probabilistic_matrix = function(t)
  t.side = "right"
  return ann.components.probabilistic_matrix(t)
end

----------------------------------------------------------------------

local ann_wrapper, ann_wrapper_methods = class("ann.components.wrapper")
ann.components.wrapper = ann_wrapper -- global environment

function ann_wrapper:constructor(t)
  local params = get_table_fields(
    {
      input = { mandatory = true, type_match="number", default=0 },
      output = { mandatory = true, type_match="number", default=0 },
      weights = { mandatory = true },
      forward = { mandatory = true, type_match="function" },
      backprop = { mandatory = false, type_match="function" },
      compute_gradients = { mandatory=false, type_match="function" },
      reset = { mandatory=false, type_match="function" },
      state = { mandatory=false },
    }, t)
  self.forward_function = params.forward
  self.backprop_function = params.backprop or
  function() error"Not implemented for THIS wrapper component" end
  self.compute_gradients_function = params.compute_gradients or
  function() error"Not implemented for THIS wrapper component" end
  self.reset_function = params.reset or function() end
  self.input = params.input
  self.output = params.output
  self.weights = params.weights
  self.state = params.state
end

function ann_wrapper_methods:has_weights_name()
  return false
end

function ann_wrapper_methods:get_is_built()
  return true
end

function ann_wrapper_methods:debug_info()
end

function ann_wrapper_methods:copy_state(tbl)
  error("Not implemented")
end

function ann_wrapper_methods:set_state(tbl)
  error("Not implemented")
end

function ann_wrapper_methods:get_input_size()
  return self.input_size
end

function ann_wrapper_methods:get_output_size()
  return self.output_size
end

function ann_wrapper_methods:get_input()
  return self.input
end

function ann_wrapper_methods:get_output()
  return self.output
end

function ann_wrapper_methods:get_error_input()
  return self.input
end

function ann_wrapper_methods:get_error_output()
  return self.output
end

function ann_wrapper_methods:forward(input, during_training)
  self:reset()
  self.input  = input
  self.output = self:forward_function(input, during_training)
  return self.output
end

function ann_wrapper_methods:backprop(input)
  self.error_input  = input
  self.error_output = self:backprop_function(input)
  return self.error_output
end

function ann_wrapper_methods:reset(n)
  self:reset_function(n)
  self.input = nil
  self.output = nil
  self.error_input = nil
  self.error_output = nil
end

function ann_wrapper_methods:compute_gradients(dict)
  local dict = dict or {}
  self:compute_gradients_function(dict)
  return dict
end

function ann_wrapper_methods:build(params)
  params = params or {}
  local input,output,weights = params.input,params.output,params.weights
  -- TODO: check input/output sizes
  if params.weights then matrix.dict.replace( self.weights, params.weights ) end
  return self,self.weights,{}
end

function ann_wrapper_methods:copy_weights()
  return self.weights
end

function ann_wrapper_methods:copy_components()
  return {}
end

function ann_wrapper_methods:ctor_name()
  error("Impossible to serialize a wrapper component")
end
function ann_wrapper_methods:ctor_params()
  error("Impossible to serialize a wrapper component")
end

function ann_wrapper_methods:get_name()
  error("Not implemented in wrapper component")
end

function ann_wrapper_methods:get_weights_name()
  error("Not implemented in wrapper component")
end

function ann_wrapper_methods:precompute_output_size()
  error("Not implemented in wrapper component")
end

function ann_wrapper_methods:clone()
  error("Not implemented in wrapper component")
end

function ann_wrapper_methods:set_use_cuda()
  error("Not implemented in wrapper component")
end

function ann_wrapper_methods:get_use_cuda()
  error("Not implemented in wrapper component")
end

function ann_wrapper_methods:get_component(name)
  error("Not implemented in wrapper component")
end

----------------------------------------------------------------------

local lua_component_methods
ann.components.lua,lua_component_methods = class("ann.components.lua",
                                                 ann.components.base)

ann.components.lua.constructor = function(self, tbl)
  local tbl = get_table_fields({
      name = { type_match="string" },
      input = { type_match="number" },
      output = { type_match="number" },
                               }, tbl or {})
  self.name = tbl.name or ann.generate_name()
  self.input_size = tbl.input
  self.output_size = tbl.output
end

lua_component_methods.ctor_name = function(self)
  return class.obj_id(self)
end

lua_component_methods.ctor_params = function(self)
  return { name=self.name, input=self.input_size, output=self.output_size }
end

lua_component_methods.set_input = function(self,tk)
  self.input_token = tk
end

lua_component_methods.set_output = function(self,tk)
  self.output_token = tk
end

lua_component_methods.set_error_input = function(self,tk)
  self.error_input_token = tk
end

lua_component_methods.set_error_output = function(self,tk)
  self.error_output_token = tk
end


lua_component_methods.build = function(self,tbl)
  tbl = tbl or {}
  self.is_built = true
  local input_size,output_size
  if tbl.input and tbl.input ~= 0 then
    input_size = tbl.input
  end
  if tbl.output and tbl.output ~= 0 then
    output_size = tbl.output
  end
  local self_input_size = rawget(self,"input_size")
  local self_output_size = rawget(self,"output_size")
  self.input_size = input_size or self_input_size
  self.output_size = output_size or self_output_size
  return self,tbl.weights or {},{ [self.name] = self }
end

lua_component_methods.forward = function(self, input, during_training)
  assert(rawget(self,"is_built"), "It is needed to call build method")
  self.input_token = input
  self.output_token = input
  return input
end

lua_component_methods.backprop = function(self, input)
  assert(rawget(self,"is_built"), "It is needed to call build method")
  assert(rawget(self,"output_token"), "It is needed to call forward method")
  self.error_input_token = input
  self.error_output_token = input
  return input
end

lua_component_methods.compute_gradients = function(self, weight_grads)
  assert(rawget(self,"is_built"), "It is needed to call build method")
  assert(rawget(self,"output_token"), "It is needed to call forward method")
  assert(rawget(self,"error_output_token"), "It is needed to call backprop method")
  return weight_grads or {}
end

lua_component_methods.reset = function(self, n)
  self.input_token = nil
  self.output_token = nil
  self.error_input_token = nil
  self.error_output_token = nil
end

lua_component_methods.get_name = function(self)
  return self.name
end

lua_component_methods.get_weights_name = function(self)
  return nil
end

lua_component_methods.has_weights_name = function(self)
  return false
end

lua_component_methods.get_is_built = function(self)
  return rawget(self,"is_built") or false
end

lua_component_methods.debug_info = function(self)
  error("Not implemented")
end

lua_component_methods.copy_state = function(self, tbl)
  tbl = tbl or {}
  tbl[self.name] = {
    input = self:get_input(),
    output = self:get_output(),
    error_input = self:get_error_input(),
    error_output = self:get_error_output(),
  }
  return tbl
end

lua_component_methods.set_state = function(self, tbl)
  local tbl = april_assert(tbl[self.name], "State %s not found", self.name)
  self.input = tbl.input
  self.output = tbl.output
  self.error_input = tbl.input
  self.error_output = tbl.output
  return self
end

lua_component_methods.get_input_size = function(self)
  return rawget(self,"input_size") or 0
end

lua_component_methods.get_output_size = function(self)
  return rawget(self,"output_size") or 0
end

lua_component_methods.get_input = function(self)
  return rawget(self,"input_token") or tokens.null()
end

lua_component_methods.get_output = function(self)
  return rawget(self,"output_token") or tokens.null()
end

lua_component_methods.get_error_input = function(self)
  return rawget(self,"error_input_token") or tokens.null()
end

lua_component_methods.get_error_output = function(self)
  return rawget(self,"error_output_token") or tokens.null()
end

lua_component_methods.precompute_output_size = function(self, tbl)
  error("Not implemented")
end

lua_component_methods.clone = function(self)
  return class.of(self){ name=self.name,
                         input=self:get_input_size(),
                         output=self:get_output_size() }
end

lua_component_methods.set_use_cuda = function(self, v)
  self.use_cuda = v
end

lua_component_methods.get_use_cuda = function(self)
  return rawget(self,"use_cuda") or false
end

lua_component_methods.copy_weights = function(self, dict)
  return dict or {}
end

lua_component_methods.copy_components = function(self, dict)
  local dict = dict or {}
  dict[self.name] = self
  return dict
end

lua_component_methods.get_component = function(self, name)
  if self.name == name then return self end
end

----------------------------------------------------------------------

function ann.connections.input_filters_image(w, shape, options)
  local params = get_table_fields({
      margin = { type_match="number", mandatory=false, default=1 },
      bgcolor = { type_match="number", mandatory=false, default=0 },
      rows = { type_match="number", mandatory=false },
      cols = { type_match="number", mandatory=false },
                                  }, options)
  local notranspose = false
  assert(type(shape) == "table", "Needs a shape table as 2nd argument")
  assert(#shape == 2 or #shape == 3,
	 "Expected shape with 2 or 3 dimensions")
  assert(#shape == 2 or shape[3] == 3,
	 "Expected 3 components at the 3rd dimension (color RGB)")
  assert(shape[1] and shape[2], "Found nil values in shape (2nd argument)")
  local margin  = params.margin
  local bgcolor = params.bgcolor
  local R = math.ceil(math.sqrt(w:dim(1)))
  local C = math.floor(math.sqrt(w:dim(1)))
  if params.rows then
    R = params.rows
    if params.cols then
      C = params.cols
      assert(R*C == w:dim(1), "Incorrect number of rows and cols")
    else
      C = math.ceil(w:dim(1)/R)
    end
  elseif params.cols then
    C = params.cols
    R = math.ceil(w:dim(1)/C)
  end
  local result = matrix(R*(shape[1]+margin)-margin,C*(shape[2]+margin)-margin,shape[3])
  result:fill(bgcolor)
  local neuron_weights
  local step = { shape[1]+margin, shape[2]+margin }
  if #shape == 3 then step[3] = 1 end
  local result_sw = result:sliding_window{ size=shape, step=step }
  local result_m
  assert(shape[1]*shape[2]*(shape[3] or 1) == w:dim(2),
         "Incorrect shape dimensions")
  for i=1,w:dim(1) do
    result_m = result_sw:get_matrix(result_m)
    neuron_weights = w:select(1,i,neuron_weights)
    local normalized = neuron_weights:clone():
      rewrap(table.unpack(shape)):clone():
      scal(1/neuron_weights:norm2()):
      adjust_range(0,1)
    result_m:copy(normalized)
    --
    result_sw:next()    
  end
  if #shape == 2 then
    return Image(result)
  else
    return ImageRGB(result)
  end
end

----------------------------------------------------------------------

ann.save = function()
  error("Not available, use util.serialize instead")
end

ann.load = util.deserialize
----------------------------------------------------------------------

april_set_doc(ann.mlp,
	      {
		class="namespace",
		summary="Namespace with utilties for easy MLP training", })

----------------------------------------------------------------------

april_set_doc(ann.mlp.all_all,
	      {
		class="namespace",
		summary="Namespace with utilities for all-all MLP training", })

----------------------------------------------------------------------

ann.mlp.all_all.generate = april_doc {
  class="function",
  summary="Function to build all-all stacked ANN models",
  description=
    {
      "This function composes a component object from the",
      "given topology description (stacked all-all).",
      "It generates default names for components and connection",
      "weights. Each layer has one ann.components.dot_product",
      "with name=PREFIX..'w'..NUMBER and",
      "weights_name=PREFIX..'w'..NUMBER,",
      "one ann.components.bias with name='b'..NUMBER and",
      "weights_name=PREFIX..'b'..NUMBER, and an",
      "ann.components.actf with",
      "name=PREFIX..'actf'..NUMBER.",
      "NUMBER is a counter initialized at 1, or with the",
      "value of second argument (count) for",
      "ann.mlp.all_all(topology, count) if it is given.",
      "PREFIX is the third argument of the function, by default",
      "is an empty string.",
    },
  params= {
    { "Topology description string as ",
      "'1024 inputs 128 logistc 10 log_softmax" },
    { "A table with extra arguments. Positional arguments are given as",
      "parameters designed by #n in topology options string. Besides,",
      "first_count and names_prefix keys can be given.", },
  },
  outputs= {
    {"A component object with the specified ",
     "neural network topology" }
  }
} ..
  function(topology, tbl, first_count, names_prefix)
    if type(tbl) == "table" then
      assert(not first_count and not names_prefix)
      first_count, names_prefix = tbl.first_count, tbl.names_prefix
    else -- for back compatibility, it will be deprecated
      first_count, names_prefix = tbl, first_count
    end
    local first_count  = tonumber(first_count or 1)
    local names_prefix = names_prefix or ""
    local thenet
    local name   = "layer"
    local count  = first_count
    local prev_size
    local names_order = {}
    -- RECURSIVE DESCENT PARSER
    -- reads a token which can be any string and symbols of {}=
    local tokenizer = coroutine.wrap(function()
        local last = 1
        for i=1,#topology do
          local c = topology:sub(i,i)
          if c:find("[%s%c,]") then
            if last < i then coroutine.yield(topology:sub(last,i-1)) end
            last = i+1
          else -- if c is space or special then ... else
            if c:find("[{}=]") then
              if last < i then coroutine.yield(topology:sub(last,i-1)) end
              coroutine.yield(c)
              last = i+1
            end -- if c in [{}=]
          end -- if c is space or special then ... else ... end
        end -- for i=1,#topology
        if last < #topology then coroutine.yield(topology:sub(last)) end
    end) -- coroutine.wrap(function()
    -- reads a sequence of options delimited by braces
    local function read_options(safe_tokenizer, params)
      local options = {}
      repeat
        local tk = safe_tokenizer()
        if tk ~= "}" then
          local key = tk
          assert(safe_tokenizer() == "=", "Incorrect topology string")
          local value = safe_tokenizer()
          if value:sub(1,1) == "#" then value = params[tonumber(value:sub(2))] end
          if     value == "true"   then value = true
          elseif value == "false"  then value = false
          else                          value = tonumber(value) or value
          end
          options[key] = value
        end
      until tk == "}"
      return options
    end
    -- reads ANN components from the topology string and push them into the
    -- stacked model
    local function read_component(tk0, tokenizer, params)
      local safe_tokenizer = function()
        return assert(tokenizer(), "Incorrect topology string")
      end
      local n0      = tonumber(tk0)
      local kind    = tk0
      local options = {}
      if n0 then kind = safe_tokenizer() end
      tk0 = tokenizer()
      if tk0 == "{" then -- the string contains options for current component
        options = read_options(safe_tokenizer, params)
        tk0 = tokenizer()
      end
      -- push the corresponding components
      if kind ~= "inputs" then
        assert(thenet, "'inputs' string is required")
        if n0 then -- hyperplane + activation function
          local size = n0
          thenet:push( ann.components.hyperplane{
                         input=prev_size, output=size,
                         bias_weights=names_prefix.."b" .. count,
                         dot_product_weights=names_prefix.."w" .. count,
                         name=names_prefix.."layer" .. count,
                         bias_name=names_prefix.."b" .. count,
                         dot_product_name=names_prefix.."w" .. count } )
          table.insert(names_order, names_prefix.."b"..count)
          table.insert(names_order, names_prefix.."w"..count)
          local ctor = ann.components.actf[kind] or ann.components[kind]
          assert(ctor, "Incorrect component class: " .. kind)
          options.name = names_prefix .. "actf" .. count
          thenet:push( ctor( options ) )
        else -- any other kind of component
          options.name = names_prefix .. "component" .. count
          local ctor = ann.components[kind] or ann.components.actf[kind]
          assert(ctor, "Incorrect component class: " .. kind)
          thenet:push( ctor( options ) )
        end
        count = count + 1
      else -- if kind ~= "inputs"
        thenet = ann.components.stack{ name=names_prefix.."stack", input=n0 }
      end 
      prev_size = n0 or prev_size
      return tk0
    end
    -- main loop of the parser
    local tk0,component = tokenizer(),nil
    while tk0 do
      tk0 = read_component(tk0, tokenizer, tbl)
    end
    --
    local aux = get_lua_properties_table(thenet)
    aux.description  = topology
    aux.first_count  = first_count
    aux.names_prefix = names_prefix
    aux.names_order  = names_order
    return thenet
  end

-------------------------------------------------------------------

function ann.mlp.all_all.save(model, filename, mode, old)
  error("DEPRECATED: this method is deprecated, please use standard ANN "..
	  "components and trainable.supervised_trainer objects")
end

-------------------------------------------------------------------

ann.mlp.all_all.load = april_doc{
  class="function",
  summary="Loads an all-all stacked ANN model",
  description={
    "This function loads an all-all ANN model. It only works",
    "with models generated via ann.mlp.all_all.generate",
    "function.",
  },
  params= {
    { "A filename string" },
  },
  outputs = {
    "An all-all ANN model"
  }
} ..
-- this function is for retro-compatibility
  function(filename)
    fprintf(io.stderr, "DEPRECATED: this method is deprecated, please use standard ANN "..
              "components and trainable.supervised_trainer objects\n")
    local c     = loadfile(filename)
    local data  = c()
    local model = ann.mlp.all_all.generate(data[1], data.first_count, data.prefix)
    local w     = data[2]
    local oldw  = data[3] or w
    local _,weights_table,_ = model:build()
    local names_order = get_lua_properties_table(model).names_order
    local pos = 0
    for i=1,#names_order,2 do
      local bname   = names_order[i]
      local wname   = names_order[i+1]
      local bias    = weights_table[bname]
      local weights = weights_table[wname]
      local colsize = ann.connections.get_input_size(weights) + 1
      ann.connections.load(bias, { w=w, oldw=oldw, first_pos=pos, column_size=colsize })
      pos = ann.connections.load(weights, { w=w, oldw=oldw,
                                            first_pos=pos+1, column_size=colsize }) - 1
    end
    return model
  end

---------------------------
-- BINDING DOCUMENTATION --
---------------------------

april_set_doc(ann,
	      {
		class="namespace",
		summary="Namespace which contains all ANN related classes",
	      })

-------------------------------------------------------------------

april_set_doc(ann.connections,
	      {
		class="namespace",
		summary="Connections namespace, stores helper functions",
	      })

-------------------------------------------------------------------

april_set_doc(ann.connections.input_filters_image,
	      {
		class="function",
		summary="Builds an image with the filters in the given weights matrix",
		params={
		  { "The weights matrix" },
		  { "The shape of the inputs, a table with 2 or 3 components" },
		  { "The margin between each filter [optional], by default it is 1" },
		  { "No transpose [optional], by default it is nil" },
		},
		outputs={
		  "A squared image (instance of Image) containing all the filters",
		}
	      })

-------------------------------------------------------------------

april_set_doc(ann.connections,
	      {
		class="function",
		summary="Builds a matrix for connections",
		description=
		  {
		    "The constructor reserves memory for the given input and",
		    "output sizes. The weights are in row-major from the outside,",
		    "but internally they are stored in col-major order.",
		  },
		params={
		  ["input"] = "Input size (number of rows).",
		  ["output"] = "Output size (number of cols).",
		},
		outputs = { "An instance of ann.connections" }
	      })

april_set_doc(ann.connections,
	      {
		class="function",
		summary="Builds a matrix for connections",
		description=
		  {
		    "The constructor reserves memory for the given input and",
		    "output sizes. It loads a matrix with",
		    "weights trained previously, or computed with other",
		    "toolkits. The weights are in row-major from the outside,",
		    "but internally they are stored in col-major order.",
		  },
		params={
		  ["input"] = "Input size (number of rows).",
		  ["output"] = "Output size (number of cols).",
		  ["w"] = "A matrix with enough number of data values.",
		  ["oldw"] = "A matrix used to compute momentum (same size of w) [optional]",
		  ["first_pos"] = "Position of the first weight on the given "..
		    "matrix w [optional]. By default is 0",
		  ["column_size"] = "Leading size of the weights [optional]. "..
		    "By default is input"
		},
		outputs = { "An instance of ann.connections" }
	      })

-------------------------------------------------------------------

april_set_doc(ann.connections.randomize_weights,
	      {
		class="function",
		summary="Initializes following uniform random distribution: [inf,sup]",
		params={
		  "A weights matrix",
		  "A table with fields: random, inf and sup",
		},
	      })

-------------------------------------------------------------------
-------------------------------------------------------------------
-------------------------------------------------------------------

april_set_doc(ann.components,
	      {
		class="namespace",
		summary="Namespace which all ANN components classes",
	      })

-------------------------------------------------------------------

april_set_doc(ann.components.base,
	      {
		class="class",
		summary="ANN component parent class",
		description=
		  {
		    "ANN components are the blocks used to build neural networks.",
		    "Each block has a name property which serves as unique",
		    "identifier.",
		    "Besides, the property weights_name is a non unique",
		    "identifier of the ann.connections object property of a",
		    "given ann.components object.",
		    "Each component has a number of inputs and a number of",
		    "outputs.",
		    "Components has options (as learning_rate, momentum, ...)",
		    "which modify they behaviour.",
		    "Tokens are the basic data which components interchange.",
                    "Matrix types are a kind of Token, so it is transparent.",
		    "The ANNs are trained following gradient descent algorithm,",
		    "so each component has four main properties: input, output,",
		    "error_input and error_output.",
		    "All classes inside the table ann.components are child of",
		    "this superclass",
		  },
	      })

-------------------------------------------------------------------

april_set_doc(ann.components.base,
	      {
		class="method",
		summary="Constructor",
		description=
		  {
		    "The base component is a dummy component which implements",
		    "identity function.",
		    "This is the parent class for the rest of components.",
		  },
		params = {
		  ["name"] = "The unique identifier of the component "..
		    "[optional]. By default it is generated automatically.",
		  ["weights"] = "The non unique identifier of its "..
		    "ann.connections property [optional]. By default is nil.",
		  ["size"] = "Input and output size, are the same [optional]. "..
		    "By default is 0. This size could be overwritten at "..
		    "build method.",
		},
		outputs = {
		  "An instance of ann.components.base"
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."precompute_output_size",
	      {
		class="method",
		summary="Precomputes the shape of the output (a table)",
		params = {
		  "A table with the input's shape (a table) [optional]",
		},
		outputs = {
		  "A table with the output's shape",
		}
	      })

-------------------------------------------------------------------

april_set_doc(ann.components.base.."get_is_built",
	      {
		class="method",
		summary="Returns the build state of the object",
		outputs = {
		  "A boolean with the build state"
		}
	      })

----------------------------------------------------------------------
 
april_set_doc(ann.components.base.."get_input_size",
	      {
		class="method",
		summary="Returns the size INPUT",
		outputs = {
		  "A number with the size",
		}
	      })

-------------------------------------------------------------------

april_set_doc(ann.components.base.."get_output_size",
	      {
		class="method",
		summary="Returns the size OUTPUT",
		outputs = {
		  "A number with the size",
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."get_input",
	      {
		class="method",
		summary="Returns the token at component input",
		outputs = {
		  "A token or nil (usually a matrix)",
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."get_output",
	      {
		class="method",
		summary="Returns the token at component output",
		outputs = {
		  "A token or nil (usually a matrix)",
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."get_error_input",
	      {
		class="method",
		summary="Returns the token at component error input",
		description={
		  "The error input is the gradient incoming from",
		  "next component(s). Note that the error input comes",
		  "in reverse order (from the output)."
		},
		outputs = {
		  "A token or nil (usually a matrix)",
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."get_error_output",
	      {
		class="method",
		summary="Returns the token at component error output",
		description={
		  "The error output is the gradient going to",
		  "previous component(s). Note that the error output goes",
		  "in reverse order (to the input).",
		},
		outputs = {
		  "A token or nil (usually a matrix)",
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."forward",
	      {
		class="method",
		summary="Computes forward step with the given token",
		params={
		  "An input token (usually a matrix)",
		  { "A boolean indicating if the forward is during_training or not.",
		    "This information is used by ann.components.actf objects to",
		    "apply dropout during training, and to halve the activation",
		    "during validation (or test). It is [optional], by default",
		    "is false.", }
		},
		outputs = {
		  "An output token (usually a matrix)",
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."backprop",
	      {
		class="method",
		summary="Computes gradient step (backprop) with the given error input",
		description={
		  "Computes gradient step (backprop) with the given error input.",
		  "This method is only valid after forward."
		},
		params={
		  "An error input token (usually a matrix)"
		},
		outputs = {
		  "An error output token (usually a matrix)",
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."reset",
	      {
		class="method",
		summary="Reset all stored tokens",
		description={
		  "This method resets all stored tokens as property of this",
		  "component. Input, output, error input and error output",
		  "tokens are set to nil",
		},
		params = {
		  { "An optional number with the current iteration. It is",
		    "used by iterative algorithms as Conjugate Gradient",
		    "to indicate components current iteration number",
		    "[optional]. By default it is 0",
		  }
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."clone",
	      {
		class="method",
		summary="Makes a deep-copy of the component, except connections",
		description={
		  "Makes a deep-copy of the component, except connections.",
		  "All component options are cloned, except use_cuda flag.",
		},
		outputs={
		  "A new instance of ann.components"
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."set_use_cuda",
	      {
		class="method",
		summary="Modifies use_cuda flag",
		description={
		  "Sets the use_cuda flag. If use_cuda=true then all the",
		  "computation will be done at GPU.",
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."build",
	      {
		class="method",
		summary="This method needs to be called after component creation",
		description={
		  "Components can be composed in a hierarchy interchanging",
		  "input/output tokens. After components composition, it is",
		  "mandatory to call this method.",
		  "It reserves memory necessary for connections and setup",
		  "auxiliary data structures. Connection weights are not valid",
		  "before calling build. Build methods of components are",
		  "automatically called recursively. A component can be",
		  "built twice, but is mandatory to call reset_connections()",
		  "method before each additional build (or always if doubt).",
		},
		params = {
		  ["input"] = {"Input size of the component [optional]. By",
			       "default it is input size given at constructor."},
		  ["output"] = {"Output size of the component [optional]. By",
				"default it is output size given at constructor."},
		  ["weights"] = {"A dictionary table ",
				 "weights_name=>ann.connections object.",
				 "If the corresponding weights_name is found,",
				 "the connections property of the",
				 "component is assigned to table value.",
				 "Otherwise, connections property is new reserved", },
		  
		},
		outputs= {
		  { "The caller ANN component."},
		  { "A table with all the weights_name=>ann.connections found",
		    "at the components hierarchy."},
		  { "A table with all the name=>ann.components found",
		    "at the hierarchy."},
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."debug_info",
	      {
		class="method",
		summary="Debug info at screen",
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."copy_weights",
	      {
		class="method",
		summary="Returns the dictionary weights_name=>ann.connections",
		outputs= {
		  { "A matrix table weights_name=>matrix with matrices found",
		    "at the components hierarchy."},
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."copy_components",
	      {
		class="method",
		summary="Returns the dictionary name=>ann.components",
		outputs= {
		  { "A table with all the name=>ann.components found",
		    "at the components hierarchy."},
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."get_component",
	      {
		class="method",
		summary="Returns the ann.component with the given name property",
		params={
		  "An string with the given name"
		},
		outputs= {
		  { "An ann.components which has the given name",
		    "at the components hierarchy."},
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."get_name",
	      {
		class="method",
		summary="Returns the name of this ann.component",
		outputs= {
		  { "A string with the name."},
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."get_weights_name",
	      {
		class="method",
		summary="Returns the weigths_name of this ann.component",
		outputs= {
		  { "A string with the weigths_name."},
		}
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.base.."has_weights_name",
	      {
		class="method",
		summary="Indicates if this component has connection weights object",
		outputs= {
		  { "A boolean"},
		}
	      })

-------------------------------
--        COMPONENTS         --
-------------------------------

april_set_doc(ann.components.dot_product, {
		class="class",
		summary="A component which implements output = input x weights",
		description = {
		  "The dot product component implements a dot product between",
		  "its input and a weights matrix. If the input is a bunch",
		  "(more than one pattern), a matrix-matrix product will be",
		  "done, as many outputs as inputs.",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.dot_product,
	      {
		class="method",
		summary="Constructor of the component",
		description = {
		  "All the paremeters are",
		  "[optional]. Names are generated automatically if non given.",
		  "Sizes are set to zero by default, so will be mandatory to",
		  "indicate the sizes at build method.",
		},
		params={
		  ["name"] = "A string with the given name [optional]",
		  ["weights"] = {
		    "A string with the weights name, two components with",
		    "the same weights name share the weights matrix [optional]", },
		  ["input"] = "Number of component input neurons [optional]",
		  ["output"] = "Number of component output neurons [optional]",
		  ["transpose"] = {
		    "Indicates if the matrix is transposed before dot/matrix",
		    "product [optional]. By default is false", }
		},
		outputs= { "An instance of ann.components.dot_product" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.bias, {
		class="class",
		summary="A component which implements output = input + bias",
		description = {
		  "The bias component implements the addition of bias to",
		  "its input. If the input is a bunch",
		  "(more than one pattern), bias is added to all of them,",
		  "producing as many outputs as inputs.",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.bias,
	      {
		class="method",
		summary="Constructor of the component",
		description = {
		  "All the paremeters are",
		  "[optional]. Names are generated automatically if non given.",
		},
		params={
		  ["name"] = "A string with the given name [optional]",
		  ["weights"] = {
		    "A string with the weights name, two components with",
		    "the same weights name share the weights matrix [optional]", },
		  ["size"]   = "A number with the size [optional]",
		},
		outputs= { "An instance of ann.components.bias" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.hyperplane, {
		class="class",
		summary="A component which implements output=input*weigths + bias",
		description = {
		  "This component is composed by an ann.components.dot_product",
		  "and a ann.components.bias.",
		  "It implements an standard ANN layer computation:",
		  "output = input * weights + bias.",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.hyperplane,
	      {
		class="method",
		summary="Constructor of the component",
		description = {
		  "All the paremeters are",
		  "[optional]. Names are generated automatically if non given.",
		  "Sizes are set to zero by default, so will be mandatory to",
		  "indicate the sizes at build method.",
		},
		params={
		  ["name"] = "A string with the given name [optional]",
		  ["dot_product_name"] = {
		    "A string with the name for the dot_product",
		    "component [optional]", },
		  ["bias_name"] = {
		    "A string with the name for the bias",
		    "component [optional]", },
		  ["dot_product_weights"] = {
		    "A string with the weights name for the dot_product",
		    "component [optional]", },
		  ["bias_weights"] = {
		    "A string with the weights name for the bias",
		    "component [optional]", },
		  ["input"] = "Number of component input neurons [optional]",
		  ["output"] = "Number of component output neurons [optional]",
		  ["transpose"] = {
		    "Indicates if the dot_product matrix is transposed",
		    "[optional]. By default is false", }
		},
		outputs= { "An instance of ann.components.hyperplane" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.stack, {
		class="class",
		summary="A container component for stack multiple components",
		description = {
		  "This component is a container composed by several",
		  "stacked components, so the output of one component is the",
		  "input of the next."
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.stack,
	      {
		class="method",
		summary="Constructor of the component",
		description = {
		  "The Name is generated automatically if non given.",
		},
		params={
		  ["name"] = "A string with the given name [optional]",
		},
		outputs= { "An instance of ann.components.stack" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.stack.."push",
	      {
		class="method",
		summary="Pushes a list of components to the stack",
		params={
		  "An instance of ann.components.base (or any child class)",
		  "An instance of ann.components.base (or any child class)",
		  "...",
		  "An instance of ann.components.base (or any child class)",
		},
		outputs = { "The caller object" },
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.stack.."unroll",
	      {
		class="method",
		summary="Returns the list of components of the stack",
		outputs = {
		  "An instance of ann.components.base",
		  "An instance of ann.components.base",
		  "...",
		  "An instance of ann.components.base",
		},
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.stack.."get",
	      {
		class="method",
		summary="Returns the components of the stack at the given indexes",
		params = {
		  "Index of component",
		  "Index of component",
		  "...",
		  "Index of component",
		},
		outputs = {
		  "An instance of ann.components.base",
		  "An instance of ann.components.base",
		  "...",
		  "An instance of ann.components.base",
		},
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.stack.."pop",
	      {
		class="method",
		summary="Pops the top component of the stack",
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.stack.."top",
	      {
		class="method",
		summary="Returns the top component of the stack",
		outputs={ "An instance of ann.components.base" },
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.join, {
		class="class",
		summary="A container component for join multiple components",
		description = {
		  "This component is a container composed by several",
		  "joined components, so the input is fragmented given each",
		  "fragment to only one component, and the output of every",
		  "component is join to compose a unique output token",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.join,
	      {
		class="method",
		summary="Constructor of the component",
		description = {
		  "The Name is generated automatically if non given.",
		},
		params={
		  ["name"] = "A string with the given name [optional]",
		},
		outputs= { "An instance of ann.components.join" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.join.."add",
	      {
		class="method",
		summary="Adds a component to the join",
		params={
		  "An instance of ann.components.base (or any child class)",
		},
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.copy, {
		class="class",
		summary="A dummy component for copy multiple times its input",})

----------------------------------------------------------------------

april_set_doc(ann.components.copy,
	      {
		class="method",
		summary="Constructor of the component",
		description = {
		  "The Name is generated automatically if non given.",
		  "Sizes are set to zero by default, so it is mandatory to give",
		  "it at build method if not given at the constructor."
		},
		params={
		  ["times"] = {
		    "A number indicating how many times the",
		    "input is replicated", },
		  ["name"] = "A string with the given name [optional]",
		  ["input"] = "Number of component input neurons [optional]",
		  ["output"] = "Number of component output neurons [optional]",
		},
		outputs= { "An instance of ann.components.copy" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.select, {
		class="class",
		summary="A component for select operation (see matrix.select)",})

----------------------------------------------------------------------

april_set_doc(ann.components.select,
	      {
		class="method",
		summary="Constructor of the component select (see matrix.select)",
		description = {
		  "The Name is generated automatically if non given.",
		},
		params={
		  ["name"] = "A string with the given name [optional]",
		  ["dimension"] = "Number with the selected dimension",
		  ["index"] = "Number with the selected index",
		},
		outputs= { "An instance of ann.components.select" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.rewrap, {
		class="class",
		summary="A component for change the dimension of matrix (see matrix.rewrap)",})

----------------------------------------------------------------------

april_set_doc(ann.components.rewrap,
	      {
		class="method",
		summary="Constructor of the component rewrap (see matrix.rewrap)",
		description = {
		  "The Name is generated automatically if non given.",
		},
		params={
		  ["name"] = "A string with the given name [optional]",
		  ["size"] = {
		    "A table with the sizes of re-wrapped matrix dimensions.",
		  }
		},
		outputs= { "An instance of ann.components.rewrap" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.slice, {
		class="class",
		summary="A component which takes a slice of a matrix (see matrix.slice)",})

----------------------------------------------------------------------

april_set_doc(ann.components.slice,
	      {
		class="method",
		summary="Constructor of the component slice (see matrix.slice)",
		description = {
		  "The name is generated automatically if non given.",
		},
		params={
		  ["name"] = "A string with the given name [optional]",
		  ["pos"] = {
		    "A table with the coordinates of the slice.",
		  },
		  ["size"] = {
		    "A table with the slice size.",
		  }
		},
		outputs= { "An instance of ann.components.slice" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.stochastic, {
		class="class",
		summary="An abstract component which implements basic interface of stochastic components",})

april_set_doc(ann.components.stochastic.."get_random",
	      {
		class="method",
		summary="Returns the underlying random object",
		outputs={ "A random object" },
	      })

april_set_doc(ann.components.stochastic.."set_random",
	      {
		class="method",
		summary="Sets the underlying random object",
		params={ "A random object" },
		outputs={ "The caller object" },
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.gaussian_noise, {
		class="class",
		summary="A component which adds Gaussian noise to data",})

april_set_doc(ann.components.gaussian_noise,
	      {
		class="method",
		summary="Constructor of the component",
		description = {
		  "The Name is generated automatically if non given.",
		  "Size is set to zero by default, so it is mandatory to give",
		  "it at build method if not given at the constructor."
		},
		params={
		  ["name"]   = "A string with the given name [optional]",
		  ["size"]   = "A number with the size [optional]",
		  ["mean"]   = "Mean of the gaussian noise [optional], by default it is 0",
		  ["var"]    = "Variance of the gaussian noise [optional], by default it is 0.1",
		  ["random"] = "Random object instance [optional]",
		},
		outputs= { "An instance of ann.components.gaussian_noise" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.salt_and_pepper, {
		class="class",
		summary="A component which adds salt and pepper noise to data",})

april_set_doc(ann.components.salt_and_pepper,
	      {
		class="method",
		summary="Constructor of the component",
		description = {
		  "The Name is generated automatically if non given.",
		  "Size is set to zero by default, so it is mandatory to give",
		  "it at build method if not given at the constructor."
		},
		params={
		  ["name"]   = "A string with the given name [optional]",
		  ["size"]   = "A number with the size [optional]",
		  ["prob"]   = "Probability of noise [optional], by default it is 0.2",
		  ["zero"]   = "Value of ZERO [optional], by default it is 0",
		  ["one"]    = "Value of ONE [optional], by default it is 1",
		  ["random"] = "Random object instance [optional]",
		},
		outputs= { "An instance of ann.components.salt_and_pepper" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.dropout, {
		class="class",
		summary="A component which adds salt and pepper noise to data",})

april_set_doc(ann.components.dropout,
	      {
		class="method",
		summary="Constructor of the component",
		description = {
		  "The Name is generated automatically if non given.",
		  "Size is set to zero by default, so it is mandatory to give",
		  "it at build method if not given at the constructor."
		},
		params={
		  ["name"]   = "A string with the given name [optional]",
		  ["size"]   = "A number with the size [optional]",
		  ["prob"]   = "Probability of noise [optional], by default it is 0.5",
		  ["value"]  = "Mask value [optional], by default it is 0",
		  ["random"] = "Random object instance [optional]",
                  ["norm"] = "Boolean value indicating if apply normalization when not training [optional] by default it is true",
		},
		outputs= { "An instance of ann.components.dropout" }
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.actf,
	      {
		class="namespace",
		summary="Namespace which contains all activation functions",
	      })

----------------------------------------------------------------------

april_set_doc(ann.components.actf, {
		class="class",
		summary="Abstract class child of ann.components.base",
		description={
		  "This class is used as parent of all activation functions.",
		  "Activation functions input/output sizes are equal,",
		  "and they are set depending on previous components",
		  "or at the build method.",
		  "This parent class implements dropout technique to improve",
		  "training of very large neural networks.",
		  "Dropout must be applied ONLY to activation function",
		  "components by using set_option method (other components",
		  "may abort the program with error).",
		  "Dropout is forbidden at OUTPUT activation components, so",
		  "don't set it indiscrimately. It must be applied component",
		  "by component."
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.logistic, {
		class="class",
		summary="Logistic (or sigmoid) activation function", })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.logistic, {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.tanh, {
		class="class",
		summary="Tanh activation function", })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.tanh, {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.softsign, {
		class="class",
		summary="Softsign activation function", })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.softsign, {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })


----------------------------------------------------------------------

april_set_doc(ann.components.actf.log_logistic, {
		class="class",
		summary="Logarithm of logistic activation function",
		description={
		  "This activation function only works if the loss function is",
		  "cross-entropy or multi-class cross-entropy",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.log_logistic, {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.softmax, {
		class="class",
		summary="Softmax activation function",
		description={
		  "This activation function is computed over each input",
		  "pattern and ensures that outputs are probabilities."
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.softmax, {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })


----------------------------------------------------------------------

april_set_doc(ann.components.actf.log_softmax, {
		class="class",
		summary="Logarithm of softmax activation function",
		description={
		  "This activation function is computed over each input",
		  "pattern and ensures that outputs are probabilities.",
		  "It only works with multi-class cross-entropy loss function.",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.log_softmax, {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })


----------------------------------------------------------------------

april_set_doc(ann.components.actf.softplus, {
		class="class",
		summary="Softplus activation function", })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.softplus, {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.relu, {
		class="class",
		summary="Rectifier Linear Unit (ReLU) activation function", })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.relu, {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.sin, {
		class="class",
		summary="Sin activation function", })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.sin, {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.linear, {
		class="class",
		summary="Linear activation function", })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.linear, {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.hardtanh, {
		class="class",
		summary="Hardtanh activation function", })

----------------------------------------------------------------------

april_set_doc(ann.components.actf.hardtanh, {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		  ["inf"] = "The name of the component [optional], by default is -1.0.",
		  ["sup"] = "The name of the component [optional], by default is 1.0.",
		}, })
