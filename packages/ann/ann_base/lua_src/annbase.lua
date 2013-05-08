get_table_from_dotted_string("ann.mlp.all_all", true)

----------------------------------------------------------------------

april_set_doc("ann.mlp.all_all",
	      {
		class="function",
		summary="Function to build all-all stacked ANN models",
		description=
		  {
		    "This function composes a component object from the",
		    "given topology description (stacked all-all).",
		    "It generates default names for components and connection",
		    "weights. Each layer has one ann.components.dot_product",
		    "with name='w'..NUMBER and weights_name='w'..NUMBER,",
		    "one ann.components.bias with name='b'..NUMBER and",
		    "weights_name='b'..NUMBER, and an ann.components.actf with",
		    "name='actf'..NUMBER.",
		    "NUMBER is a counter initialized at 1, or with the",
		    "value of second argument (count) for",
		    "ann.mlp.all_all(topology, count) if it is given.",
		  },
		params= {
		  { "Topology description string as ",
		    "'1024 inputs 128 logistc 10 log_softmax" },
		  { "First count parameter (count) ",
		    "[optional]. By default 1." },
		},
		outputs= {
		  {"A component object with the especified ",
		   "neural network topology" }
		}
	      })

function ann.mlp.all_all.generate(topology, first_count)
  local thenet = ann.components.stack()
  local name   = "layer"
  local count  = first_count or 1
  local t      = string.tokenize(topology)
  local prev_size = tonumber(t[1])
  for i=3,#t,2 do
    local size = tonumber(t[i])
    local actf = t[i+1]
    thenet:push( ann.components.hyperplane{
		   input=prev_size, output=size,
		   bias_weights="b" .. count,
		   dot_product_weights="w" .. count,
		   name="layer" .. count,
		   bias_name="b" .. count,
		   dot_product_name="w" .. count } )
    if not ann.components.actf[actf] then
      error("Incorrect activation function: " .. actf)
    end
    thenet:push( ann.components.actf[actf]{ name = "actf" .. count } )
    count = count + 1
    prev_size = size
  end
  local obj = {
    topology    = topology,
    first_count = first_count,
    thenet      = thenet, }
  -- we make obj a wrapper of thenet, so we keep useful information as a lua
  -- table
  local current = thenet
  while (getmetatable(current) and getmetatable(current).__index and
	 getmetatable(current).__index ~= current) do
    current = getmetatable(current).__index
    for i,v in pairs(current) do
      if type(v) == "function" then
	obj[i] =
	  function(...)
	    local obj = arg[1]
	    table.remove(arg, 1)
	    return obj.thenet[i](obj.thenet, unpack(arg))
	  end
      end
    end
  end
  return obj
end

function ann.mlp.all_all.save(model, filename, mode, old)
  if type(model) ~= "table" or not model.topology then
    error ("Incorrect ANN mode!!!")
  end
  error("NOT IMPLEMENTED")
end

function ann.mlp.all_all.load(filename, bunch_size)
  error("NOT IMPLEMENTED")
  local c = loadfile(filename)
  local data = c()
  return c and ann.mlp.all_all.generate{
    topology    = data[1],
    w           = data[2],
    oldw        = data[3],
    bunch_size  = bunch_size,
  }
end


---------------------------
-- BINDING DOCUMENTATION --
---------------------------

april_set_doc("ann",
	      {
		class="namespace",
		summary="Namespace which contains all ANN related classes",
	      })

-------------------------------------------------------------------

april_set_doc("ann.connections",
	      {
		class="class",
		summary="Connections class, stores weights and useful methods",
		description={
		    "The ann.connections class is used at ann.components ",
		    "objects to store weights when needed. This objects have",
		    "an ROWSxCOLS matrix of float parameters, being ROWS",
		    "the input size of a given component, and COLS the output",
		    "size.",
		  },
	      })

-------------------------------------------------------------------

april_set_doc("ann.connections.__call",
	      {
		class="method",
		summary="Constructor",
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

april_set_doc("ann.connections.__call",
	      {
		class="method",
		summary="Constructor",
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

april_set_doc("ann.connections.clone",
	      {
		class="method",
		summary="Makes a deep copy of the object",
		outputs = { "An instance of ann.connections" }
	      })

-------------------------------------------------------------------

april_set_doc("ann.connections.load",
	      {
		class="method",
		summary="Load weights from a matrix",
		description=
		  {
		    "The method load connection weights from a matrix.",
		    "The weights are in row-major from the outside,",
		    "but internally they are stored in col-major order.",
		  },
		params={
		  ["w"] = "A matrix with enough number of data values.",
		  ["oldw"] = "A matrix used to compute momentum (same size of w) [optional]",
		  ["first_pos"] = "Position of the first weight on the given "..
		    "matrix w [optional]. By default is 0",
		  ["column_size"] = "Leading size of the weights [optional]. "..
		    "By default is input"
		},
		outputs = { "A number: first_pos + column_size*OUTPUT" }
	      })

-------------------------------------------------------------------

april_set_doc("ann.connections.weights",
	      {
		class="method",
		summary="Returns the weights as a matrix",
		description=
		  {
		    "The method copies the weights to the given matrixes or",
		    "create new ones to store the data.",
		    "Note that matrixes are a copy, so any modification won't",
		    "affect to connections object.",
		  },
		params={
		  ["w"] = "A matrix with enough number of data values.",
		  ["oldw"] = "A matrix used to compute momentum (same size of w) [optional]",
		  ["first_pos"] = "Position of the first weight on the given "..
		    "matrix w [optional]. By default is 0",
		  ["column_size"] = "Leading size of the weights [optional]. "..
		    "By default is input"
		},
		outputs = {
		  "A matrix with the weights",
		  "A matrix with the weights of previous iteration (for momentum)",
		  "A number: first_pos + column_size*OUTPUT",
		}
	      })

-------------------------------------------------------------------

april_set_doc("ann.connections.size",
	      {
		class="method",
		summary="Returns the size INPUTxOUTPUT",
		outputs = {
		  "A number with the size",
		}
	      })

-------------------------------------------------------------------

april_set_doc("ann.connections.get_input_size",
	      {
		class="method",
		summary="Returns the size INPUT",
		outputs = {
		  "A number with the size",
		}
	      })

-------------------------------------------------------------------

april_set_doc("ann.connections.get_output_size",
	      {
		class="method",
		summary="Returns the size OUTPUT",
		outputs = {
		  "A number with the size",
		}
	      })

-------------------------------------------------------------------

april_set_doc("ann.connections.randomize_weights",
	      {
		class="method",
		summary="Initializes following uniform random distribution: [inf,sup]",
		params={
		  ["random"] = "A random object",
		  ["inf"] = "Inferior limit [optional]. By default is -1.0",
		  ["first_pos"] = "Superior limit [optional]. By default is -1.0",
		},
	      })

-------------------------------------------------------------------
-------------------------------------------------------------------
-------------------------------------------------------------------

april_set_doc("ann.components",
	      {
		class="namespace",
		summary="Namespace which all ANN components classes",
	      })

-------------------------------------------------------------------

april_set_doc("ann.components.base",
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
		    "The ANNs are trained following gradient descent algorithm,",
		    "so each component has four main properties: input, output,",
		    "error_input and error_output.",
		    "All classes inside the table ann.components are child of",
		    "this superclass",
		  },
	      })

-------------------------------------------------------------------

april_set_doc("ann.components.base.__call",
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

april_set_doc("ann.components.base.get_is_built",
	      {
		class="method",
		summary="Returns the build state of the object",
		outputs = {
		  "A boolean with the build state"
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.set_option",
	      {
		class="method",
		summary="Changes an option of the component",
		description=
		  {
		    "This method changes the value of an option.",
		    "Not all components implement the same options.",
		  },
		params = {
		  "A string with the name of the option",
		  "A number with its value"
		},
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.get_option",
	      {
		class="method",
		summary="Returns the value of a given option name",
		params = {
		  "A string with the name of the option",
		},
		outputs = {
		  "A number with its value"
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.has_option",
	      {
		class="method",
		summary="Returns true/false if the option is valid",
		params = {
		  "A string with the name of the option",
		},
		outputs = {
		  "A boolean"
		}
	      })

----------------------------------------------------------------------
 
april_set_doc("ann.components.base.get_input_size",
	      {
		class="method",
		summary="Returns the size INPUT",
		outputs = {
		  "A number with the size",
		}
	      })

-------------------------------------------------------------------

april_set_doc("ann.components.base.get_output_size",
	      {
		class="method",
		summary="Returns the size OUTPUT",
		outputs = {
		  "A number with the size",
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.get_input",
	      {
		class="method",
		summary="Returns the token at component input",
		outputs = {
		  "A token or nil",
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.get_output",
	      {
		class="method",
		summary="Returns the token at component output",
		outputs = {
		  "A token or nil",
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.get_error_input",
	      {
		class="method",
		summary="Returns the token at component error input",
		description={
		  "The error input is the gradient incoming from",
		  "next component(s). Note that the error input comes",
		  "in reverse order (from the output)."
		},
		outputs = {
		  "A token or nil",
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.get_error_output",
	      {
		class="method",
		summary="Returns the token at component error output",
		description={
		  "The error output is the gradient going to",
		  "previous component(s). Note that the error output goes",
		  "in reverse order (to the input).",
		},
		outputs = {
		  "A token or nil",
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.forward",
	      {
		class="method",
		summary="Computes forward step with the given token",
		params={
		  "An input token"
		},
		outputs = {
		  "An output token",
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.backprop",
	      {
		class="method",
		summary="Computes gradient step (backprop) with the given error input",
		description={
		  "Computes gradient step (backprop) with the given error input.",
		  "This method is only valid after forward."
		},
		params={
		  "An error input token"
		},
		outputs = {
		  "An error output token",
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.update",
	      {
		class="method",
		summary="Updates connection weights of the component",
		description={
		  "Updates connection weights of the component.",
		  "This method is only valid after forward and backprop calls."
		},
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.reset",
	      {
		class="method",
		summary="Reset all stored tokens",
		description={
		  "This method resets all stored tokens as property of this",
		  "component. Input, output, error input and error output",
		  "tokens are set to nil",
		},
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.clone",
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

april_set_doc("ann.components.base.set_use_cuda",
	      {
		class="method",
		summary="Modifies use_cuda flag",
		description={
		  "Sets the use_cuda flag. If use_cuda=true then all the",
		  "computation will be done at GPU.",
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.reset_connections",
	      {
		class="method",
		summary="Set to zero the count reference of shared connections",
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.build",
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
		  { "A table with all the weights_name=>ann.connections found",
		    "at the components hierarchy."},
		  { "A table with all the name=>ann.components found",
		    "at the hierarchy."},
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.copy_weights",
	      {
		class="method",
		summary="Returns the dictionary weights_name=>ann.connections",
		outputs= {
		  { "A table with all the weights_name=>ann.connections found",
		    "at the components hierarchy."},
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.copy_components",
	      {
		class="method",
		summary="Returns the dictionary name=>ann.components",
		outputs= {
		  { "A table with all the name=>ann.components found",
		    "at the components hierarchy."},
		}
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.base.get_component",
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

-------------------------------
--        COMPONENTS         --
-------------------------------

april_set_doc("ann.components.dot_product", {
		class="class",
		summary="A component which implements output = input x weights",
		description = {
		  "The dot product component implements a dot product between",
		  "its input and a weights matrix. If the input is a bunch",
		  "(more than one pattern), a matrix-matrix product will be",
		  "done, as many outputs as inputs.",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.dot_product.__call",
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

april_set_doc("ann.components.bias", {
		class="class",
		summary="A component which implements output = input + bias",
		description = {
		  "The bias component implements the addition of bias to",
		  "its input. If the input is a bunch",
		  "(more than one pattern), bias is added to all of them,",
		  "producing as many outputs as inputs.",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.bias.__call",
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
		},
		outputs= { "An instance of ann.components.bias" }
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.hyperplane", {
		class="class",
		summary="A component which implements output=input*weigths + bias",
		description = {
		  "This component is composed by an ann.components.dot_product",
		  "and a ann.components.bias.",
		  "It implements an standard ANN layer computation:",
		  "output = input * weights + bias.",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.hyperplane.__call",
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

april_set_doc("ann.components.stack", {
		class="class",
		summary="A container component for stack multiple components",
		description = {
		  "This component is a container composed by several",
		  "stacked components, so the output of one component is the",
		  "input of the next."
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.stack.__call",
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

april_set_doc("ann.components.stack.push",
	      {
		class="method",
		summary="Pushes a component to the stack",
		params={
		  "An instance of ann.components.base (or any child class)",
		},
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.stack.pop",
	      {
		class="method",
		summary="Pops the top component of the stack",
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.stack.top",
	      {
		class="method",
		summary="Returns the top component of the stack",
		outputs={ "An instance of ann.components.base" },
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.join", {
		class="class",
		summary="A container component for join multiple components",
		description = {
		  "This component is a container composed by several",
		  "joined components, so the input is fragmented given each",
		  "fragment to only one component, and the output of every",
		  "component is join to compose a unique output token",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.join.__call",
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

april_set_doc("ann.components.join.add",
	      {
		class="method",
		summary="Adds a component to the join",
		params={
		  "An instance of ann.components.base (or any child class)",
		},
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.copy", {
		class="class",
		summary="A dummy component for copy multiple times its input",})

----------------------------------------------------------------------

april_set_doc("ann.components.copy.__call",
	      {
		class="method",
		summary="Constructor of the component",
		description = {
		  "The Name is generated automatically if non given.",
		  "Sizes are set to zero by default, so it is mandatory to give",
		  "it at build method is not given at the constructor."
		},
		params={
		  ["times"] = {
		    "A number indicating how many times the",
		    "input is replicated", },
		  ["name"] = "A string with the given name [optional]",
		  ["input"] = "Number of component input neurons [optional]",
		  ["output"] = "Number of component output neurons [optional]",
		},
		outputs= { "An instance of ann.components.join" }
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.actf",
	      {
		class="namespace",
		summary="Namespace which contains all activation functions",
	      })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.__base__", {
		class="class",
		summary="Abstract class child of ann.components.base",
		description={
		  "This class is used as parent of all activation functions.",
		  "Activation functions input/output sizes are equal,",
		  "and they are set depending on previous components",
		  "or at the build method.",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.logistic", {
		class="class",
		summary="Logistic (or sigmoid) activation function", })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.logistic.__call", {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.tanh", {
		class="class",
		summary="Tanh activation function", })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.tanh.__call", {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.softsign", {
		class="class",
		summary="Softsign activation function", })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.softsign.__call", {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })


----------------------------------------------------------------------

april_set_doc("ann.components.actf.log_logistic", {
		class="class",
		summary="Logarithm of logistic activation function",
		description={
		  "This activation function only works if the loss function is",
		  "cross-entropy or multi-class cross-entropy",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.log_logistic.__call", {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.softmax", {
		class="class",
		summary="Softmax activation function",
		description={
		  "This activation function is computed over each input",
		  "pattern and ensures that outputs are probabilities."
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.softmax.__call", {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })


----------------------------------------------------------------------

april_set_doc("ann.components.actf.log_softmax", {
		class="class",
		summary="Logarithm of softmax activation function",
		description={
		  "This activation function is computed over each input",
		  "pattern and ensures that outputs are probabilities.",
		  "It only works with multi-class cross-entropy loss function.",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.log_softmax.__call", {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })


----------------------------------------------------------------------

april_set_doc("ann.components.actf.softplus", {
		class="class",
		summary="Softplus activation function", })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.softplus.__call", {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.sin", {
		class="class",
		summary="Sin activation function", })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.sin.__call", {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.linear", {
		class="class",
		summary="Linear activation function", })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.linear.__call", {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.hardtanh", {
		class="class",
		summary="Hardtanh activation function", })

----------------------------------------------------------------------

april_set_doc("ann.components.actf.hardtanh.__call", {
		class="method",
		summary="Constructor of the component",
		params={
		  ["name"] = "The name of the component [optional].",
		}, })
