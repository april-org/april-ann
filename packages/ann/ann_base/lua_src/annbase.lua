ann.mlp.all_all = get_table_from_dotted_string("ann.mlp.all_all")

----------------------------------------------------------------------

april_set_doc("ann.mlp.all_all",
	      {
		class="function",
		summary="Function to build all-all stacked ANN models",
		description=table.concat(
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
		  }, " "),
		params= { "Topology description string as "..
			    "'1024 inputs 128 logistc 10 log_softmax",
			  "First count parameter (count) "..
			    "[optional]. By default 1."
		},
		outputs= { "A component object with the especified "..
			     "neural network topology" }
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
    if not ann.components[actf] then
      error("Incorrect activation function: " .. actf)
    end
    thenet:push( ann.components[actf]{ name = "actf" .. count } )
    count = count + 1
    prev_size = size
  end
  return thenet
end

---------------------------
-- BINDING DOCUMENTATION --
---------------------------

april_set_doc("ann.connections",
	      {
		class="class",
		summary="Connections class, stores weights and useful methods",
		description=table.concat(
		  {
		    "The ann.connections class is used at ann.components ",
		    "objects to store weights when needed. This objects have",
		    "an ROWSxCOLS matrix of float parameters, being ROWS",
		    "the input size of a given component, and COLS the output",
		    "size.",
		  }, " ")
	      })

-------------------------------------------------------------------

april_set_doc("ann.connections.__call",
	      {
		class="method",
		summary="Constructor",
		description=table.concat(
		  {
		    "The constructor reserves memory for the given input and",
		    "output sizes. The weights are in row-major from the outside,",
		    "but internally they are stored in col-major order.",
		  }, " "),
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
		description=table.concat(
		  {
		    "The constructor reserves memory for the given input and",
		    "output sizes. It loads a matrix with",
		    "weights trained previously, or computed with other",
		    "toolkits. The weights are in row-major from the outside,",
		    "but internally they are stored in col-major order.",
		  }, " "),
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

april_set_doc("ann.connections.__call",
	      {
		class="method",
		summary="Makes a deep copy of the object",
		outputs = { "An instance of ann.connections" }
	      })


//BIND_METHOD Connections load
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1,table);
  check_table_fields(L, 1, "w", "oldw", "first_pos", "column_size", 0);

  unsigned int	 first_pos, column_size;
  MatrixFloat	*w, *oldw;
  
  LUABIND_GET_TABLE_PARAMETER(1, w, MatrixFloat, w);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, oldw, MatrixFloat, oldw, w);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, first_pos, uint, first_pos, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, column_size, uint, column_size,
				       obj->getNumInputs());

  LUABIND_RETURN(uint, obj->loadWeights(w, oldw, first_pos, column_size));
}
//BIND_END

//BIND_METHOD Connections weights
{
  LUABIND_CHECK_ARGN(<=,1);
  LUABIND_CHECK_ARGN(>=,0);
  int nargs;
  LUABIND_TABLE_GETN(1, nargs);
  unsigned int	 first_pos=0, column_size=obj->getNumInputs();
  MatrixFloat	*w=0, *oldw=0;
  
  if (nargs == 1) {
    LUABIND_CHECK_PARAMETER(1,table);
    check_table_fields(L, 1, "w", "oldw", "first_pos", "column_size", 0);

    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, w, MatrixFloat, w, w);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, oldw, MatrixFloat, oldw, oldw);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, first_pos, uint, first_pos,
					 first_pos);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, column_size, uint, column_size,
					 column_size);
  }

  int size = static_cast<int>(obj->size());
  if (!w)    w    = new MatrixFloat(1, first_pos + size);
  if (!oldw) oldw = new MatrixFloat(1, w->size);
  
  if (first_pos + obj->size() > static_cast<unsigned int>(w->size) ||
      first_pos + obj->size() > static_cast<unsigned int>(oldw->size) )
    LUABIND_ERROR("Incorrect matrix size!!\n");

  unsigned int sz = obj->copyWeightsTo(w, oldw, first_pos, column_size);
  LUABIND_RETURN(MatrixFloat, w);
  LUABIND_RETURN(MatrixFloat, oldw);
  LUABIND_RETURN(uint, sz);
}
//BIND_END

//BIND_METHOD Connections size
{
  LUABIND_RETURN(uint, obj->size());
}
//BIND_END

//BIND_METHOD Connections get_input_size
{
  LUABIND_RETURN(uint, obj->getInputSize());
}
//BIND_END

//BIND_METHOD Connections get_output_size
{
  LUABIND_RETURN(uint, obj->getOutputSize());
}
//BIND_END

//BIND_METHOD Connections randomize_weights
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "random", "inf", "sup", 0);
  MTRand *rnd;
  float inf, sup;
  bool use_fanin;
  LUABIND_GET_TABLE_PARAMETER(1, random, MTRand, rnd);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, inf, float, inf, -1.0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, sup, float,  sup, 1.0);
  obj->randomizeWeights(rnd, inf, sup);
}
//BIND_END

/////////////////////////////////////////////////////
//                  ANNComponent                   //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME ANNComponent ann.components.base
//BIND_CPP_CLASS    ANNComponent

//BIND_CONSTRUCTOR ANNComponent
//DOC_BEGIN
// base(name)
/// Superclass for ann.components objects. It is also a dummy by-pass object (does nothing with input/output data).
/// @param name A lua string with the name of the component.
//DOC_END
{
  LUABIND_CHECK_ARGN(<=,1);
  int argn = lua_gettop(L);
  const char *name    = 0;
  const char *weights = 0;
  unsigned int size   = 0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", "weights", "size", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, weights, string, weights, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint, size, 0);
  }
  obj = new ANNComponent(name, weights, size, size);
  LUABIND_RETURN(ANNComponent, obj);
}
//BIND_END

//BIND_METHOD ANNComponent get_is_built
{
  LUABIND_RETURN(bool, obj->getIsBuilt());
}
//BIND_END

//BIND_METHOD ANNComponent set_option
//DOC_BEGIN
// set_option(name, value)
/// Method to modify the value of a given option name.
/// @param name A lua string with the name of the option.
/// @param value A lua number with the desired value.
//DOC_END
{
  const char *name;
  double value;
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_GET_PARAMETER(1, string, name);
  LUABIND_GET_PARAMETER(2, double, value);
  obj->setOption(name, value);
}
//BIND_END

//BIND_METHOD ANNComponent get_option
//DOC_BEGIN
// get_option(name)
/// Method to retrieve the value of a given option name.
/// @param name A lua string with the name of the option.
//DOC_END
{
  const char *name;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, string, name);
  LUABIND_RETURN(double, obj->getOption(name));
}
//BIND_END

//BIND_METHOD ANNComponent has_option
//DOC_BEGIN
// has_option(name)
/// Method to ask for the existence of a given option name.
/// @param name A lua string with the name of the option.
//DOC_END
{
  const char *name;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, string, name);
  LUABIND_RETURN(bool, obj->hasOption(name));
}
//BIND_END

//BIND_METHOD ANNComponent get_input_size
{
  LUABIND_RETURN(uint, obj->getInputSize());
}
//BIND_END

//BIND_METHOD ANNComponent get_output_size
{
  LUABIND_RETURN(uint, obj->getOutputSize());
}
//BIND_END

//BIND_METHOD ANNComponent get_input
{
  Token *aux = obj->getInput();
  if (aux == 0)
    LUABIND_RETURN_NIL();
  else LUABIND_RETURN(Token, aux);
}
//BIND_END

//BIND_METHOD ANNComponent get_output
{
  Token *aux = obj->getOutput();
  if (aux == 0)
    LUABIND_RETURN_NIL();
  else LUABIND_RETURN(Token, aux);
}
//BIND_END

//BIND_METHOD ANNComponent get_error_input
{
  Token *aux = obj->getErrorInput();
  if (aux == 0)
    LUABIND_RETURN_NIL();
  else LUABIND_RETURN(Token, aux);
}
//BIND_END

//BIND_METHOD ANNComponent get_error_output
{
  Token *aux = obj->getErrorOutput();
  if (aux == 0)
    LUABIND_RETURN_NIL();
  else LUABIND_RETURN(Token, aux);
}
//BIND_END

//BIND_METHOD ANNComponent forward
{
  Token *input;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, Token, input);
  LUABIND_RETURN(Token, obj->doForward(input, false));
}
//BIND_END

//BIND_METHOD ANNComponent backprop
{
  Token *input;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, Token, input);
  LUABIND_RETURN(Token, obj->doBackprop(input));
}
//BIND_END

//BIND_METHOD ANNComponent update
{
  obj->doUpdate();
}
//BIND_END

//BIND_METHOD ANNComponent reset
{
  obj->reset();
}
//BIND_END

//BIND_METHOD ANNComponent clone
{
  LUABIND_RETURN(ANNComponent, obj->clone());
}
//BIND_END

//BIND_METHOD ANNComponent set_use_cuda
{
  bool use_cuda;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, bool, use_cuda);
  obj->setUseCuda(use_cuda);
}
//BIND_END

//BIND_METHOD ANNComponent build
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  unsigned int input_size=0, output_size=0;
  hash<string,Connections*> weights_dict;
  hash<string,ANNComponent*> components_dict;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "input", "output", "weights", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, input, uint, input_size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, output, uint, output_size, 0);
    lua_getfield(L, 1, "weights");
    if (lua_istable(L, -1)) {
      // stack now contains: -1 => table
      lua_pushvalue(L, -1);
      // stack now contains: -1 => nil; -2 => table
      lua_pushnil(L);
      while (lua_next(L, -2)) {
	// stack now contains: -1 => value; -2 => key; -3 => table
	// copy the key so that lua_tostring does not modify the original
	lua_pushvalue(L, -2);
	// stack now contains: -1 => key; -2 => value; -3 => key; -4 => table
	string key(lua_tostring(L, -1));
	Connections *value = lua_toConnections(L, -2);
	weights_dict[key]  = value;
	// pop value + copy of key, leaving original key
	lua_pop(L, 2);
	// stack now contains: -1 => key; -2 => table
      }
      // stack now contains: -1 => table (when lua_next returns 0 it pops the key
      // but does not push anything.)
      // Pop table
      lua_pop(L, 1);
    }
  }
  //
  obj->build(input_size, output_size, weights_dict, components_dict);
  //
  pushHashTableInLuaStack(L, components_dict, lua_pushANNComponent);
  LUABIND_RETURN_FROM_STACK(-1);
  pushHashTableInLuaStack(L, weights_dict, lua_pushConnections);
  LUABIND_RETURN_FROM_STACK(-2);
}
//BIND_END

//BIND_METHOD ANNComponent copy_weights
{
  hash<string,Connections*> weights_dict;
  obj->copyWeights(weights_dict);
  pushHashTableInLuaStack(L, weights_dict, lua_pushConnections);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD ANNComponent copy_components
{
  hash<string,ANNComponent*> components_dict;
  obj->copyComponents(components_dict);
  pushHashTableInLuaStack(L, components_dict, lua_pushANNComponent);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD ANNComponent get_component
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1,string);
  const char *name;
  LUABIND_GET_PARAMETER(1, string, name);
  string name_string(name);
  ANNComponent *component = obj->getComponent(name_string);
  LUABIND_RETURN(ANNComponent, component);
}
//BIND_END

/////////////////////////////////////////////////////
//             DotProductANNComponent              //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME DotProductANNComponent ann.components.dot_product
//BIND_CPP_CLASS    DotProductANNComponent
//BIND_SUBCLASS_OF  DotProductANNComponent ANNComponent

//BIND_CONSTRUCTOR DotProductANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0, *weights_name=0;
  unsigned int input_size=0, output_size=0;
  bool transpose_weights=false;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", "weights", 
		       "input", "output", "transpose", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, weights, string, weights_name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, input, uint, input_size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, output, uint, output_size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, transpose, bool, transpose_weights,
					 false);
  }
  obj = new DotProductANNComponent(name, weights_name,
				   input_size, output_size,
				   transpose_weights);
  LUABIND_RETURN(DotProductANNComponent, obj);
}
//BIND_END

//BIND_METHOD DotProductANNComponent clone
{
  LUABIND_RETURN(DotProductANNComponent,
		 dynamic_cast<DotProductANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//                BiasANNComponent                 //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME BiasANNComponent ann.components.bias
//BIND_CPP_CLASS    BiasANNComponent
//BIND_SUBCLASS_OF  BiasANNComponent ANNComponent

//BIND_CONSTRUCTOR BiasANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0, *weights_name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", "weights", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, weights, string, weights_name, 0);
  }
  obj = new BiasANNComponent(name, weights_name);
  LUABIND_RETURN(BiasANNComponent, obj);
}
//BIND_END

//BIND_METHOD BiasANNComponent clone
{
  LUABIND_RETURN(BiasANNComponent,
		 dynamic_cast<BiasANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//             HyperplaneANNComponent              //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME HyperplaneANNComponent ann.components.hyperplane
//BIND_CPP_CLASS    HyperplaneANNComponent
//BIND_SUBCLASS_OF  HyperplaneANNComponent ANNComponent

//BIND_CONSTRUCTOR HyperplaneANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  const char *dot_product_name=0,    *bias_name=0;
  const char *dot_product_weights=0, *bias_weights=0;
  unsigned int input_size=0, output_size=0;
  bool transpose_weights=false;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", "dot_product_name", "bias_name",
		       "dot_product_weights", "bias_weights",
		       "input", "output", "transpose", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, dot_product_name, string, dot_product_name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, bias_name, string, bias_name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, dot_product_weights, string, dot_product_weights, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, bias_weights, string, bias_weights, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, input, uint, input_size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, output, uint, output_size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, transpose, bool, transpose_weights,
					 false);
  }
  obj = new HyperplaneANNComponent(name,
				   dot_product_name, bias_name,
				   dot_product_weights, bias_weights,
				   input_size, output_size,
				   transpose_weights);
  LUABIND_RETURN(HyperplaneANNComponent, obj);
}
//BIND_END

//BIND_METHOD HyperplaneANNComponent clone
{
  LUABIND_RETURN(HyperplaneANNComponent,
		 dynamic_cast<HyperplaneANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//              StackANNComponent                  //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME StackANNComponent ann.components.stack
//BIND_CPP_CLASS    StackANNComponent
//BIND_SUBCLASS_OF  StackANNComponent ANNComponent

//BIND_CONSTRUCTOR StackANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new StackANNComponent(name);
  LUABIND_RETURN(StackANNComponent, obj);
}
//BIND_END

//BIND_METHOD StackANNComponent push
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, ANNComponent);
  ANNComponent *component;
  LUABIND_GET_PARAMETER(1, ANNComponent, component);
  obj->pushComponent(component);
}
//BIND_END

//BIND_METHOD StackANNComponent top
{
  LUABIND_RETURN(ANNComponent, obj->topComponent());
}
//BIND_END

//BIND_METHOD StackANNComponent pop
{
  obj->popComponent();
}
//BIND_END

//BIND_METHOD StackANNComponent clone
{
  LUABIND_RETURN(StackANNComponent,
		 dynamic_cast<StackANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//               JoinANNComponent                  //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME JoinANNComponent ann.components.join
//BIND_CPP_CLASS    JoinANNComponent
//BIND_SUBCLASS_OF  JoinANNComponent ANNComponent

//BIND_CONSTRUCTOR JoinANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new JoinANNComponent(name);
  LUABIND_RETURN(JoinANNComponent, obj);
}
//BIND_END

//BIND_METHOD JoinANNComponent add
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, ANNComponent);
  ANNComponent *component;
  LUABIND_GET_PARAMETER(1, ANNComponent, component);
  obj->addComponent(component);
}
//BIND_END

//BIND_METHOD JoinANNComponent clone
{
  LUABIND_RETURN(JoinANNComponent,
		 dynamic_cast<JoinANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//               CopyANNComponent                  //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME CopyANNComponent ann.components.copy
//BIND_CPP_CLASS    CopyANNComponent
//BIND_SUBCLASS_OF  CopyANNComponent ANNComponent

//BIND_CONSTRUCTOR CopyANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  int argn = lua_gettop(L);
  const char *name=0;
  unsigned int input_size=0, output_size=0, times;
  check_table_fields(L, 1, "times", "name", "input", "output", 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, times, uint, times, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, input, uint, input_size, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, output, uint, output_size, 0);
  obj = new CopyANNComponent(times, name, input_size, output_size);
  LUABIND_RETURN(CopyANNComponent, obj);
}
//BIND_END

//BIND_METHOD CopyANNComponent clone
{
  LUABIND_RETURN(CopyANNComponent,
		 dynamic_cast<CopyANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//         ActivationFunctionANNComponent          //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME ActivationFunctionANNComponent ann.components.actf
//BIND_CPP_CLASS    ActivationFunctionANNComponent
//BIND_SUBCLASS_OF  ActivationFunctionANNComponent ANNComponent

//BIND_CONSTRUCTOR ActivationFunctionANNComponent
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD ActivationFunctionANNComponent clone
{
  LUABIND_RETURN(ActivationFunctionANNComponent,
		 dynamic_cast<ActivationFunctionANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//            LogisticActfANNComponent             //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LogisticActfANNComponent ann.components.logistic
//BIND_CPP_CLASS    LogisticActfANNComponent
//BIND_SUBCLASS_OF  LogisticActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR LogisticActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new LogisticActfANNComponent(name);
  LUABIND_RETURN(LogisticActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//              TanhActfANNComponent               //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME TanhActfANNComponent ann.components.tanh
//BIND_CPP_CLASS    TanhActfANNComponent
//BIND_SUBCLASS_OF  TanhActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR TanhActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new TanhActfANNComponent(name);
  LUABIND_RETURN(TanhActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//            SoftsignActfANNComponent             //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME SoftsignActfANNComponent ann.components.softsign
//BIND_CPP_CLASS    SoftsignActfANNComponent
//BIND_SUBCLASS_OF  SoftsignActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR SoftsignActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new SoftsignActfANNComponent(name);
  LUABIND_RETURN(SoftsignActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//           LogLogisticActfANNComponent           //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LogLogisticActfANNComponent ann.components.log_logistic
//BIND_CPP_CLASS    LogLogisticActfANNComponent
//BIND_SUBCLASS_OF  LogLogisticActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR LogLogisticActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new LogLogisticActfANNComponent(name);
  LUABIND_RETURN(LogLogisticActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//            SoftmaxActfANNComponent              //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME SoftmaxActfANNComponent ann.components.softmax
//BIND_CPP_CLASS    SoftmaxActfANNComponent
//BIND_SUBCLASS_OF  SoftmaxActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR SoftmaxActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new SoftmaxActfANNComponent(name);
  LUABIND_RETURN(SoftmaxActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//           LogSoftmaxActfANNComponent            //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LogSoftmaxActfANNComponent ann.components.log_softmax
//BIND_CPP_CLASS    LogSoftmaxActfANNComponent
//BIND_SUBCLASS_OF  LogSoftmaxActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR LogSoftmaxActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new LogSoftmaxActfANNComponent(name);
  LUABIND_RETURN(LogSoftmaxActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//            SoftplusActfANNComponent             //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME SoftplusActfANNComponent ann.components.softplus
//BIND_CPP_CLASS    SoftplusActfANNComponent
//BIND_SUBCLASS_OF  SoftplusActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR SoftplusActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new SoftplusActfANNComponent(name);
  LUABIND_RETURN(SoftplusActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//            HardtanhActfANNComponent             //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME HardtanhActfANNComponent ann.components.hardtanh
//BIND_CPP_CLASS    HardtanhActfANNComponent
//BIND_SUBCLASS_OF  HardtanhActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR HardtanhActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new HardtanhActfANNComponent(name);
  LUABIND_RETURN(HardtanhActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//               SinActfANNComponent               //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME SinActfANNComponent ann.components.sin
//BIND_CPP_CLASS    SinActfANNComponent
//BIND_SUBCLASS_OF  SinActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR SinActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new SinActfANNComponent(name);
  LUABIND_RETURN(SinActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//              LinearActfANNComponent             //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LinearActfANNComponent ann.components.linear
//BIND_CPP_CLASS    LinearActfANNComponent
//BIND_SUBCLASS_OF  LinearActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR LinearActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new LinearActfANNComponent(name);
  LUABIND_RETURN(LinearActfANNComponent, obj);  
}
//BIND_END
