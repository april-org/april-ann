/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
 *
 * The APRIL-ANN toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 */
//BIND_HEADER_C
#include "bind_function_interface.h"
#include "bind_matrix.h"
#include "bind_mtrand.h"
#include "bind_tokens.h"
#include "table_of_token_codes.h"

static bool rewrapToAtLeastDim2(Token *&tk) {
  if (tk->getTokenCode() == table_of_token_codes::token_matrix) {
    TokenMatrixFloat *tk_mat = tk->convertTo<TokenMatrixFloat*>();
    MatrixFloat *m = tk_mat->getMatrix();
    if (m->getNumDim() == 1) {
      int dims[2] = { 1, m->getDimSize(0) };
      tk = new TokenMatrixFloat(m->rewrap(dims, 2));
      return true;
    }
  }
  return false;
}

static void unwrapToDim1(Token *&tk) {
  if (tk->getTokenCode() == table_of_token_codes::token_matrix) {
    TokenMatrixFloat *tk_mat = tk->convertTo<TokenMatrixFloat*>();
    MatrixFloat *m = tk_mat->getMatrix();
    int dim = m->getDimSize(1);
    MatrixFloat *new_m = m->rewrap(&dim, 1);
    tk = new TokenMatrixFloat(new_m);
  }
}

template<typename Value, typename PushFunction>
void pushHashTableInLuaStack(lua_State *L,
			     hash<string,Value> &hashobject,
			     PushFunction push_function) {
  lua_createtable(L, 0, hashobject.size());
  for (typename hash<string,Value>::iterator it = hashobject.begin();
       it != hashobject.end(); ++it) {
    push_function(L, it->second);
    lua_setfield(L, -2, it->first.c_str());
  }
}

//BIND_END

//BIND_HEADER_H
#include "ann_component.h"
#include "dot_product_component.h"
#include "bias_component.h"
#include "hyperplane_component.h"
#include "stack_component.h"
#include "join_component.h"
#include "copy_component.h"
#include "select_component.h"
#include "rewrap_component.h"
#include "slice_component.h"
#include "flatten_component.h"
#include "gaussian_noise_component.h"
#include "salt_and_pepper_component.h"
#include "convolution_component.h"
#include "convolution_bias_component.h"
#include "maxpooling_component.h"
#include "activation_function_component.h"
#include "connection.h"
#include "activation_function_component.h"
#include "logistic_actf_component.h"
#include "tanh_actf_component.h"
#include "softsign_actf_component.h"
#include "log_logistic_actf_component.h"
#include "softmax_actf_component.h"
#include "log_softmax_actf_component.h"
#include "softplus_actf_component.h"
#include "relu_actf_component.h"
#include "hardtanh_actf_component.h"
#include "sin_actf_component.h"
#include "linear_actf_component.h"
#include "dropout_component.h"
#include "bind_function_interface.h"

using namespace Functions;
using namespace ANN;

//BIND_END

/////////////////////////////////////////////////////
//                  Connections                    //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME Connections ann.connections
//BIND_CPP_CLASS    Connections

//BIND_CONSTRUCTOR Connections
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "input", "output",
		     "w", "oldw", "first_pos", "column_size",
		     (const char *)0);
  MatrixFloat *w, *oldw;
  unsigned int input_size, output_size, first_pos, column_size;
  LUABIND_GET_TABLE_PARAMETER(1, input, uint, input_size);
  LUABIND_GET_TABLE_PARAMETER(1, output, uint, output_size);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, w, MatrixFloat, w, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, oldw, MatrixFloat, oldw, w);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, first_pos, uint, first_pos, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, column_size, uint, column_size,
				       input_size);
  if (oldw && !w) LUABIND_ERROR("Parameter w is mandatory with oldw!!!\n");
  if (w && w->getMajorOrder() == CblasColMajor)
    obj=new Connections(input_size, output_size, w, oldw);
  else {
    obj=new Connections(input_size, output_size);
    if (w) obj->loadWeights(w, oldw, first_pos, column_size);
  }
  LUABIND_RETURN(Connections, obj);
}
//BIND_END

//BIND_METHOD Connections clone
{
  Connections *cnn = obj->clone();
  LUABIND_RETURN(Connections, cnn);
}
//BIND_END

//BIND_METHOD Connections swap
{
  obj->swap();
}
//BIND_END

//BIND_METHOD Connections get_shared_count
{
  LUABIND_RETURN(uint, obj->getSharedCount());
}
//BIND_END

//BIND_METHOD Connections set_shared_count
{
  unsigned int count;
  LUABIND_GET_PARAMETER(1,uint,count);
  obj->resetSharedCount();
  obj->addToSharedCount(count);
}
//BIND_END

//BIND_METHOD Connections to_lua_string
{
  char *str = obj->toLuaString();
  LUABIND_RETURN(string, str);
  delete[] str;
}
//BIND_END

//BIND_METHOD Connections load
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1,table);
  check_table_fields(L, 1, "w", "oldw", "first_pos", "column_size",
		     (const char *)0);

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

//BIND_METHOD Connections prune_subnormal_and_check_normal
{
  obj->pruneSubnormalAndCheckNormal();
}
//BIND_END

//BIND_METHOD Connections copy_to
{
  LUABIND_CHECK_ARGN(<=,1);
  int argn = lua_gettop(L);
  unsigned int  first_pos=0, column_size=obj->getNumInputs();
  MatrixFloat  *w=0, *oldw=0;
  
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1,table);
    check_table_fields(L, 1, "w", "oldw", "first_pos", "column_size",
		       (const char *)0);

    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, w, MatrixFloat, w, w);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, oldw, MatrixFloat, oldw, w);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, first_pos, uint, first_pos,
					 first_pos);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, column_size, uint, column_size,
					 column_size);
  }

  int size = static_cast<int>(obj->size());
  if (!w)    w    = new MatrixFloat(1, first_pos + size);
  if (!oldw) oldw = new MatrixFloat(1, w->size());
  
  if (first_pos + obj->size() > static_cast<unsigned int>(w->size()) ||
      first_pos + obj->size() > static_cast<unsigned int>(oldw->size()) )
    LUABIND_ERROR("Incorrect matrix size!!\n");

  unsigned int lastpos = obj->copyWeightsTo(w, oldw, first_pos, column_size);
  LUABIND_RETURN(MatrixFloat, w);
  LUABIND_RETURN(MatrixFloat, oldw);
  LUABIND_RETURN(uint, lastpos);
}
//BIND_END

//BIND_METHOD Connections matrix
{
  LUABIND_RETURN(MatrixFloat, obj->getPtr());
  LUABIND_RETURN(MatrixFloat, obj->getPrevPtr());
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
  check_table_fields(L, 1, "random", "inf", "sup", (const char *)0);
  MTRand *rnd;
  float inf, sup;
  bool use_fanin;
  LUABIND_GET_TABLE_PARAMETER(1, random, MTRand, rnd);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, inf, float, inf, -1.0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, sup, float,  sup, 1.0);
  obj->randomizeWeights(rnd, inf, sup);
  LUABIND_RETURN(Connections, obj);
}
//BIND_END

//BIND_METHOD Connections print_debug
{
  obj->printDebug();
  LUABIND_RETURN(Connections, obj);
}
//BIND_END

/////////////////////////////////////////////////////
//                  ANNComponent                   //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME FunctionInterface functions

//BIND_LUACLASSNAME ANNComponent ann.components.base
//BIND_CPP_CLASS    ANNComponent
//BIND_SUBCLASS_OF  ANNComponent FunctionInterface

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
    check_table_fields(L, 1, "name", "weights", "size", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, weights, string, weights, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint, size, 0);
  }
  obj = new ANNComponent(name, weights, size, size);
  LUABIND_RETURN(ANNComponent, obj);
}
//BIND_END

//BIND_FUNCTION ann.components.reset_id_counters
{
  ANNComponent::resetIdCounters();
}
//BIND_END

//BIND_METHOD ANNComponent to_lua_string
{
  char *str = obj->toLuaString();
  LUABIND_RETURN(string, str);
  delete[] str;
}
//BIND_END

//BIND_METHOD ANNComponent get_name
{
  LUABIND_RETURN(string, obj->getName().c_str());
}
//BIND_END

//BIND_METHOD ANNComponent get_weights_name
{
  LUABIND_RETURN(string, obj->getWeightsName().c_str());
}
//BIND_END

//BIND_METHOD ANNComponent has_weigths_name
{
  LUABIND_RETURN(bool, obj->hasWeightsName());
}
//BIND_END

//BIND_METHOD ANNComponent get_is_built
{
  LUABIND_RETURN(bool, obj->getIsBuilt());
}
//BIND_END

//BIND_METHOD ANNComponent debug_info
{
  obj->debugInfo();
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
  bool during_training;
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 2);
  LUABIND_GET_PARAMETER(1, Token, input);
  LUABIND_GET_OPTIONAL_PARAMETER(2, bool, during_training, false);
  bool rewrapped = rewrapToAtLeastDim2(input);
  Token *output = obj->doForward(input, during_training);
  if (rewrapped) unwrapToDim1(output);
  LUABIND_RETURN(Token, output);
}
//BIND_END

//BIND_METHOD ANNComponent backprop
{
  Token *input;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, Token, input);
  bool rewrapped = rewrapToAtLeastDim2(input);
  Token *gradient = obj->doBackprop(input);
  if (gradient != 0) {
    if (rewrapped) unwrapToDim1(gradient);
    LUABIND_RETURN(Token, gradient);
  }
  else LUABIND_RETURN_NIL();
}
//BIND_END

//BIND_METHOD ANNComponent reset
{
  obj->reset();
}
//BIND_END

//BIND_METHOD ANNComponent compute_gradients
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  hash<string,MatrixFloat*> weight_grads_dict;
  if (argn == 1) {
    if (lua_istable(L, 1)) {
      lua_pushvalue(L, 1);
      // stack now contains: -1 => table
      lua_pushnil(L);
      // stack now contains: -1 => nil; -2 => table
      while (lua_next(L, -2)) {
	// copy the key so that lua_tostring does not modify the original
	lua_pushvalue(L, -2);
	// stack now contains: -1 => value; -2 => key; -3 => table
	string key(lua_tostring(L, -1));
	// stack now contains: -1 => key; -2 => value; -3 => key; -4 => table
	MatrixFloat *value     = lua_toMatrixFloat(L, -2);
	weight_grads_dict[key] = value;
	// pop value + copy of key, leaving original key
	lua_pop(L, 2);
	// stack now contains: -1 => key; -2 => table
      }
      // stack now contains: -1 => table (when lua_next returns 0 it pops the key
      // but does not push anything.)
    }
    else if (!lua_isnil(L, 1))
      LUABIND_ERROR("Expected a table with a dictionary of weights matrices");
  }
  //
  obj->computeAllGradients(weight_grads_dict);
  pushHashTableInLuaStack(L, weight_grads_dict, lua_pushMatrixFloat);
  LUABIND_RETURN_FROM_STACK(-1);
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
  LUABIND_RETURN(ANNComponent, obj);
}
//BIND_END

//BIND_METHOD ANNComponent get_use_cuda
{
  LUABIND_RETURN(bool, obj->getUseCuda());
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
    check_table_fields(L, 1, "input", "output", "weights", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, input, uint, input_size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, output, uint, output_size, 0);
    lua_getfield(L, 1, "weights");
    if (lua_istable(L, -1)) {
      lua_pushvalue(L, -1);
      // stack now contains: -1 => table
      lua_pushnil(L);
      // stack now contains: -1 => nil; -2 => table
      while (lua_next(L, -2)) {
	// copy the key so that lua_tostring does not modify the original
	lua_pushvalue(L, -2);
	// stack now contains: -1 => value; -2 => key; -3 => table
	string key(lua_tostring(L, -1));
	// stack now contains: -1 => key; -2 => value; -3 => key; -4 => table
	Connections *value = lua_toConnections(L, -2);
	weights_dict[key]  = value;
	// pop value + copy of key, leaving original key
	lua_pop(L, 2);
	// stack now contains: -1 => key; -2 => table
      }
      // stack now contains: -1 => table (when lua_next returns 0 it pops the key
      // but does not push anything.)
    }
    else if (!lua_isnil(L, -1))
      LUABIND_ERROR("Expected a table at field weights");
    // Pop table or nil value
    lua_pop(L, 1);
  }
  //
  obj->build(input_size, output_size, weights_dict, components_dict);
  //
  pushHashTableInLuaStack(L, components_dict, lua_pushANNComponent);
  LUABIND_RETURN_FROM_STACK(-1);
  pushHashTableInLuaStack(L, weights_dict, lua_pushConnections);
  LUABIND_RETURN_FROM_STACK(-2);
  LUABIND_RETURN(ANNComponent, obj);
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
		       "input", "output", "transpose", (const char *)0);
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
  unsigned int size=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", "weights", "size", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint, size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, weights, string, weights_name, 0);
  }
  obj = new BiasANNComponent(size, name, weights_name);
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
		       "input", "output", "transpose", (const char *)0);
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
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new StackANNComponent(name);
  LUABIND_RETURN(StackANNComponent, obj);
}
//BIND_END

//BIND_METHOD StackANNComponent size
{
  LUABIND_RETURN(uint, obj->size());
}
//BIND_END

//BIND_METHOD StackANNComponent push
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, ANNComponent);
  ANNComponent *component;
  LUABIND_GET_PARAMETER(1, ANNComponent, component);
  obj->pushComponent(component);
  LUABIND_RETURN(StackANNComponent, obj);
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
  LUABIND_RETURN(StackANNComponent, obj);
}
//BIND_END

//BIND_METHOD StackANNComponent clone
{
  LUABIND_RETURN(StackANNComponent,
		 dynamic_cast<StackANNComponent*>(obj->clone()));
}
//BIND_END

//BIND_METHOD StackANNComponent set_use_cuda
{
  bool use_cuda;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, bool, use_cuda);
  obj->setUseCuda(use_cuda);
  LUABIND_RETURN(StackANNComponent, obj);
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
    check_table_fields(L, 1, "name", (const char *)0);
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
  LUABIND_RETURN(JoinANNComponent, obj);
}
//BIND_END

//BIND_METHOD JoinANNComponent clone
{
  LUABIND_RETURN(JoinANNComponent,
		 dynamic_cast<JoinANNComponent*>(obj->clone()));
}
//BIND_END

//BIND_METHOD JoinANNComponent set_use_cuda
{
  bool use_cuda;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, bool, use_cuda);
  obj->setUseCuda(use_cuda);
  LUABIND_RETURN(JoinANNComponent, obj);
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
  check_table_fields(L, 1, "times", "name", "input", "output", (const char *)0);
  LUABIND_GET_TABLE_PARAMETER(1, times, uint, times);
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
//              SelectANNComponent                 //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME SelectANNComponent ann.components.select
//BIND_CPP_CLASS    SelectANNComponent
//BIND_SUBCLASS_OF  SelectANNComponent ANNComponent

//BIND_CONSTRUCTOR SelectANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0;
  int dimension, index;
  check_table_fields(L, 1, "name", "dimension", "index", (const char *)0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  LUABIND_GET_TABLE_PARAMETER(1, dimension, int, dimension);
  LUABIND_GET_TABLE_PARAMETER(1, index, int, index);
  obj = new SelectANNComponent(dimension-1, index-1, name);
  LUABIND_RETURN(SelectANNComponent, obj);
}
//BIND_END

//BIND_METHOD SelectANNComponent clone
{
  LUABIND_RETURN(SelectANNComponent,
		 dynamic_cast<SelectANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//              RewrapANNComponent                 //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME RewrapANNComponent ann.components.rewrap
//BIND_CPP_CLASS    RewrapANNComponent
//BIND_SUBCLASS_OF  RewrapANNComponent ANNComponent

//BIND_CONSTRUCTOR RewrapANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0;
  int *size, n;
  check_table_fields(L, 1, "name", "size", (const char *)0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  lua_getfield(L, 1, "size");
  if (!lua_istable(L, -1))
    LUABIND_ERROR("Expected a table at field size");
  LUABIND_TABLE_GETN(-1, n);
  size = new int[n];
  LUABIND_TABLE_TO_VECTOR(-1, int, size, n);
  lua_pop(L, 1);
  obj = new RewrapANNComponent(size, n, name);
  delete[] size;
  LUABIND_RETURN(RewrapANNComponent, obj);
}
//BIND_END

//BIND_METHOD RewrapANNComponent clone
{
  LUABIND_RETURN(RewrapANNComponent,
		 dynamic_cast<RewrapANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//              SliceANNComponent                 //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME SliceANNComponent ann.components.slice
//BIND_CPP_CLASS    SliceANNComponent
//BIND_SUBCLASS_OF  SliceANNComponent ANNComponent

//BIND_CONSTRUCTOR SliceANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0;
  int *size, *pos, n;
  check_table_fields(L, 1, "name", "pos", "size", (const char *)0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  //
  lua_getfield(L, 1, "pos");
  if (!lua_istable(L, -1))
    LUABIND_ERROR("Expected a table at field pos");
  LUABIND_TABLE_GETN(-1, n);
  pos = new int[n];
  LUABIND_TABLE_TO_VECTOR_SUB1(-1, int, pos, n);
  lua_pop(L, 1);
  //
  lua_getfield(L, 1, "size");
  if (!lua_istable(L, -1))
    LUABIND_ERROR("Expected a table at field size");
  LUABIND_TABLE_GETN(-1, n);
  size = new int[n];
  LUABIND_TABLE_TO_VECTOR(-1, int, size, n);
  lua_pop(L, 1);
  //
  obj = new SliceANNComponent(pos, size, n, name);
  delete[] pos;
  delete[] size;
  LUABIND_RETURN(SliceANNComponent, obj);
}
//BIND_END

//BIND_METHOD SliceANNComponent clone
{
  LUABIND_RETURN(SliceANNComponent,
		 dynamic_cast<SliceANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//              FlattenANNComponent                //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME FlattenANNComponent ann.components.flatten
//BIND_CPP_CLASS    FlattenANNComponent
//BIND_SUBCLASS_OF  FlattenANNComponent ANNComponent

//BIND_CONSTRUCTOR FlattenANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new FlattenANNComponent(name);
  LUABIND_RETURN(FlattenANNComponent, obj);
}
//BIND_END

//BIND_METHOD FlattenANNComponent clone
{
  LUABIND_RETURN(FlattenANNComponent,
		 dynamic_cast<FlattenANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//               GaussianNoiseANNComponent         //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME GaussianNoiseANNComponent ann.components.gaussian_noise
//BIND_CPP_CLASS    GaussianNoiseANNComponent
//BIND_SUBCLASS_OF  GaussianNoiseANNComponent ANNComponent

//BIND_CONSTRUCTOR GaussianNoiseANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0;
  float mean, var;
  unsigned int size=0;
  MTRand *random;
  check_table_fields(L, 1, "size", "random", "mean", "var", "name",
		     (const char *)0);
  LUABIND_GET_TABLE_PARAMETER(1, random, MTRand, random);
  LUABIND_GET_TABLE_PARAMETER(1, mean, float, mean);
  LUABIND_GET_TABLE_PARAMETER(1, var, float, var);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint, size, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  obj = new GaussianNoiseANNComponent(random, mean, var, name, size);
  LUABIND_RETURN(GaussianNoiseANNComponent, obj);
}
//BIND_END

//BIND_METHOD GaussianNoiseANNComponent clone
{
  LUABIND_RETURN(GaussianNoiseANNComponent,
		 dynamic_cast<GaussianNoiseANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//               SaltAndPepperANNComponent         //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME SaltAndPepperANNComponent ann.components.salt_and_pepper
//BIND_CPP_CLASS    SaltAndPepperANNComponent
//BIND_SUBCLASS_OF  SaltAndPepperANNComponent ANNComponent

//BIND_CONSTRUCTOR SaltAndPepperANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0;
  float zero, one, prob;
  unsigned int size=0;
  MTRand *random;
  check_table_fields(L, 1, "size", "random", "one", "zero", "prob", "name",
		     (const char *)0);
  LUABIND_GET_TABLE_PARAMETER(1, random, MTRand, random);
  LUABIND_GET_TABLE_PARAMETER(1, prob, float, prob);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, zero, float,  zero, 0.0f);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, one,  float,  one,  1.0f);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint,   size, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  obj = new SaltAndPepperANNComponent(random, zero, one, prob, name, size);
  LUABIND_RETURN(SaltAndPepperANNComponent, obj);
}
//BIND_END

//BIND_METHOD SaltAndPepperANNComponent clone
{
  LUABIND_RETURN(SaltAndPepperANNComponent,
		 dynamic_cast<SaltAndPepperANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//               ConvolutionANNComponent           //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME ConvolutionANNComponent ann.components.convolution
//BIND_CPP_CLASS    ConvolutionANNComponent
//BIND_SUBCLASS_OF  ConvolutionANNComponent ANNComponent

//BIND_CONSTRUCTOR ConvolutionANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0, *weights=0;
  int *kernel, *step, n, input_planes_dim;
  check_table_fields(L, 1, "name", "weights", "kernel", "input_planes_dim",
		     "step", "n", (const char *)0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, weights, string, weights, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, input_planes_dim, int,
				       input_planes_dim, 1);
  LUABIND_GET_TABLE_PARAMETER(1, n, int, n);
  //
  lua_getfield(L, 1, "kernel");
  if (!lua_istable(L, -1))
    LUABIND_ERROR("Expected a table at field 'kernel'");
  int size;
  LUABIND_TABLE_GETN(-1, size);
  kernel = new int[size];
  step = new int[size];
  LUABIND_TABLE_TO_VECTOR(-1, int, kernel, size);
  lua_pop(L, 1);
  //
  lua_getfield(L, 1, "step");
  if (lua_isnil(L, -1)) {
    for (int i=0; i<size; ++i) step[i] = 1;
  }
  else if (!lua_istable(L, -1))
    LUABIND_ERROR("Expected a table at field 'step'");
  else {
    int size2;
    LUABIND_TABLE_GETN(-1, size2);
    if (size != size2)
      LUABIND_ERROR("Tables kernel and step must have the same length");
    LUABIND_TABLE_TO_VECTOR(-1, int, step, size);
  }
  lua_pop(L, 1);
  obj = new ConvolutionANNComponent(size, kernel, step,
				    input_planes_dim, n,
				    name, weights);
  LUABIND_RETURN(ConvolutionANNComponent, obj);
  delete[] kernel;
  delete[] step;
}
//BIND_END

//BIND_METHOD ConvolutionANNComponent clone
{
  LUABIND_RETURN(ConvolutionANNComponent,
		 dynamic_cast<ConvolutionANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//           ConvolutionBiasANNComponent           //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME ConvolutionBiasANNComponent ann.components.convolution_bias
//BIND_CPP_CLASS    ConvolutionBiasANNComponent
//BIND_SUBCLASS_OF  ConvolutionBiasANNComponent ANNComponent

//BIND_CONSTRUCTOR ConvolutionBiasANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0, *weights=0;
  int n, ndims;
  check_table_fields(L, 1, "name", "weights", "n", "ndims", (const char *)0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, weights, string, weights, 0);
  LUABIND_GET_TABLE_PARAMETER(1, n, int, n);
  LUABIND_GET_TABLE_PARAMETER(1, ndims, int, ndims);
  //
  obj = new ConvolutionBiasANNComponent(ndims, n, name, weights);
  LUABIND_RETURN(ConvolutionBiasANNComponent, obj);
}
//BIND_END

//BIND_METHOD ConvolutionBiasANNComponent clone
{
  LUABIND_RETURN(ConvolutionBiasANNComponent,
		 dynamic_cast<ConvolutionBiasANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//                MaxPoolingANNComponent           //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME MaxPoolingANNComponent ann.components.max_pooling
//BIND_CPP_CLASS    MaxPoolingANNComponent
//BIND_SUBCLASS_OF  MaxPoolingANNComponent ANNComponent

//BIND_CONSTRUCTOR MaxPoolingANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0;
  int *kernel, *step;
  check_table_fields(L, 1, "name", "kernel", "step",
		     (const char *)0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  //
  lua_getfield(L, 1, "kernel");
  if (!lua_istable(L, -1))
    LUABIND_ERROR("Expected a table at field 'kernel'");
  int size;
  LUABIND_TABLE_GETN(-1, size);
  kernel = new int[size];
  step = new int[size];
  LUABIND_TABLE_TO_VECTOR(-1, int, kernel, size);
  lua_pop(L, 1);
  //
  lua_getfield(L, 1, "step");
  if (lua_isnil(L, -1)) {
    for (int i=0; i<size; ++i) step[i] = kernel[i];
  }
  else if (!lua_istable(L, -1))
    LUABIND_ERROR("Expected a table at field 'step'");
  else {
    int size2;
    LUABIND_TABLE_GETN(-1, size2);
    if (size != size2)
      LUABIND_ERROR("Tables kernel and step must have the same length");
    LUABIND_TABLE_TO_VECTOR(-1, int, step, size);
  }
  lua_pop(L, 1);
  obj = new MaxPoolingANNComponent(size, kernel, step, name);
  LUABIND_RETURN(MaxPoolingANNComponent, obj);
  delete[] kernel;
  delete[] step;
}
//BIND_END

//BIND_METHOD MaxPoolingANNComponent clone
{
  LUABIND_RETURN(MaxPoolingANNComponent,
		 dynamic_cast<MaxPoolingANNComponent*>(obj->clone()));
}
//BIND_END

////////////////////////////////////////////////////
//              DropoutANNComponent               //
////////////////////////////////////////////////////

//BIND_LUACLASSNAME DropoutANNComponent ann.components.dropout
//BIND_CPP_CLASS    DropoutANNComponent
//BIND_SUBCLASS_OF  DropoutANNComponent ANNComponent

//BIND_CONSTRUCTOR DropoutANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  float prob=0.5f, value=0.0f;
  unsigned int size=0;
  MTRand *random=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", "size", "prob", "value", "random",
		       (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint, size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, prob, float, prob, 0.5f);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, value, float, value, 0.0f);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, random, MTRand, random, 0);
  }
  if (!random) random = new MTRand();
  obj = new DropoutANNComponent(random, value, prob, name, size);
  LUABIND_RETURN(DropoutANNComponent, obj);  
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

//BIND_LUACLASSNAME LogisticActfANNComponent ann.components.actf.logistic
//BIND_CPP_CLASS    LogisticActfANNComponent
//BIND_SUBCLASS_OF  LogisticActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR LogisticActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new LogisticActfANNComponent(name);
  LUABIND_RETURN(LogisticActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//              TanhActfANNComponent               //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME TanhActfANNComponent ann.components.actf.tanh
//BIND_CPP_CLASS    TanhActfANNComponent
//BIND_SUBCLASS_OF  TanhActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR TanhActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new TanhActfANNComponent(name);
  LUABIND_RETURN(TanhActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//            SoftsignActfANNComponent             //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME SoftsignActfANNComponent ann.components.actf.softsign
//BIND_CPP_CLASS    SoftsignActfANNComponent
//BIND_SUBCLASS_OF  SoftsignActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR SoftsignActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new SoftsignActfANNComponent(name);
  LUABIND_RETURN(SoftsignActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//           LogLogisticActfANNComponent           //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LogLogisticActfANNComponent ann.components.actf.log_logistic
//BIND_CPP_CLASS    LogLogisticActfANNComponent
//BIND_SUBCLASS_OF  LogLogisticActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR LogLogisticActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new LogLogisticActfANNComponent(name);
  LUABIND_RETURN(LogLogisticActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//            SoftmaxActfANNComponent              //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME SoftmaxActfANNComponent ann.components.actf.softmax
//BIND_CPP_CLASS    SoftmaxActfANNComponent
//BIND_SUBCLASS_OF  SoftmaxActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR SoftmaxActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new SoftmaxActfANNComponent(name);
  LUABIND_RETURN(SoftmaxActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//           LogSoftmaxActfANNComponent            //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LogSoftmaxActfANNComponent ann.components.actf.log_softmax
//BIND_CPP_CLASS    LogSoftmaxActfANNComponent
//BIND_SUBCLASS_OF  LogSoftmaxActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR LogSoftmaxActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new LogSoftmaxActfANNComponent(name);
  LUABIND_RETURN(LogSoftmaxActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//            SoftplusActfANNComponent             //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME SoftplusActfANNComponent ann.components.actf.softplus
//BIND_CPP_CLASS    SoftplusActfANNComponent
//BIND_SUBCLASS_OF  SoftplusActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR SoftplusActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new SoftplusActfANNComponent(name);
  LUABIND_RETURN(SoftplusActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//            ReLUActfANNComponent                 //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME ReLUActfANNComponent ann.components.actf.relu
//BIND_CPP_CLASS    ReLUActfANNComponent
//BIND_SUBCLASS_OF  ReLUActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR ReLUActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new ReLUActfANNComponent(name);
  LUABIND_RETURN(ReLUActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//            HardtanhActfANNComponent             //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME HardtanhActfANNComponent ann.components.actf.hardtanh
//BIND_CPP_CLASS    HardtanhActfANNComponent
//BIND_SUBCLASS_OF  HardtanhActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR HardtanhActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  float inf=-1.0f, sup=1.0f;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", "inf", "sup", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, inf, float, inf, -1.0f);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, sup, float, sup,  1.0f);
  }
  obj = new HardtanhActfANNComponent(name, inf, sup);
  LUABIND_RETURN(HardtanhActfANNComponent, obj);
}
//BIND_END

/////////////////////////////////////////////////////
//               SinActfANNComponent               //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME SinActfANNComponent ann.components.actf.sin
//BIND_CPP_CLASS    SinActfANNComponent
//BIND_SUBCLASS_OF  SinActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR SinActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new SinActfANNComponent(name);
  LUABIND_RETURN(SinActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//              LinearActfANNComponent             //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LinearActfANNComponent ann.components.actf.linear
//BIND_CPP_CLASS    LinearActfANNComponent
//BIND_SUBCLASS_OF  LinearActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR LinearActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new LinearActfANNComponent(name);
  LUABIND_RETURN(LinearActfANNComponent, obj);  
}
//BIND_END
