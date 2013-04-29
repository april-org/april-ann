/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#include "bind_matrix.h"
#include "bind_mtrand.h"
#include "bind_tokens.h"

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
#include "activation_function_component.h"
#include "connection.h"
#include "activation_function_component.h"
#include "logistic_actf_component.h"
#include "tanh_actf_component.h"
#include "softsign_actf_component.h"
#include "softplus_actf_component.h"
#include "log_logistic_actf_component.h"
#include "softmax_actf_component.h"
#include "log_softmax_actf_component.h"

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
		     "w", "oldw", "first_pos", "column_size", 0);
  MatrixFloat *w, *oldw;
  unsigned int input_size, output_size, first_pos, column_size;
  LUABIND_GET_TABLE_PARAMETER(1, input, uint, input_size);
  LUABIND_GET_TABLE_PARAMETER(1, output, uint, output_size);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, w, MatrixFloat, w, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, oldw, MatrixFloat, oldw, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, first_pos, uint, first_pos, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, column_size, uint, column_size,
				       input_size);
  if (oldw && !w) LUABIND_ERROR("Parameter w is mandatory with oldw!!!\n");
  obj=new Connections(input_size, output_size);
  if (w) obj->loadWeights(w, oldw, first_pos, column_size);
  LUABIND_RETURN(Connections, obj);
}
//BIND_END

//BIND_METHOD Connections clone
{
  Connections *cnn = obj->clone();
  LUABIND_RETURN(Connections, cnn);
}
//BIND_END

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
  const char *name = 0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new ANNComponent(name);
  LUABIND_RETURN(ANNComponent, obj);
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

//BIND_METHOD ANNComponent copyWeights
{
  hash<string,Connections*> weights_dict;
  obj->copyWeights(weights_dict);
  pushHashTableInLuaStack(L, weights_dict, lua_pushConnections);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD ANNComponent copyComponents
{
  hash<string,ANNComponent*> components_dict;
  obj->copyComponents(components_dict);
  pushHashTableInLuaStack(L, components_dict, lua_pushANNComponent);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD ANNComponent getComponent
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
