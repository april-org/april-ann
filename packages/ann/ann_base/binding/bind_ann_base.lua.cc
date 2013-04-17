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

template<Value,PushFunction>
void pushHashTableInLuaStack(hash<string,Value> &h,
			     PushFunction push_function) {
  lua_createtable(L, 0, h.size());
  for (hash<string,Value>::iterator it = h.begin(); it != h.end(); ++it) {
    push_function(L, it->second);
    lua_setfield(L, -2, it->first.c_str());
  }
}

//BIND_END

//BIND_HEADER_H
#include "ann_component.h"
#include "dot_product_component.h"
#include "bias_component.h"
#include "activation_function_component.h"
#include "connection.h"

using namespace ANN;

//BIND_END

/////////////////////////////////////////////////////
//                  ANNComponent                   //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME ANNComponent ann.__component__
//BIND_CPP_CLASS    ANNComponent

//BIND_CONSTRUCTOR ANNComponent
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD ANNComponent set_option
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
{
  const char *name;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, string, name);
  LUABIND_RETURN(double, obj->getOption(name));
}
//BIND_END

//BIND_METHOD ANNComponent has_option
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
  LUABIND_RETURN(Token, obj->getInput());
}
//BIND_END

//BIND_METHOD ANNComponent get_output
{
  LUABIND_RETURN(Token, obj->getOutput());
}
//BIND_END

//BIND_METHOD ANNComponent get_error_input
{
  LUABIND_RETURN(Token, obj->getErrorInput());
}
//BIND_END

//BIND_METHOD ANNComponent get_output
{
  LUABIND_RETURN(Token, obj->getErrorOutput());
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
  LUABIND_RETURN(Token, obj->doBackprop(input, false));
}
//BIND_END

//BIND_METHOD ANNComponent update
{
  obj->update();
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
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  unsigned int input_size, output_size;
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, input_size, uint, input_size, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, output_size, uint, output_size, 0);
  hash<string,Connection*> weights_dict;
  hash<string,ANNComponent*> components_dict;
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
  //
  obj->build(input_size, output_size, weights_dict, components_dict);
  //
  pushHashTableInLuaStack(weights_dict, lua_pushConnections);
  LUABIND_RETURN_FROM_STACK(-1);
  pushHashTableInLuaStack(components_dict, lua_pushANNComponent);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD ANNComponent copyWeights
{
  hash<string,Connection*> weights_dict;
  obj->copyWeights(weights_dict);
  pushHashTableInLuaStack(weights_dict, lua_pushConnections);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD ANNComponent copyComponents
{
  hash<string,ANNComponent*> components_dict;
  obj->copyComponents(components_dict);
  pushHashTableInLuaStack(components_dict, lua_pushANNComponent);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD ANNComponent copyComponents
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

//BIND_METHOD ANNComponent computeFanInAndFanOut
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1,string);
  const char *name;
  LUABIND_GET_PARAMETER(1, string, name);
  string name_string(name);
  unsigned int fan_in, fan_out;
  obj->computeFanInAndFanOut(name_string, fan_in, fan_out);
  LUABIND_RETURN(uint, fan_in);
  LUABIND_RETURN(uint, fan_out);
}
//BIND_END

/////////////////////////////////////////////////////
//                  ANNComponent                   //
/////////////////////////////////////////////////////
