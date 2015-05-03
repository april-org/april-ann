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
#include <typeinfo>

#include "bind_function_interface.h"
#include "bind_matrix.h"
#include "bind_sparse_matrix.h"
#include "bind_mtrand.h"
#include "bind_tokens.h"
#include "bind_util.h"
#include "table_of_token_codes.h"

using namespace AprilUtils;
using namespace Basics;

namespace ANN {
  static bool rewrapToAtLeastDim2(AprilUtils::SharedPtr<Token> &tk) {
    if (tk->getTokenCode() == table_of_token_codes::token_matrix) {
      Basics::TokenMatrixFloat *tk_mat = tk->convertTo<Basics::TokenMatrixFloat*>();
      Basics::MatrixFloat *m = tk_mat->getMatrix();
      if (m->getNumDim() == 1) {
        int dims[2] = { 1, m->getDimSize(0) };
        tk.reset( new Basics::TokenMatrixFloat(m->rewrap(dims, 2)) );
        return true;
      }
    }
    return false;
  }

  static void unwrapToDim1(AprilUtils::SharedPtr<Token> &tk) {
    if (tk->getTokenCode() == table_of_token_codes::token_matrix) {
      Basics::TokenMatrixFloat *tk_mat = tk->convertTo<Basics::TokenMatrixFloat*>();
      Basics::MatrixFloat *m = tk_mat->getMatrix();
      int dim = m->getDimSize(1);
      Basics::MatrixFloat *new_m = m->rewrap(&dim, 1);
      tk.reset( new Basics::TokenMatrixFloat(new_m) );
    }
  }

}

void lua_pushAuxANNComponent(lua_State *L, ANNComponent *value) {
  if (typeid(*value) == typeid(StackANNComponent)) {
    lua_pushStackANNComponent(L, (StackANNComponent*)value);
  }
  else if (typeid(*value) == typeid(JoinANNComponent)) {
    lua_pushJoinANNComponent(L, (JoinANNComponent*)value);
  }
  else if (dynamic_cast<ActivationFunctionANNComponent*>(value)) {
    lua_pushActivationFunctionANNComponent(L, (ActivationFunctionANNComponent*)value);
  }
  else if (dynamic_cast<StochasticANNComponent*>(value)) {
    lua_pushStochasticANNComponent(L, (StochasticANNComponent*)value);
  }
  else if (dynamic_cast<ConvolutionANNComponent*>(value)) {
    lua_pushConvolutionANNComponent(L, (ConvolutionANNComponent*)value);
  }
  else if (dynamic_cast<RewrapANNComponent*>(value)) {
    lua_pushRewrapANNComponent(L, (RewrapANNComponent*)value);
  }
  else if (dynamic_cast<FlattenANNComponent*>(value)) {
    lua_pushFlattenANNComponent(L, (FlattenANNComponent*)value);
  }
  else if (dynamic_cast<ProbabilisticMatrixANNComponent*>(value)) {
    lua_pushProbabilisticMatrixANNComponent(L, (ProbabilisticMatrixANNComponent*)value);
  }
  else {
    lua_pushANNComponent(L, value);
  }
}

namespace AprilUtils {

  template<> ANN::ANNComponent *LuaTable::
  convertTo<ANN::ANNComponent *>(lua_State *L, int idx) {
    return lua_toANNComponent(L, idx);
  }
  
  template<> void LuaTable::
  pushInto<ANN::ANNComponent *>(lua_State *L, ANN::ANNComponent *value) {
    lua_pushAuxANNComponent(L, value);
  }

  template<> bool LuaTable::
  checkType<ANN::ANNComponent *>(lua_State *L, int idx) {
    return lua_isANNComponent(L, idx);
  }
  
}

//BIND_END

//BIND_HEADER_H
#include "activation_function_component.h"
#include "ann_component.h"
#include "bias_component.h"
#include "bind_function_interface.h"
#include "connection.h"
#include "const_component.h"
#include "convolution_bias_component.h"
#include "convolution_component.h"
#include "copy_component.h"
#include "dot_product_component.h"
#include "dropout_component.h"
#include "error_print.h"
#include "exp_actf_component.h"
#include "flatten_component.h"
#include "gaussian_noise_component.h"
#include "hardtanh_actf_component.h"
#include "hyperplane_component.h"
#include "join_component.h"
#include "leaky_relu_actf_component.h"
#include "linear_actf_component.h"
#include "log_actf_component.h"
#include "log_logistic_actf_component.h"
#include "log_softmax_actf_component.h"
#include "logistic_actf_component.h"
#include "maxpooling_component.h"
#include "prelu_actf_component.h"
#include "probabilistic_matrix_component.h"
#include "pca_whitening_component.h"
#include "relu_actf_component.h"
#include "rewrap_component.h"
#include "salt_and_pepper_component.h"
#include "select_component.h"
#include "sin_actf_component.h"
#include "slice_component.h"
#include "softmax_actf_component.h"
#include "softplus_actf_component.h"
#include "softsign_actf_component.h"
#include "stack_component.h"
#include "tanh_actf_component.h"
#include "transpose_component.h"
#include "zca_whitening_component.h"

using namespace Functions;
using namespace ANN;

void lua_pushAuxANNComponent(lua_State *L, ANNComponent *value);

//BIND_END

/////////////////////////////////////////////////////
//                  Connections                    //
/////////////////////////////////////////////////////

//BIND_FUNCTION ann.connections
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "input", "output",
		     "w", "oldw", "first_pos", "column_size",
		     (const char *)0);
  Basics::MatrixFloat *w, *oldw;
  unsigned int input_size, output_size, first_pos, column_size;
  LUABIND_GET_TABLE_PARAMETER(1, input, uint, input_size);
  LUABIND_GET_TABLE_PARAMETER(1, output, uint, output_size);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, w, MatrixFloat, w, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, oldw, MatrixFloat, oldw, 0);
  if (oldw != 0) ERROR_PRINT("oldw field is deprecated\n");
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, first_pos, uint, first_pos, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, column_size, uint, column_size,
				       input_size);
  //
  Basics::MatrixFloat *obj;
  if (w) {
    obj = w->clone();
  }
  else {
    obj = Connections::build(input_size, output_size);
  }
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_FUNCTION ann.connections.to_lua_string
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1,MatrixFloat);
  Basics::MatrixFloat *obj;
  LUABIND_GET_PARAMETER(1, MatrixFloat, obj);
  char *str = Connections::toLuaString(obj);
  LUABIND_RETURN(string, str);
  delete[] str;
}
//BIND_END

//BIND_FUNCTION ann.connections.load
{
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_CHECK_PARAMETER(1,MatrixFloat);
  LUABIND_CHECK_PARAMETER(2,table);
  check_table_fields(L, 2, "w", "oldw", "first_pos", "column_size",
		     (const char *)0);
  
  unsigned int	 first_pos, column_size;
  Basics::MatrixFloat	*w, *oldw, *obj;
  
  LUABIND_GET_PARAMETER(1, MatrixFloat, obj);
  
  LUABIND_GET_TABLE_PARAMETER(2, w, MatrixFloat, w);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(2, oldw, MatrixFloat, oldw, 0);
  if (oldw != 0) ERROR_PRINT("oldw field is deprecated\n");
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(2, first_pos, uint, first_pos, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(2, column_size, uint, column_size,
				       Connections::getNumInputs(obj));
  LUABIND_RETURN(uint, Connections::loadWeights(obj, w, first_pos, column_size));
}
//BIND_END

//BIND_FUNCTION ann.connections.copy_to
{
  LUABIND_CHECK_ARGN(>=,1);
  LUABIND_CHECK_ARGN(<=,2);
  LUABIND_CHECK_PARAMETER(1, MatrixFloat);
  
  int argn = lua_gettop(L);
  Basics::MatrixFloat *w=0, *oldw=0, *obj;
  LUABIND_GET_PARAMETER(1, MatrixFloat, obj);
  unsigned int first_pos=0, column_size=Connections::getNumInputs(obj);
  
  if (argn == 2) {
    LUABIND_CHECK_PARAMETER(2,table);
    check_table_fields(L, 2, "w", "oldw", "first_pos", "column_size",
		       (const char *)0);

    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, w, MatrixFloat, w, w);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, oldw, MatrixFloat, oldw, 0);
    if (oldw != 0) ERROR_PRINT("oldw field is deprecated\n");
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, first_pos, uint, first_pos,
					 first_pos);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, column_size, uint, column_size,
					 column_size);
  }

  int size = static_cast<int>(obj->size());
  if (!w)    w    = new Basics::MatrixFloat(1, first_pos + size);
  
  if (first_pos + obj->size() > static_cast<unsigned int>(w->size()))
    LUABIND_ERROR("Incorrect matrix size!!\n");

  unsigned int lastpos;
  lastpos = Connections::copyWeightsTo(obj, w, first_pos, column_size);
  LUABIND_RETURN(MatrixFloat, w);
  LUABIND_RETURN(uint, lastpos);
}
//BIND_END

//BIND_FUNCTION ann.connections.get_input_size
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, MatrixFloat);
  Basics::MatrixFloat *obj;
  LUABIND_GET_PARAMETER(1, MatrixFloat, obj);
  LUABIND_RETURN(uint, Connections::getInputSize(obj));
}
//BIND_END

//BIND_FUNCTION ann.connections.get_output_size
{
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, MatrixFloat);
  Basics::MatrixFloat *obj;
  LUABIND_GET_PARAMETER(1, MatrixFloat, obj);
  LUABIND_RETURN(uint, Connections::getOutputSize(obj));
}
//BIND_END

//BIND_FUNCTION ann.connections.randomize_weights
{
  LUABIND_CHECK_ARGN(==, 2);
  LUABIND_CHECK_PARAMETER(1, MatrixFloat);
  LUABIND_CHECK_PARAMETER(2, table);
  check_table_fields(L, 2, "random", "inf", "sup", (const char *)0);
  Basics::MTRand *rnd;
  float inf, sup;
  bool use_fanin;
  Basics::MatrixFloat *obj;
  LUABIND_GET_PARAMETER(1, MatrixFloat, obj);
  LUABIND_GET_TABLE_PARAMETER(2, random, MTRand, rnd);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(2, inf, float, inf, -1.0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(2, sup, float,  sup, 1.0);
  Connections::randomizeWeights(obj, rnd, inf, sup);
  LUABIND_RETURN(MatrixFloat, obj);
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

//BIND_METHOD ANNComponent has_weights_name
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

//BIND_METHOD ANNComponent copy_state
{
  LuaTable state;
  if (lua_istable(L,1)) state = lua_toLuaTable(L,1);
  obj->copyState(state);
  LUABIND_RETURN(LuaTable, state);
}
//BIND_END

//BIND_METHOD ANNComponent set_state
{
  LuaTable state(L,1);
  obj->setState(state);
  LUABIND_RETURN(AuxANNComponent, obj);
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
  AprilUtils::SharedPtr<Basics::Token> aux( obj->getInput() );
  LUABIND_RETURN(AuxToken, aux);
}
//BIND_END

//BIND_METHOD ANNComponent get_output
{
  AprilUtils::SharedPtr<Basics::Token> aux( obj->getOutput() );
  LUABIND_RETURN(AuxToken, aux);
}
//BIND_END

//BIND_METHOD ANNComponent get_error_input
{
  AprilUtils::SharedPtr<Basics::Token> aux( obj->getErrorInput() );
  LUABIND_RETURN(AuxToken, aux);
}
//BIND_END

//BIND_METHOD ANNComponent get_error_output
{
  AprilUtils::SharedPtr<Basics::Token> aux( obj->getErrorOutput() );
  LUABIND_RETURN(AuxToken, aux);
}
//BIND_END

//BIND_METHOD ANNComponent precompute_output_size
{
  vector<unsigned int> input_size, output_size;
  int argn = lua_gettop(L);
  if (argn == 0) input_size.push_back(0);
  else {  
    int n;
    LUABIND_TABLE_GETN(1, n);
    input_size.resize(n);
    LUABIND_TABLE_TO_VECTOR(1, uint, input_size, n);
  }
  obj->precomputeOutputSize(input_size, output_size);
  LUABIND_FORWARD_CONTAINER_TO_NEW_TABLE(vector<unsigned int>,
					 uint, output_size);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD ANNComponent forward
{
  AprilUtils::SharedPtr<Basics::Token> input;
  bool during_training;
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 2);
  LUABIND_GET_PARAMETER(1, AuxToken, input);
  LUABIND_GET_OPTIONAL_PARAMETER(2, bool, during_training, false);
  bool rewrapped = rewrapToAtLeastDim2(input);
  AprilUtils::SharedPtr<Basics::Token> output( obj->doForward(input.get(),
                                                              during_training) );
  if (rewrapped) unwrapToDim1(output);
  LUABIND_RETURN(AuxToken, output);
}
//BIND_END

//BIND_METHOD ANNComponent backprop
{
  AprilUtils::SharedPtr<Basics::Token> input;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, AuxToken, input);
  bool rewrapped = rewrapToAtLeastDim2(input);
  AprilUtils::SharedPtr<Basics::Token> gradient( obj->doBackprop(input.get()) );
  if (!gradient.empty()) {
    if (rewrapped) unwrapToDim1(gradient);
    LUABIND_RETURN(AuxToken, gradient);
  }
  else LUABIND_RETURN_NIL();
}
//BIND_END

//BIND_METHOD ANNComponent reset
{
  unsigned int it;
  LUABIND_GET_OPTIONAL_PARAMETER(1, int, it, 0);
  obj->reset(it);
}
//BIND_END

//BIND_METHOD ANNComponent compute_gradients
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  AprilUtils::LuaTable weight_grads_dict;
  if (argn == 1) {
    weight_grads_dict = AprilUtils::LuaTable(L,1);
  }
  //
  obj->computeAllGradients(weight_grads_dict);
  LUABIND_RETURN(LuaTable, weight_grads_dict);
}
//BIND_END

//BIND_METHOD ANNComponent clone
{
  LUABIND_RETURN(AuxANNComponent, obj->clone());
}
//BIND_END

//BIND_METHOD ANNComponent set_use_cuda
{
  bool use_cuda;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, bool, use_cuda);
  obj->setUseCuda(use_cuda);
  LUABIND_RETURN(AuxANNComponent, obj);
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
  AprilUtils::LuaTable weights_dict(L), components_dict(L);
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "input", "output", "weights", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, input, uint, input_size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, output, uint, output_size, 0);
    lua_getfield(L, 1, "weights");
    if (!lua_isnil(L, -1)) weights_dict = lua_toLuaTable(L,-1);
    lua_pop(L, 1);
  }
  //
  obj->build(input_size, output_size, weights_dict, components_dict);
  //
  LUABIND_RETURN(AuxANNComponent, obj);
  LUABIND_RETURN(LuaTable, weights_dict);
  LUABIND_RETURN(LuaTable, components_dict);
}
//BIND_END

//BIND_METHOD ANNComponent copy_weights
{
  AprilUtils::LuaTable weights_dict(L);
  int argn = lua_gettop(L);
  if (argn == 1) {
    weights_dict = lua_toLuaTable(L, 1);
  }
  obj->copyWeights(weights_dict);
  LUABIND_RETURN(LuaTable, weights_dict);
}
//BIND_END

//BIND_METHOD ANNComponent copy_components
{
  AprilUtils::LuaTable components_dict(L);
  int argn = lua_gettop(L);
  if (argn == 1) {
    components_dict = lua_toLuaTable(L, 1);
  }
  obj->copyComponents(components_dict);
  LUABIND_RETURN(LuaTable, components_dict);
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
  LUABIND_RETURN(AuxANNComponent, component);
}
//BIND_END

//BIND_FUNCTION ann.generate_name
{
  const char *prefix;
  LUABIND_GET_OPTIONAL_PARAMETER(1, string, prefix, 0);
  AprilUtils::string name = ANNComponent::generateName(prefix);
  LUABIND_RETURN(string, name.c_str());
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
//         ProbabilisticMatrixANNComponent         //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME ProbabilisticMatrixANNComponent ann.components.probabilistic_matrix
//BIND_CPP_CLASS    ProbabilisticMatrixANNComponent
//BIND_SUBCLASS_OF  ProbabilisticMatrixANNComponent ANNComponent

//BIND_CONSTRUCTOR ProbabilisticMatrixANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0, *weights_name=0, *side_str = 0;
  unsigned int input_size=0, output_size=0;
  ProbabilisticMatrixANNComponent::NormalizationSide side;
  side = ProbabilisticMatrixANNComponent::LEFT;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "side", "name", "weights", 
		       "input", "output", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, side, string, side_str, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, weights, string, weights_name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, input, uint, input_size, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, output, uint, output_size, 0);
    if (side_str != 0) {
      if (strcmp(side_str, "left")==0) {
        side = ProbabilisticMatrixANNComponent::LEFT;
      }
      else if (strcmp(side_str, "right")==0) {
        side = ProbabilisticMatrixANNComponent::RIGHT;
      }
      else {
        side = ProbabilisticMatrixANNComponent::LEFT; // avoid compiler warning
        LUABIND_ERROR("Incorrect side string, expected 'left' or 'right'");
      }
    } // if (side_str != 0)
  }
  obj = new ProbabilisticMatrixANNComponent(side, name, weights_name,
                                            input_size, output_size);
  LUABIND_RETURN(ProbabilisticMatrixANNComponent, obj);
}
//BIND_END

//BIND_METHOD ProbabilisticMatrixANNComponent get_normalized_weights
{
  LUABIND_RETURN(MatrixFloat, obj->getNormalizedWeights());
}
//BIND_END

//BIND_METHOD ProbabilisticMatrixANNComponent clone
{
  LUABIND_RETURN(ProbabilisticMatrixANNComponent,
		 dynamic_cast<ProbabilisticMatrixANNComponent*>(obj->clone()));
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
  LUABIND_CHECK_ARGN(>=, 1);
  int argn = lua_gettop(L);
  for (int i=1; i<=argn; ++i) {
    LUABIND_CHECK_PARAMETER(i, ANNComponent);
    ANNComponent *component;
    LUABIND_GET_PARAMETER(i, ANNComponent, component);
    obj->pushComponent(component);
  }
  LUABIND_RETURN(StackANNComponent, obj);
}
//BIND_END

//BIND_METHOD StackANNComponent unroll
{
  lua_checkstack(L, obj->size());
  for (unsigned int i=0; i<obj->size(); ++i) {
    LUABIND_RETURN(AuxANNComponent, obj->getComponentAt(i));
  }
}
//BIND_END

//BIND_METHOD StackANNComponent get
{
  LUABIND_CHECK_ARGN(>=,1);
  int argn = lua_gettop(L);
  lua_checkstack(L, argn);
  for (int i=1; i<=argn; ++i) {
    unsigned int idx = lua_tointeger(L, i);
    if (idx > obj->size())
      LUABIND_FERROR2("Incorrect index, expected <= %d, found %d\n",
		      obj->size(), idx);
    LUABIND_RETURN(AuxANNComponent, obj->getComponentAt(idx - 1));
  }
}
//BIND_END

//BIND_METHOD StackANNComponent top
{
  if (obj->size() > 0) LUABIND_RETURN(AuxANNComponent, obj->topComponent());
}
//BIND_END

//BIND_METHOD StackANNComponent pop
{
  if (obj->size() > 0) obj->popComponent();
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
//              TransposeANNComponent              //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME TransposeANNComponent ann.components.transpose
//BIND_CPP_CLASS    TransposeANNComponent
//BIND_SUBCLASS_OF  TransposeANNComponent ANNComponent

//BIND_CONSTRUCTOR TransposeANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0;
  check_table_fields(L, 1, "name", "dims", (const char *)0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  lua_getfield(L, 1, "dims");
  AprilUtils::UniquePtr<int[]> which;
  if (!lua_isnil(L, -1)) {
    if (!lua_istable(L, -1)) {
      LUABIND_ERROR("Expected a table at field dims");
    }
    int n;
    LUABIND_TABLE_GETN(-1, n);
    if (n != 2) {
      LUABIND_ERROR("Needs a dims table field with two elements");
    }
    which = new int[2];
    LUABIND_TABLE_TO_VECTOR_SUB1(-1, int, which, n);
  }
  lua_pop(L, 1);
  obj = new TransposeANNComponent(which.get(), name);
  LUABIND_RETURN(TransposeANNComponent, obj);
}
//BIND_END

//BIND_METHOD TransposeANNComponent clone
{
  LUABIND_RETURN(TransposeANNComponent,
		 dynamic_cast<TransposeANNComponent*>(obj->clone()));
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
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, input_planes_dim, int,
				       input_planes_dim, -1);
  if (input_planes_dim > 1) {
    LUABIND_ERROR("Deprecated property, new version only allowed for input_planes_dim==1\n");
  }
  else if (input_planes_dim == 1) {
    ERROR_PRINT("Deprecated property, not needed in the new version\n");
  }
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, weights, string, weights, 0);
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
  obj = new ConvolutionANNComponent(size, kernel, step, n,
				    name, weights);
  LUABIND_RETURN(ConvolutionANNComponent, obj);
  delete[] kernel;
  delete[] step;
}
//BIND_END

//BIND_METHOD ConvolutionANNComponent get_kernel_shape
{
  const int *kernel;
  int n;
  kernel = obj->getKernelShape(n);
  LUABIND_VECTOR_TO_NEW_TABLE(int, kernel, n);
  LUABIND_INCREASE_NUM_RETURNS(1);
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

/////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME StochasticANNComponent ann.components.stochastic
//BIND_CPP_CLASS    StochasticANNComponent
//BIND_SUBCLASS_OF  StochasticANNComponent ANNComponent

//BIND_CONSTRUCTOR StochasticANNComponent
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD StochasticANNComponent set_random
{
  Basics::MTRand *random;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_CHECK_PARAMETER(1, MTRand);
  LUABIND_GET_PARAMETER(1, MTRand, random);
  obj->setRandom(random);
  LUABIND_RETURN(StochasticANNComponent, obj);
}
//BIND_END

//BIND_METHOD StochasticANNComponent get_random
{
  LUABIND_RETURN(MTRand, obj->getRandom());
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
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  float mean=0.0f, var=0.1f;
  unsigned int size=0;
  Basics::MTRand *random=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "size", "random", "mean", "var", "name",
		       (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, random, MTRand, random, random);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, mean, float, mean, mean);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, var, float, var, var);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint, size, size);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, name);
  }
  if (!random) random = new Basics::MTRand();
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
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  float zero=0.0f, one=1.0f, prob=0.2f;
  unsigned int size=0;
  Basics::MTRand *random=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "size", "random", "one", "zero", "prob", "name",
		       (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, random, MTRand, random, random);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, prob, float, prob, prob);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, zero, float,  zero, zero);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, one,  float,  one,  one);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint,   size, size);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, name);
  }
  if (!random) random = new Basics::MTRand();  
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
  Basics::MTRand *random=0;
  bool norm = true; // normalize_after_training
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", "size", "prob", "value", "random", "norm",
		       (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, name);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint, size, size);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, value, float, value, value);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, prob, float, prob, prob);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, norm, bool, norm, norm);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, random, MTRand, random, random);
  }
  if (!random) random = new Basics::MTRand();
  obj = new DropoutANNComponent(random, value, prob, norm, name, size);
  LUABIND_RETURN(DropoutANNComponent, obj);  
}
//BIND_END

//BIND_METHOD SaltAndPepperANNComponent clone
{
  LUABIND_RETURN(DropoutANNComponent,
		 dynamic_cast<DropoutANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

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

//BIND_LUACLASSNAME SparseLogisticActfANNComponent ann.components.actf.sparse_logistic
//BIND_CPP_CLASS    SparseLogisticActfANNComponent
//BIND_SUBCLASS_OF  SparseLogisticActfANNComponent LogisticActfANNComponent
////BIND_CONSTRUCTOR SparseLogisticActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  float sparsity = 1.0f;
  float penalty = 0.05f;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", "sparsity", "penalty", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, penalty, float, penalty, 1.0f); 
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, sparsity, float, sparsity, 0.05f); 
  }
  obj = new SparseLogisticActfANNComponent(name, penalty, sparsity);
  LUABIND_RETURN(SparseLogisticActfANNComponent, obj);  
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
//       LeakyReLUActfANNComponent                 //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LeakyReLUActfANNComponent ann.components.actf.leaky_relu
//BIND_CPP_CLASS    LeakyReLUActfANNComponent
//BIND_SUBCLASS_OF  LeakyReLUActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR LeakyReLUActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  float leak=0.01f;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", "leak", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, leak, float, leak, 0.01f);
  }
  obj = new LeakyReLUActfANNComponent(leak, name);
  LUABIND_RETURN(LeakyReLUActfANNComponent, obj);  
}
//BIND_END

///////////////////////////////////////////////////////
//             PReLUActfANNComponent                 //
///////////////////////////////////////////////////////

//BIND_LUACLASSNAME PReLUActfANNComponent ann.components.actf.prelu
//BIND_CPP_CLASS    PReLUActfANNComponent
//BIND_SUBCLASS_OF  PReLUActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR PReLUActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0, *weights=0;
  unsigned int size=0;
  bool shared=false;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", "size", "shared", "weights", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, weights, string, weights, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, shared, bool, shared, false);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, size, uint, size, 0);
  }
  obj = new PReLUActfANNComponent(shared, size, name, weights);
  LUABIND_RETURN(PReLUActfANNComponent, obj);  
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
//               LogActfANNComponent               //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME LogActfANNComponent ann.components.actf.log
//BIND_CPP_CLASS    LogActfANNComponent
//BIND_SUBCLASS_OF  LogActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR LogActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new LogActfANNComponent(name);
  LUABIND_RETURN(LogActfANNComponent, obj);  
}
//BIND_END

/////////////////////////////////////////////////////
//               ExpActfANNComponent               //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME ExpActfANNComponent ann.components.actf.exp
//BIND_CPP_CLASS    ExpActfANNComponent
//BIND_SUBCLASS_OF  ExpActfANNComponent ActivationFunctionANNComponent

//BIND_CONSTRUCTOR ExpActfANNComponent
{
  LUABIND_CHECK_ARGN(<=, 1);
  int argn = lua_gettop(L);
  const char *name=0;
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1, "name", (const char *)0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  }
  obj = new ExpActfANNComponent(name);
  LUABIND_RETURN(ExpActfANNComponent, obj);  
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

/////////////////////////////////////////////////////
//           PCAWhiteningANNComponent              //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME PCAWhiteningANNComponent ann.components.pca_whitening
//BIND_CPP_CLASS    PCAWhiteningANNComponent
//BIND_SUBCLASS_OF  PCAWhiteningANNComponent ANNComponent

//BIND_CONSTRUCTOR PCAWhiteningANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0;
  float epsilon;
  int takeN;
  Basics::MatrixFloat *U;
  Basics::SparseMatrixFloat *S;
  check_table_fields(L, 1, "name", "U", "S", "takeN", "epsilon", (const char *)0);
  LUABIND_GET_TABLE_PARAMETER(1, U, MatrixFloat, U);
  LUABIND_GET_TABLE_PARAMETER(1, S, SparseMatrixFloat, S);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, takeN, int, takeN, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, epsilon, float, epsilon, 0);
  //
  obj = new PCAWhiteningANNComponent(U, S, epsilon, takeN, name);
  LUABIND_RETURN(PCAWhiteningANNComponent, obj);
}
//BIND_END

//BIND_METHOD PCAWhiteningANNComponent clone
{
  LUABIND_RETURN(PCAWhiteningANNComponent,
		 dynamic_cast<PCAWhiteningANNComponent*>(obj->clone()));
}
//BIND_END

/////////////////////////////////////////////////////
//           ZCAWhiteningANNComponent              //
/////////////////////////////////////////////////////

//BIND_LUACLASSNAME ZCAWhiteningANNComponent ann.components.zca_whitening
//BIND_CPP_CLASS    ZCAWhiteningANNComponent
//BIND_SUBCLASS_OF  ZCAWhiteningANNComponent ANNComponent

//BIND_CONSTRUCTOR ZCAWhiteningANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0;
  float epsilon;
  int takeN;
  Basics::MatrixFloat *U;
  Basics::SparseMatrixFloat *S;
  check_table_fields(L, 1, "name", "U", "S", "takeN", "epsilon", (const char *)0);
  LUABIND_GET_TABLE_PARAMETER(1, U, MatrixFloat, U);
  LUABIND_GET_TABLE_PARAMETER(1, S, SparseMatrixFloat, S);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, takeN, int, takeN, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, epsilon, float, epsilon, 0);
  //
  obj = new ZCAWhiteningANNComponent(U, S, epsilon, takeN, name);
  LUABIND_RETURN(ZCAWhiteningANNComponent, obj);
}
//BIND_END

//BIND_METHOD ZCAWhiteningANNComponent clone
{
  LUABIND_RETURN(ZCAWhiteningANNComponent,
		 dynamic_cast<ZCAWhiteningANNComponent*>(obj->clone()));
}
//BIND_END

//////////////////////////////////////////////
//           ConstANNComponent              //
//////////////////////////////////////////////

//BIND_LUACLASSNAME ConstANNComponent ann.components.const
//BIND_CPP_CLASS    ConstANNComponent
//BIND_SUBCLASS_OF  ConstANNComponent ANNComponent

//BIND_CONSTRUCTOR ConstANNComponent
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  const char *name=0;
  ANNComponent *component;
  check_table_fields(L, 1, "name", "component", (const char *)0);
  LUABIND_GET_TABLE_PARAMETER(1, component, ANNComponent, component);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, name, string, name, 0);
  //
  obj = new ConstANNComponent(component, name);
  LUABIND_RETURN(ConstANNComponent, obj);
}
//BIND_END

//BIND_METHOD ConstANNComponent clone
{
  LUABIND_RETURN(ConstANNComponent,
		 dynamic_cast<ConstANNComponent*>(obj->clone()));
}
//BIND_END
