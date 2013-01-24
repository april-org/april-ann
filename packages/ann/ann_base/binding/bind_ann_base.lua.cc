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
#include "bind_activation_function.h"
#include "bind_matrix.h"
#include "bind_mtrand.h"
//BIND_END

//BIND_HEADER_H
#include "ann.h"
#include "connection.h"
#include "all_all_connection.h"
#include "bias_connection.h"
#include "action.h"
#include "dot_product_action.h"
#include "forward_bias_action.h"
#include "activations_action.h"
#include "actunit.h"
#include "errorfunc.h"
#include "bind_trainablesuper.h"
#include "bind_function_interface.h"
#include "trainsuper.h"

using namespace Trainable;
using namespace ANN;
using namespace Functions;

//BIND_END

//BIND_LUACLASSNAME TrainableSupervised __trainable__
//BIND_LUACLASSNAME ANNBase ann.__base__
//BIND_CPP_CLASS    ANNBase
//BIND_SUBCLASS_OF  ANNBase TrainableSupervised


//////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME ActivationUnits      ann.units.__base__
//BIND_LUACLASSNAME RealActivationUnits  ann.units.real_cod
//BIND_LUACLASSNAME LocalActivationUnits ann.units.local_cod

//BIND_CPP_CLASS    ActivationUnits
//BIND_CPP_CLASS    RealActivationUnits
//BIND_CPP_CLASS    LocalActivationUnits

//BIND_SUBCLASS_OF  RealActivationUnits  ActivationUnits
//BIND_SUBCLASS_OF  LocalActivationUnits ActivationUnits

//BIND_CONSTRUCTOR ActivationUnits
{
  LUABIND_ERROR("Abstract class!!!\n");
}
//BIND_END

//BIND_METHOD ActivationUnits num_neurons
{
  LUABIND_RETURN(uint, obj->numNeurons());
}
//BIND_END

//BIND_METHOD ActivationUnits clone
{
  LUABIND_CHECK_ARGN(>=,1);
  LUABIND_CHECK_ARGN(<=,2);
  ANNBase *ann;
  const char *type;
  LUABIND_GET_PARAMETER(1, ANNBase, ann);
  LUABIND_GET_OPTIONAL_PARAMETER(2, string, type, 0);
  ActivationUnitsType type_enum = HIDDEN_TYPE;
  if (type == 0) type_enum = obj->getType();
  else if (strcmp(type, "inputs") == 0)
    type_enum = INPUTS_TYPE;
  else if (strcmp(type, "outputs") == 0)
    type_enum = OUTPUTS_TYPE;
  ActivationUnits *units = obj->clone(ann->getConfReference());
  units->setType(type_enum);
  ann->registerActivationUnits(units);
  if (type_enum == INPUTS_TYPE) ann->registerInput(units);
  else if (type_enum == OUTPUTS_TYPE) ann->registerOutput(units);
  LUABIND_RETURN(ActivationUnits, units);
}
//BIND_END

//////////////////////////////////////////////////////////////////


//BIND_CONSTRUCTOR RealActivationUnits
//DOC_BEGIN
// ann.units.real{ size = ..., ann = ..., type = (inputs, hidden, outputs) }
/// Create a real activation units layer
/// @param size The number of neurons
/// @param ann  The ANN which belongs
/// @param type The type, could be inputs, hidden, outputs
//DOC_END
{
  
  LUABIND_CHECK_ARGN(==,1);
  check_table_fields(L, 1, "size", "ann",
		     "type", 0);
  
  unsigned int		 size;
  ANNBase		*ann;
  const char		*type;
  ActivationUnitsType    type_enum = HIDDEN_TYPE;
  
  LUABIND_GET_TABLE_PARAMETER(1, size, uint, size);
  LUABIND_GET_TABLE_PARAMETER(1, ann, ANNBase, ann);
  LUABIND_GET_TABLE_PARAMETER(1, type, string, type);
  
  if (strcmp(type, "inputs")!=0 &&
      strcmp(type, "hidden")!=0 &&
      strcmp(type, "outputs")!=0) {
    LUABIND_FERROR1("Incorrect type '%s'!!\n", type);
  }
  
  if (strcmp(type, "inputs") == 0)
    type_enum = INPUTS_TYPE;
  else if (strcmp(type, "outputs") == 0)
    type_enum = OUTPUTS_TYPE;
  
  obj = new RealActivationUnits(size, ann->getConfReference(),
				type_enum,
				type_enum != INPUTS_TYPE);
  ann->registerActivationUnits(obj);
  if (type_enum == INPUTS_TYPE)
    ann->registerInput(obj);
  else if (type_enum == OUTPUTS_TYPE)
    ann->registerOutput(obj);
  LUABIND_RETURN(RealActivationUnits, obj);
}
//BIND_END

//BIND_CONSTRUCTOR LocalActivationUnits
//DOC_BEGIN
// ann.units.real{ size = ..., ann = ... }
/// Create a local activation input units layer
/// @param size The number of neurons
/// @param ann  The ANN which belongs
//DOC_END
{
  
  LUABIND_CHECK_ARGN(==,1);
  check_table_fields(L, 1, "size", "num_groups", "ann", 0);
  
  unsigned int		 size, num_groups;
  ANNBase		*ann;
  ActivationUnitsType    type_enum = INPUTS_TYPE;

  LUABIND_GET_TABLE_PARAMETER(1, size, uint, size);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, num_groups, uint, num_groups, 1);
  LUABIND_GET_TABLE_PARAMETER(1, ann, ANNBase, ann);
  
  obj = new LocalActivationUnits(num_groups, size, ann->getConfReference(),
				 type_enum);
  ann->registerActivationUnits(obj);
  ann->registerInput(obj);
  LUABIND_RETURN(LocalActivationUnits, obj);
}
//BIND_END

////////////////////////////////////////////////

//BIND_CONSTRUCTOR ANNBase
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD ANNBase get_num_weights
{
  LUABIND_RETURN(uint, obj->getNumWeights());
}
//BIND_END

//BIND_METHOD ANNBase release_output
{
  obj->releaseOutput();
}
//BIND_END

//BIND_METHOD ANNBase register_action
{
  Action *action;
  LUABIND_GET_PARAMETER(1, Action, action);
  obj->registerAction(action);
  LUABIND_RETURN(Action, action);
}
//BIND_END

//BIND_METHOD ANNBase register_input
{
  ActivationUnits *actu;
  LUABIND_GET_PARAMETER(1, ActivationUnits, actu);
  obj->registerInput(actu);
}
//BIND_END

//BIND_METHOD ANNBase register_output
{
  ActivationUnits *actu;
  LUABIND_GET_PARAMETER(1, ActivationUnits, actu);
  obj->registerOutput(actu);
}
//BIND_END

//BIND_METHOD ANNBase set_use_cuda
{
  bool use_cuda, pinned;
  LUABIND_GET_PARAMETER(1, bool, use_cuda);
  LUABIND_GET_OPTIONAL_PARAMETER(2, bool, pinned, true);
  obj->setUseCuda(use_cuda, pinned);
}
//BIND_END

//BIND_METHOD ANNBase get_layer_connections
{
  unsigned int layer;
  LUABIND_GET_PARAMETER(1, uint, layer);
  if (layer < 1)
    LUABIND_ERROR("First layer is numbered with 1");
  LUABIND_RETURN(Connections, obj->getLayerConnections(layer-1));
}
//BIND_END

//BIND_METHOD ANNBase get_layer_activations
{
  unsigned int layer;
  LUABIND_GET_PARAMETER(1, uint, layer);
  if (layer < 1)
    LUABIND_ERROR("First layer is numbered with 1");
  LUABIND_RETURN(ActivationUnits, obj->getLayerActivations(layer-1));
}
//BIND_END

//BIND_METHOD ANNBase get_layer_connections_size
{
  LUABIND_RETURN(uint, obj->getLayerConnectionsSize());
}
//BIND_END

//BIND_METHOD ANNBase get_layer_activations_size
{
  LUABIND_RETURN(uint, obj->getLayerActivationsSize());
}
//BIND_END


//BIND_METHOD ANNBase get_layer_connections_vector
{
  unsigned int sz = obj->getLayerConnectionsSize();
  lua_createtable(L, static_cast<int>(sz), 0);
  for (unsigned int i=0; i<sz; ++i) {
    Connections *cnn = obj->getLayerConnections(i);
    lua_pushConnections(L, cnn);
    lua_rawseti(L, -2, i+1);
  }
  LUABIND_RETURN_FROM_STACK(1);
}
//BIND_END

//BIND_METHOD ANNBase get_layer_activations_vector
{
  unsigned int sz = obj->getLayerActivationsSize();
  lua_createtable(L, static_cast<int>(sz), 0);
  for (unsigned int i=0; i<sz; ++i) {
    ActivationUnits *units = obj->getLayerActivations(i);
    lua_pushActivationUnits(L, units);
    lua_rawseti(L, -2, i+1);
  }
  LUABIND_RETURN_FROM_STACK(1);
}
//BIND_END


/////////////////////////////////////////////////////

//BIND_LUACLASSNAME Action ann.actions.__base__
//BIND_CPP_CLASS    Action

//BIND_CONSTRUCTOR Action
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD Action set_option
{
  const char *name;
  double value;
  LUABIND_GET_PARAMETER(1, string, name);
  LUABIND_GET_PARAMETER(2, double, value);
  obj->setOption(name, value);
}
//BIND_END

//BIND_METHOD Action get_option
{
  const char *name;
  LUABIND_GET_PARAMETER(1, string, name);
  LUABIND_RETURN(double, obj->getOption(name));
}
//BIND_END

//BIND_METHOD Action has_option
{
  const char *name;
  LUABIND_GET_PARAMETER(1, string, name);
  LUABIND_RETURN(bool, obj->hasOption(name));
}
//BIND_END

//////////////////////////////////////////////////////

//BIND_LUACLASSNAME ForwardBiasAction ann.actions.forward_bias
//BIND_CPP_CLASS    ForwardBiasAction
//BIND_SUBCLASS_OF  ForwardBiasAction Action

//BIND_CONSTRUCTOR ForwardBiasAction
{
  LUABIND_CHECK_ARGN(==,1);
  check_table_fields(L, 1, "ann", "output", "connections", 0);
  
  ActivationUnits *output;
  Connections     *conn;
  ANNBase	  *ann;
  
  LUABIND_GET_TABLE_PARAMETER(1, output, ActivationUnits, output);
  LUABIND_GET_TABLE_PARAMETER(1, connections, Connections, conn);
  LUABIND_GET_TABLE_PARAMETER(1, ann, ANNBase, ann);
  
  obj = new ForwardBiasAction(ann->getConfReference(),
			      output, conn);
  ann->registerAction(obj);
  LUABIND_RETURN(ForwardBiasAction, obj);
}
//BIND_END

//////////////////////////////////////////////////////

//BIND_LUACLASSNAME DotProductAction ann.actions.dot_product
//BIND_CPP_CLASS    DotProductAction
//BIND_SUBCLASS_OF  DotProductAction Action

//BIND_CONSTRUCTOR DotProductAction
{
  LUABIND_CHECK_ARGN(==,1);
  check_table_fields(L, 1, "ann", "input", "output", "connections",
		     "transpose", 0);
  
  ActivationUnits *input;
  ActivationUnits *output;
  Connections     *conn;
  ANNBase	  *ann;
  bool             transpose;
  
  LUABIND_GET_TABLE_PARAMETER(1, input, ActivationUnits, input);
  LUABIND_GET_TABLE_PARAMETER(1, output, ActivationUnits, output);
  LUABIND_GET_TABLE_PARAMETER(1, connections, Connections, conn);
  LUABIND_GET_TABLE_PARAMETER(1, ann, ANNBase, ann);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, transpose, bool, transpose, false);
  
  obj = new DotProductAction(ann->getConfReference(),
			     input, output, conn,
			     transpose);
  ann->registerAction(obj);
  LUABIND_RETURN(DotProductAction, obj);
}
//BIND_END

//////////////////////////////////////////////////////

//BIND_LUACLASSNAME ActivationsAction ann.actions.activations
//BIND_CPP_CLASS    ActivationsAction
//BIND_SUBCLASS_OF  ActivationsAction Action

//BIND_CONSTRUCTOR ActivationsAction
{
  LUABIND_CHECK_ARGN(==,1);
  check_table_fields(L, 1, "ann", "actfunc", "output", 0);
  
  ActivationUnits    *output;
  ActivationFunction *actfunc;
  ANNBase	     *ann;
  
  LUABIND_GET_TABLE_PARAMETER(1, output, ActivationUnits, output);
  LUABIND_GET_TABLE_PARAMETER(1, actfunc, ActivationFunction, actfunc);
  LUABIND_GET_TABLE_PARAMETER(1, ann, ANNBase, ann);
  
  obj = new ActivationsAction(ann->getConfReference(),
			      output, actfunc);
  ann->registerAction(obj);
  LUABIND_RETURN(ActivationsAction, obj);
}
//BIND_END

//////////////////////////////////////////////////////

//BIND_LUACLASSNAME ErrorFunction ann.error_functions.__base__
//BIND_CPP_CLASS    ErrorFunction

//BIND_CONSTRUCTOR ErrorFunction
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_LUACLASSNAME MSE ann.error_functions.mse
//BIND_CPP_CLASS    MSE
//BIND_SUBCLASS_OF  MSE ErrorFunction

//BIND_CONSTRUCTOR MSE
{
  obj = new MSE();
  LUABIND_RETURN(MSE, obj);
}
//BIND_END

//BIND_LUACLASSNAME Tanh ann.error_functions.tanh
//BIND_CPP_CLASS    Tanh
//BIND_SUBCLASS_OF  Tanh ErrorFunction

//BIND_CONSTRUCTOR Tanh
{
  obj = new Tanh();
  LUABIND_RETURN(Tanh, obj);
}
//BIND_END
 
//BIND_LUACLASSNAME CrossEntropy ann.error_functions.cross_entropy
//BIND_CPP_CLASS    CrossEntropy
//BIND_SUBCLASS_OF  CrossEntropy ErrorFunction

//BIND_CONSTRUCTOR CrossEntropy
{
  obj = new CrossEntropy();
  LUABIND_RETURN(CrossEntropy, obj);
}
//BIND_END

//BIND_LUACLASSNAME FullCrossEntropy ann.error_functions.full_cross_entropy
//BIND_CPP_CLASS    FullCrossEntropy
//BIND_SUBCLASS_OF  FullCrossEntropy ErrorFunction

//BIND_CONSTRUCTOR FullCrossEntropy
{
  obj = new FullCrossEntropy();
  LUABIND_RETURN(FullCrossEntropy, obj);
}
//BIND_END

//BIND_LUACLASSNAME LocalFMeasure ann.error_functions.local_fmeasure
//BIND_CPP_CLASS    LocalFMeasure
//BIND_SUBCLASS_OF  LocalFMeasure ErrorFunction

//BIND_CONSTRUCTOR LocalFMeasure
{
  obj = new LocalFMeasure();
  LUABIND_RETURN(LocalFMeasure, obj);
}
//BIND_END

///////////////////////////////////////////////////

//BIND_LUACLASSNAME Connections ann.connections.__base__
//BIND_CPP_CLASS    Connections

//BIND_CONSTRUCTOR Connections
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD Connections clone
{
  LUABIND_CHECK_ARGN(==,1);
  ANNBase *ann;
  LUABIND_GET_PARAMETER(1, ANNBase, ann);
  Connections *cnn = obj->clone();
  ann->registerConnections(cnn);
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

  LUABIND_RETURN(uint, obj->copyWeightsTo(w, oldw, first_pos, column_size));
  LUABIND_RETURN(MatrixFloat, w);
  LUABIND_RETURN(MatrixFloat, oldw);
}
//BIND_END

//BIND_METHOD Connections size
{
  LUABIND_RETURN(uint, obj->size());
}
//BIND_END

//BIND_METHOD Connections randomize_weights
{
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L, 1, "random", "inf", "sup", 0);
  MTRand *rnd;
  float inf, sup;
  LUABIND_GET_TABLE_PARAMETER(1, random, MTRand, rnd);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, inf, float, inf, -0.7);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, sup, float,  sup, 0.7);
  obj->randomizeWeights(rnd, inf, sup);
}
//BIND_END

////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME AllAllConnections ann.connections.all_all
//BIND_CPP_CLASS    AllAllConnections
//BIND_SUBCLASS_OF  AllAllConnections Connections

//BIND_CONSTRUCTOR AllAllConnections
{
  check_table_fields(L, 1, "input_size", "output_size", "ann", 0);
  unsigned int input_size, output_size;
  ANNBase *ann;
  LUABIND_GET_TABLE_PARAMETER(1, input_size, uint, input_size);
  LUABIND_GET_TABLE_PARAMETER(1, output_size, uint, output_size);
  LUABIND_GET_TABLE_PARAMETER(1, ann, ANNBase, ann);
  obj=new AllAllConnections(input_size, output_size);
  ann->registerConnections(obj);
  LUABIND_RETURN(AllAllConnections, obj);
}
//BIND_END

//BIND_LUACLASSNAME BiasConnections ann.connections.bias
//BIND_CPP_CLASS    BiasConnections
//BIND_SUBCLASS_OF  BiasConnections Connections

//BIND_CONSTRUCTOR BiasConnections
{
  check_table_fields(L, 1, "size", "ann", 0);
  unsigned int size;
  ANNBase *ann;
  LUABIND_GET_TABLE_PARAMETER(1, size, uint, size);
  LUABIND_GET_TABLE_PARAMETER(1, ann, ANNBase, ann);
  obj=new BiasConnections(size);
  ann->registerConnections(obj);
  LUABIND_RETURN(BiasConnections, obj);
}
//BIND_END

/////////////////////////////////////////////////////

//BIND_LUACLASSNAME TimeSeriesErrorFunction ann.time_series_error_functions.__base__
//BIND_CPP_CLASS    TimeSeriesErrorFunction

//BIND_CONSTRUCTOR TimeSeriesErrorFunction
{
  LUABIND_ERROR("Abstract class!!!");
}
//BIND_END

//BIND_METHOD TimeSeriesErrorFunction computeErrorFromTimeSerie
{
  float *outputs;
  float *target_outputs;
  unsigned int output_size;
  check_table_fields(L, 1, "outputs", "target_outputs", 0);
  lua_getfield(L, 1, "outputs");
  LUABIND_TABLE_GETN(-1, output_size);
  outputs = new float[output_size];
  LUABIND_TABLE_TO_VECTOR(-1, float, outputs, output_size);
  lua_pop(L, 1);
  lua_getfield(L, 1, "target_outputs");
  unsigned int aux;
  LUABIND_TABLE_GETN(-1, aux);
  if (aux != output_size)
    LUABIND_ERROR ("outputs and target_outputs sizes are diferent!!!\n");
  target_outputs = new float[output_size];
  LUABIND_TABLE_TO_VECTOR(-1, float, target_outputs, output_size);
  lua_pop(L, 1);
  LUABIND_RETURN(float, obj->computeErrorFromTimeSerie(outputs, target_outputs,
						       output_size));
  delete[] outputs;
  delete[] target_outputs;
}
//BIND_END

////////////////////////////////////////////////////

//BIND_LUACLASSNAME NormalizedRootMSE ann.time_series_error_functions.nrmse
//BIND_CPP_CLASS    NormalizedRootMSE
//BIND_SUBCLASS_OF  NormalizedRootMSE TimeSeriesErrorFunction

//BIND_CONSTRUCTOR NormalizedRootMSE
{
  obj = new NormalizedRootMSE();
  LUABIND_RETURN(NormalizedRootMSE, obj);
}
//BIND_END


////////////////////////////////////////////////////

//BIND_LUACLASSNAME RootMSE ann.time_series_error_functions.rmse
//BIND_CPP_CLASS    RootMSE
//BIND_SUBCLASS_OF  RootMSE TimeSeriesErrorFunction

//BIND_CONSTRUCTOR RootMSE
{
  obj = new RootMSE();
  LUABIND_RETURN(RootMSE, obj);
}
//BIND_END


////////////////////////////////////////////////////

//BIND_LUACLASSNAME AbsoluteError ann.time_series_error_functions.absolute_error
//BIND_CPP_CLASS    AbsoluteError
//BIND_SUBCLASS_OF  AbsoluteError TimeSeriesErrorFunction

//BIND_CONSTRUCTOR AbsoluteError
{
  obj = new AbsoluteError();
  LUABIND_RETURN(AbsoluteError, obj);
}
//BIND_END
