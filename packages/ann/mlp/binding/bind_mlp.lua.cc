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
#include "errorfunc.h"

//BIND_END

//BIND_HEADER_H
#include "mlp.h"
#include "all_all_mlp.h"
#include "all_all_connection.h"
//#include "local_connection.h"
//#include "tabular_connection.h"
#include "bind_ann_base.h"
#include "bind_mtrand.h"
#include "bind_matrix.h"

using namespace ANN;
//BIND_END

//BIND_LUACLASSNAME ANNBase ann.__base__

//BIND_LUACLASSNAME MLP ann.mlp
//BIND_CPP_CLASS    MLP
//BIND_SUBCLASS_OF  MLP ANNBase

//BIND_LUACLASSNAME AllAllMLP ann.mlp.all_all
//BIND_CPP_CLASS    AllAllMLP
//BIND_SUBCLASS_OF  AllAllMLP MLP

//////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MLP
//DOC_BEGIN
// mlp({ [bunch_size = number] })
/// MLP constructor. Builds a LUA/C++ object that represents a general
/// MLP. You could add layers, connections and actions in a free
/// manner.
///@param bunch_size Is the mini-batch (or bunch) size. Is used to
///enhance the efficiency of the system. A tipical value is 32.
//DOC_END
{
  unsigned int bunch_size;
  LUABIND_CHECK_ARGN(==,1);
  check_table_fields(L, 1, "bunch_size", 0);
  LUABIND_GET_TABLE_PARAMETER(1, bunch_size, uint, bunch_size);
  obj = new MLP(ANNConfiguration(bunch_size,bunch_size));
  LUABIND_RETURN(MLP, obj);
}
//BIND_END

//BIND_METHOD MLP clone
//DOC_BEGIN
// mlp clone()
/// Makes an exact deep copy of the object.
//DOC_END
{
  LUABIND_RETURN(MLP, obj->clone());
}
//BIND_END

//BIND_METHOD MLP show_weights
//DOC_BEGIN
// void show_weights()
/// Show at stdout weights values: for debugging
//DOC_END
{
  obj->showWeights();
}
//BIND_END

//BIND_METHOD MLP randomize_weights
//DOC_BEGIN
// void randomize_weights({ random = random(...), [inf = number], [sup = number] })
/// Initializes the weights using a random numbers generator and an
/// optional inferior and superior interval values.
///@param random A random number generator, instance of MTRand (random in LUA)
///@param inf Inferior bound of the interval. By default is -0.7.
///@param sup Superior bound of the interval. By default is  0.7.
//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  check_table_fields(L, 1, "random", "inf", "sup", 0);

  MTRand	*random;
  float		 inf, sup;
  
  LUABIND_GET_TABLE_PARAMETER(1, random, MTRand, random);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, inf, float, inf, -0.7);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, sup, float, sup,  0.7);
  
  obj->randomizeWeights(random, inf, sup);
}
//BIND_END

//BIND_METHOD MLP set_error_function
//DOC_BEGIN
// void set_error_function(error_functions.__base__ error_func)
//DOC_END
{
  ErrorFunction *error_func;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, ErrorFunction, error_func);
  obj->setErrorFunction(error_func);
}
//BIND_END


///////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR AllAllMLP
{
  LUABIND_ERROR("Use generate method");
}
//BIND_END

//BIND_CLASS_METHOD AllAllMLP generate
{
  LUABIND_CHECK_ARGN(==,1);
  check_table_fields(L, 1, "topology", "bunch_size",
		     "random", "inf", "sup",
		     "w", "oldw", 0);

  const char	*topology;
  unsigned int	 bunch_size;
  MTRand	*random;
  float		 inf, sup;
  MatrixFloat	*w, *oldw;
  
  LUABIND_GET_TABLE_PARAMETER(1, topology, string, topology);
  LUABIND_GET_TABLE_PARAMETER(1, bunch_size, uint, bunch_size);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, random, MTRand, random, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, inf, float, inf, -0.7);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, sup, float, sup,  0.7);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, w, MatrixFloat, w, 0);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, oldw, MatrixFloat, oldw, 0);

  if ( (w != 0 || oldw != 0) && random) {
    LUABIND_ERROR("w/oldw is forbidden with random parameter!!!\n");
  }
  
  if (w == 0 && random == 0) {
    LUABIND_ERROR("w or random parameter is needed!!!\n");
  }
  
  AllAllMLP *obj = new AllAllMLP(ANNConfiguration(bunch_size,bunch_size));

  if (w)           obj->generateAllAll(topology, w, oldw);
  else if (random) obj->generateAllAll(topology, random, inf, sup);
  
  LUABIND_RETURN(AllAllMLP, obj);
}
//BIND_END

//BIND_METHOD AllAllMLP weights
{
  MatrixFloat *weights     = 0;
  MatrixFloat *old_weights = 0;
  
  obj->copyWeightsToMatrix(&weights, &old_weights);
  
  LUABIND_RETURN(MatrixFloat, weights);
  LUABIND_RETURN(MatrixFloat, old_weights);
}
//BIND_END

//BIND_METHOD AllAllMLP description
{
  LUABIND_RETURN(string, obj->getDescription());
}
//BIND_END

//BIND_METHOD AllAllMLP clone
{
  LUABIND_RETURN(AllAllMLP, obj->clone());
}
//BIND_END

//BIND_METHOD AllAllMLP get_layer_connections
{
  unsigned int layer;
  LUABIND_GET_PARAMETER(1, uint, layer);
  if (layer < 1)
    LUABIND_ERROR("First layer is numbered with 1");
  LUABIND_RETURN(Connections, obj->getLayerConnections(layer-1));
}
//BIND_END
