/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#include "bind_mathcore.h"
#include "bind_matrix.h"
#include "utilMatrixDouble.h"
#include "luabindutil.h"
#include "luabindmacros.h"

#include "matrix_ext.h"
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Boolean;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilMath::MatrixExt::Misc;
using namespace AprilMath::MatrixExt::LAPACK;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(MatrixDouble);

namespace Basics {
#define FUNCTION_NAME "read_vector"
  static int *read_vector(lua_State *L, const char *key, int num_dim, int add) {
    int *v=0;
    lua_getfield(L, 1, key);
    if (!lua_isnil(L, -1)) {
      LUABIND_CHECK_PARAMETER(-1, table);
      int table_len;
      LUABIND_TABLE_GETN(-1, table_len);
      if (table_len != num_dim)
        LUABIND_FERROR3("Table '%s' with incorrect size, expected %d, found %d",
                        key, num_dim, table_len);
      v = new int[num_dim];
      for(int i=0; i < num_dim; i++) {
        lua_rawgeti(L, -1, i+1);
        v[i] = static_cast<int>(lua_tonumber(L, -1)) + add;
        lua_pop(L,1);
      }
    }
    lua_pop(L, 1);
    return v;
  }
#undef FUNCTION_NAME

  int sliding_window_matrixDouble_iterator_function(lua_State *L) {
    SlidingWindowMatrixDouble *obj = lua_toSlidingWindowMatrixDouble(L,1);
    if (obj->isEnd()) {
      lua_pushnil(L);
      return 1;
    }
    MatrixDouble *mat = obj->getMatrix();
    lua_pushMatrixDouble(L, mat);
    obj->next();
    return 1;
  }
}
//BIND_END

//BIND_HEADER_H
#include "matrixDouble.h"
using namespace Basics;
typedef MatrixDouble::sliding_window SlidingWindowMatrixDouble;
//BIND_END

//BIND_LUACLASSNAME MatrixDouble matrixDouble
//BIND_CPP_CLASS MatrixDouble
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixDouble Serializable

//BIND_LUACLASSNAME SlidingWindowMatrixDouble matrixDouble.__sliding_window__
//BIND_CPP_CLASS SlidingWindowMatrixDouble

//BIND_CONSTRUCTOR SlidingWindowMatrixDouble
{
  LUABIND_ERROR("Use matrixDouble.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble get_matrix
{
  MatrixDouble *dest;
  LUABIND_GET_OPTIONAL_PARAMETER(1, MatrixDouble, dest, 0);
  LUABIND_RETURN(MatrixDouble, obj->getMatrix(dest));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble next
{
  LUABIND_RETURN(SlidingWindowMatrixDouble, obj->next());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble set_at_window
{
  int windex;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, int, windex);
  if (windex < 1) LUABIND_ERROR("Index must be >= 1\n");
  obj->setAtWindow(windex-1);
  LUABIND_RETURN(SlidingWindowMatrixDouble, obj);
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble num_windows
{
  LUABIND_RETURN(int, obj->numWindows());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble coords
{
  LUABIND_VECTOR_TO_NEW_TABLE(int, obj->getCoords(), obj->getNumDim());
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble is_end
{
  LUABIND_RETURN(bool, obj->isEnd());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixDouble iterate
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,sliding_window_matrixDouble_iterator_function);
  LUABIND_RETURN(SlidingWindowMatrixDouble,obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixDouble
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::constructor(L));
}
//BIND_END

//BIND_METHOD MatrixDouble size
{
  LUABIND_RETURN(int, obj->size());
}
//BIND_END

//BIND_METHOD MatrixDouble rewrap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::rewrap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble squeeze
{
  LUABIND_RETURN(MatrixDouble,obj->squeeze());
}
//BIND_END

//BIND_METHOD MatrixDouble get_reference_string
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::
                               get_reference_string(L,obj));
}
//BIND_END


//BIND_METHOD MatrixDouble copy_from_table
//DOC_BEGIN
// void copy_from_table(table matrix_values)
/// Permite dar valores a una matriz. Require una tabla con un numero
/// de argumentos igual al numero de elementos de la matriz.
///@param matrix_values Tabla con los elementos de la matriz.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::
                               copy_from_table(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble get
//DOC_BEGIN
// double get(coordinates)
/// Permite ver valores de una matriz. Requiere tantos indices como dimensiones tenga la matriz.
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble set
//DOC_BEGIN
// double set(coordinates,value)
/// Permite cambiar el valor de un elemento en la matriz. Requiere
/// tantos indices como dimensiones tenga la matriz y adicionalmente
/// el valor a cambiar
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble offset
{
  LUABIND_RETURN(int, obj->getOffset());
}
//BIND_END

//BIND_METHOD MatrixDouble raw_get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::raw_get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble raw_set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::raw_set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble get_use_cuda
{
  LUABIND_RETURN(bool, obj->getCudaFlag());
}
//BIND_END

//BIND_METHOD MatrixDouble set_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::set_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble num_dim
{
  LUABIND_RETURN(int, obj->getNumDim());
}
//BIND_END

//BIND_METHOD MatrixDouble stride
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::stride(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble slice
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::slice(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble select
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::select(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble clone
//DOC_BEGIN
// matrix *clone()
/// Devuelve un <em>clon</em> de la matriz.
//DOC_END
{
  LUABIND_RETURN(MatrixDouble, obj->clone());
}
//BIND_END

// returns a matrix with size as the given matrix, but without data copy
//BIND_CLASS_METHOD MatrixDouble as
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::as(L));
}
//BIND_END

//BIND_METHOD MatrixDouble transpose
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::transpose(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble isfinite
//DOC_BEGIN
// bool isfinite
/// Devuelve false si algun valor es nan o infinito.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  bool resul=true;
  for (MatrixDouble::iterator it(obj->begin()); resul && it!=obj->end(); ++it)
    if ((*it) - (*it) != 0.0f) resul = false;
  LUABIND_RETURN(boolean,resul);
}
//BIND_END

//BIND_METHOD MatrixDouble toTable
// Permite salvar una matriz en una tabla lua
// TODO: Tener en cuenta las dimensiones de la matriz
  {
    LUABIND_FORWARD_CONTAINER_TO_NEW_TABLE(MatrixDouble, double, *obj);
    LUABIND_INCREASE_NUM_RETURNS(1);
  }
//BIND_END

//BIND_METHOD MatrixDouble contiguous
{
  LUABIND_RETURN(MatrixDouble, obj->getIsContiguous() ? obj : obj->clone());
}
//BIND_END

//BIND_METHOD MatrixDouble map
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::map(L, obj));
}
//BIND_END

//BIND_METHOD MatrixDouble diagonalize
{
  LUABIND_RETURN(MatrixDouble, obj->diagonalize());
}
//BIND_END

//BIND_METHOD MatrixDouble get_shared_count
{
  LUABIND_RETURN(uint, obj->getSharedCount());
}
//BIND_END

//BIND_METHOD MatrixDouble reset_shared_count
{
  obj->resetSharedCount();
}
//BIND_END

//BIND_METHOD MatrixDouble add_to_shared_count
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::add_to_shared_count(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble update
{
  obj->update();
}
//BIND_END

//BIND_METHOD MatrixDouble padding_all
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::padding_all(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble padding
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::padding(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble uniform
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::uniform(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble uniformf
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::uniformf(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble linspace
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::linspace(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble logspace
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::logspace(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble linear
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::linear(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble fill
//DOC_BEGIN
// void fill(double value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, string);
  double value;
  LUABIND_GET_PARAMETER(1,double,value);
  LUABIND_RETURN(MatrixDouble, 
                 matFill(obj, value));
}
//BIND_END

//BIND_METHOD MatrixDouble zeros
{
  LUABIND_RETURN(MatrixDouble, 
                 matZeros(obj));
}
//BIND_END

//BIND_METHOD MatrixDouble ones
{
  LUABIND_RETURN(MatrixDouble, 
                 matOnes(obj));
}
//BIND_END

//BIND_METHOD MatrixDouble diag
{
  LUABIND_CHECK_ARGN(==,1);
  double v;
  LUABIND_GET_PARAMETER(1, double, v);
  LUABIND_RETURN(MatrixDouble, 
                 matDiag(obj,v));
}
//BIND_END

//BIND_METHOD MatrixDouble sliding_window
{
  int *sub_matrix_size=0, *offset=0, *step=0, *num_steps=0, *order_step=0;
  int argn = lua_gettop(L); // number of arguments
  const int num_dim = obj->getNumDim();
  if (argn > 1)
    LUABIND_ERROR("incorrect number of arguments");
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1,
		       "offset",
		       "size",
		       "step",
		       "numSteps",
		       "orderStep",
		       (const char*)0);
    
    offset = read_vector(L, "offset", num_dim, 0);
    sub_matrix_size = read_vector(L, "size", num_dim, 0);
    step = read_vector(L, "step", num_dim, 0);
    num_steps = read_vector(L, "numSteps", num_dim, 0);
    order_step = read_vector(L, "orderStep", num_dim, -1);
  }
  SlidingWindowMatrixDouble *window = new SlidingWindowMatrixDouble(obj,
								    sub_matrix_size,
								    offset,
								    step,
								    num_steps,
								    order_step);
  LUABIND_RETURN(SlidingWindowMatrixDouble, window);
  delete[] sub_matrix_size;
  delete[] offset;
  delete[] step;
  delete[] num_steps;
  delete[] order_step;
}
//BIND_END

//BIND_METHOD MatrixDouble is_contiguous
{
  LUABIND_RETURN(bool, obj->getIsContiguous());
}
//BIND_END

//BIND_METHOD MatrixDouble copy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 1);
  MatrixDouble *mat;
  LUABIND_GET_PARAMETER(1, MatrixDouble, mat);
  LUABIND_RETURN(MatrixDouble, 
                 matCopy(obj,mat));
}
//BIND_END

//// MATRIX SERIALIZATION ////

//BIND_CLASS_METHOD MatrixDouble deserialize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::deserialize(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixDouble read
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::read(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixDouble fromMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::fromMMap(L));
}
//BIND_END

//BIND_METHOD MatrixDouble toMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::toMMap(L, obj));
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_METHOD MatrixDouble data
{
  LUABIND_RETURN(DoubleGPUMirroredMemoryBlock, obj->getRawDataAccess());
}
//BIND_END

//BIND_METHOD MatrixDouble convert_to
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::convert_to(L,obj));
}
//BIND_END

//BIND_METHOD MatrixDouble equals
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<double>::equals(L,obj));
}
//BIND_END
