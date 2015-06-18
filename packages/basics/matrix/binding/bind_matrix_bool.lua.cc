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
#include "bind_matrix.h"
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

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(MatrixBool);

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

  int sliding_window_matrixBool_iterator_function(lua_State *L) {
    SlidingWindowMatrixBool *obj = lua_toSlidingWindowMatrixBool(L,1);
    if (obj->isEnd()) {
      lua_pushnil(L);
      return 1;
    }
    MatrixBool *mat = obj->getMatrix();
    lua_pushMatrixBool(L, mat);
    obj->next();
    return 1;
  }
  
}
//BIND_END

//BIND_HEADER_H
#include "matrixBool.h"
using namespace Basics;
typedef MatrixBool::sliding_window SlidingWindowMatrixBool;
//BIND_END

//BIND_LUACLASSNAME MatrixBool matrixBool
//BIND_CPP_CLASS MatrixBool
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixBool Serializable

//BIND_LUACLASSNAME SlidingWindowMatrixBool matrixBool.__sliding_window__
//BIND_CPP_CLASS SlidingWindowMatrixBool

//BIND_CONSTRUCTOR SlidingWindowMatrixBool
{
  LUABIND_ERROR("Use matrixBool.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool get_matrix
{
  MatrixBool *dest;
  LUABIND_GET_OPTIONAL_PARAMETER(1, MatrixBool, dest, 0);
  LUABIND_RETURN(MatrixBool, obj->getMatrix(dest));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool next
{
  LUABIND_RETURN(SlidingWindowMatrixBool, obj->next());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool set_at_window
{
  int windex;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, int, windex);
  if (windex < 1) LUABIND_ERROR("Index must be >= 1\n");
  obj->setAtWindow(windex-1);
  LUABIND_RETURN(SlidingWindowMatrixBool, obj);
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool num_windows
{
  LUABIND_RETURN(int, obj->numWindows());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool coords
{
  LUABIND_VECTOR_TO_NEW_TABLE(int, obj->getCoords(), obj->getNumDim());
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool is_end
{
  LUABIND_RETURN(bool, obj->isEnd());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixBool iterate
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,sliding_window_matrixBool_iterator_function);
  LUABIND_RETURN(SlidingWindowMatrixBool,obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixBool
{
  if (lua_isMatrixFloat(L,1)) {
    MatrixFloat *m;
    LUABIND_GET_PARAMETER(1, MatrixFloat, m);
    MatrixBool *obj = new MatrixBool(m->getNumDim(), m->getDimPtr());
    MatrixBool::iterator bool_it(obj->begin());
    MatrixFloat::const_iterator float_it(m->begin());
    while(bool_it != obj->end()) {
      if (*float_it == 0.0f) *bool_it = false;
      else if (*float_it == 1.0f) *bool_it = true;
      else LUABIND_ERROR("Needs a 0/1 matrix argument\n");
      ++bool_it;
      ++float_it;
    }
    LUABIND_RETURN(MatrixBool, obj);
  }
  else {
    LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::constructor(L));
  }
}
//BIND_END

//BIND_METHOD MatrixBool size
{
  LUABIND_RETURN(int, obj->size());
}
//BIND_END

//BIND_METHOD MatrixBool rewrap
{
  LUABIND_CHECK_ARGN(>=, 1);
  int ndims;
  ndims = lua_gettop(L); // number of dimensions
  int *dims = new int[ndims];
  for (int i=1; i <= ndims; i++) {
    LUABIND_GET_PARAMETER(i, int, dims[i-1]);
    if (dims[i-1] <= 0)
      LUABIND_FERROR1("incorrect argument to matrix dimension (arg %d must be >0)",i);
  }
  MatrixBool *new_obj = obj->rewrap(dims, ndims);
  delete[] dims;
  LUABIND_RETURN(MatrixBool,new_obj);
}
//BIND_END

//BIND_METHOD MatrixBool squeeze
{
  LUABIND_RETURN(MatrixBool,obj->squeeze());
}
//BIND_END

//BIND_METHOD MatrixBool get_reference_string
{
  char buff[128];
  sprintf(buff,"%p data= %p",
	  (void*)obj,
	  (void*)obj->getRawDataAccess());
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_METHOD MatrixBool copy_from_table
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  int veclen;
  LUABIND_TABLE_GETN(1, veclen);
  if (veclen != obj->size())
    LUABIND_FERROR2("wrong size %d instead of %d",veclen,obj->size());
  int i=1;
  for (MatrixBool::iterator it(obj->begin()); it != obj->end(); ++it) {
    lua_rawgeti(L,1,i);
    if (!lua_isboolean(L,-1))
      LUABIND_FERROR1("The given table has a no boolean value at position %d, "
                      "the table could be smaller than matrix size", i);
    *it = lua_toboolean(L,-1);
    lua_remove(L,-1);
    ++i;
  }
  LUABIND_RETURN(MatrixBool, obj);
}
//BIND_END

//BIND_METHOD MatrixBool get
//DOC_BEGIN
// bool get(coordinates)
/// Permite ver valores de una matriz. Requiere tantos indices como dimensiones tenga la matriz.
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim())
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim());
  bool ret;
  if (obj->getNumDim() == 1) {
    int v1;
    LUABIND_GET_PARAMETER(1,int,v1);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    ret = (*obj)(v1-1);
  }
  else if (obj->getNumDim() == 2) {
    int v1, v2;
    LUABIND_GET_PARAMETER(1,int,v1);
    LUABIND_GET_PARAMETER(2,int,v2);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    if (v2<1 || v2 > obj->getDimSize(1)) {
      LUABIND_FERROR2("wrong index parameter: 2 <= %d <= %d is incorrect",
		      v2, obj->getDimSize(1));
    }
    ret = (*obj)(v1-1, v2-1);
  }
  else {
    int *coords = new int[obj->getNumDim()];
    for (int i=0; i<obj->getNumDim(); ++i) {
      LUABIND_GET_PARAMETER(i+1,int,coords[i]);
      if (coords[i]<1 || coords[i] > obj->getDimSize(i)) {
	LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
			coords[i], obj->getDimSize(i));
      }
      coords[i]--;
    }
    ret = (*obj)(coords, obj->getNumDim());
    delete[] coords;
  }
  LUABIND_RETURN(boolean, ret);
}
//BIND_END

//BIND_METHOD MatrixBool set
//DOC_BEGIN
// bool set(coordinates,value)
/// Permite cambiar el valor de un elemento en la matriz. Requiere
/// tantos indices como dimensiones tenga la matriz y adicionalmente
/// el valor a cambiar
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim()+1)
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim()+1);
  bool v;
  if (obj->getNumDim() == 1) {
    int v1;
    LUABIND_GET_PARAMETER(1,int,v1);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,boolean,v);
    (*obj)(v1-1) = v;
  }
  else if (obj->getNumDim() == 2) {
    int v1, v2;
    LUABIND_GET_PARAMETER(1,int,v1);
    LUABIND_GET_PARAMETER(2,int,v2);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    if (v2<1 || v2 > obj->getDimSize(1)) {
      LUABIND_FERROR2("wrong index parameter: 2 <= %d <= %d is incorrect",
		      v2, obj->getDimSize(1));
    }
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,boolean,v);
    (*obj)(v1-1, v2-1) = v;
  }
  else {
    int *coords = new int[obj->getNumDim()];
    for (int i=0; i<obj->getNumDim(); ++i) {
      LUABIND_GET_PARAMETER(i+1,int,coords[i]);
      if (coords[i]<1 || coords[i] > obj->getDimSize(i)) {
	LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
			coords[i], obj->getDimSize(i));
      }
      coords[i]--;
    }
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,boolean,v);
    (*obj)(coords, obj->getNumDim()) = v;
    delete[] coords;
  }
  LUABIND_RETURN(MatrixBool, obj);
}
//BIND_END

//BIND_METHOD MatrixBool offset
{
  LUABIND_RETURN(int, obj->getOffset());
}
//BIND_END

//BIND_METHOD MatrixBool raw_get
{
  int raw_pos;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  LUABIND_RETURN(bool, (*obj)[raw_pos]);
}
//BIND_END

//BIND_METHOD MatrixBool raw_set
{
  int raw_pos;
  bool value;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  LUABIND_GET_PARAMETER(2, boolean, value);
  (*obj)[raw_pos] = value;
  LUABIND_RETURN(MatrixBool, obj);
}
//BIND_END

//BIND_METHOD MatrixBool fill
//DOC_BEGIN
// void fill(bool value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, boolean);
  bool value;
  LUABIND_GET_PARAMETER(1,boolean,value);
  LUABIND_RETURN(MatrixBool, 
                 matFill(obj, value));
}
//BIND_END

//BIND_METHOD MatrixBool zeros
{
  LUABIND_RETURN(MatrixBool, 
                 matFill(obj, false));
}
//BIND_END

//BIND_METHOD MatrixBool ones
{
  LUABIND_RETURN(MatrixBool, 
                 matFill(obj, true));
}
//BIND_END

//BIND_METHOD MatrixBool dim
{
  LUABIND_CHECK_ARGN(>=, 0);
  LUABIND_CHECK_ARGN(<=, 1);
  int pos;
  const int *d=obj->getDimPtr();
  LUABIND_GET_OPTIONAL_PARAMETER(1, int, pos, -1);
  if (pos < 1) {
    LUABIND_VECTOR_TO_NEW_TABLE(int, d, obj->getNumDim());
    LUABIND_RETURN_FROM_STACK(-1);
  }
  else LUABIND_RETURN(int, d[pos-1]);
}
//BIND_END

//BIND_METHOD MatrixBool num_dim
{
  LUABIND_RETURN(int, obj->getNumDim());
}
//BIND_END

//BIND_METHOD MatrixBool stride
{
  LUABIND_CHECK_ARGN(>=, 0);
  LUABIND_CHECK_ARGN(<=, 1);
  int pos;
  const int *s=obj->getStridePtr();
  LUABIND_GET_OPTIONAL_PARAMETER(1, int, pos, -1);
  if (pos < 1) {
    LUABIND_VECTOR_TO_NEW_TABLE(int, s, obj->getNumDim());
    LUABIND_RETURN_FROM_STACK(-1);
  }
  else LUABIND_RETURN(int, s[pos-1]);
}
//BIND_END

//BIND_METHOD MatrixBool slice
{
  LUABIND_CHECK_ARGN(>=,2);
  LUABIND_CHECK_ARGN(<=,3);
  LUABIND_CHECK_PARAMETER(1, table);
  LUABIND_CHECK_PARAMETER(2, table);
  int *coords, *sizes, coords_len, sizes_len;
  bool clone;
  LUABIND_TABLE_GETN(1, coords_len);
  LUABIND_TABLE_GETN(2, sizes_len);
  if (coords_len != sizes_len || coords_len != obj->getNumDim())
    LUABIND_FERROR3("Incorrect number of dimensions, expected %d, "
		    "found %d and %d\n",
		    obj->getNumDim(), coords_len, sizes_len);
  coords = new int[coords_len];
  sizes  = new int[sizes_len];
  LUABIND_TABLE_TO_VECTOR_SUB1(1, int, coords, coords_len);
  LUABIND_TABLE_TO_VECTOR(2, int, sizes,  sizes_len);
  for (int i=0; i<sizes_len; ++i)
    if (coords[i] < 0 || sizes[i] < 1 ||
	sizes[i]+coords[i] > obj->getDimSize(i))
      LUABIND_FERROR1("Incorrect size or coord at position %d\n", i+1);
  LUABIND_GET_OPTIONAL_PARAMETER(3, bool, clone, false);
  MatrixBool *obj2 = new MatrixBool(obj, coords, sizes, clone);
  LUABIND_RETURN(MatrixBool, obj2);
  delete[] coords;
  delete[] sizes;
}
//BIND_END

//BIND_METHOD MatrixBool select
{
  LUABIND_CHECK_ARGN(>=,2);
  LUABIND_CHECK_ARGN(<=,3);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(2, int);
  int dim, index;
  MatrixBool *dest;
  LUABIND_GET_PARAMETER(1, int, dim);
  LUABIND_GET_PARAMETER(2, int, index);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixBool, dest, 0);
  MatrixBool *obj2 = obj->select(dim-1, index-1, dest);
  LUABIND_RETURN(MatrixBool, obj2);
}
//BIND_END

//BIND_METHOD MatrixBool clone
//DOC_BEGIN
// matrix *clone()
/// Devuelve un <em>clon</em> de la matriz.
//DOC_END
{
  MatrixBool *obj2 = obj->clone();
  LUABIND_RETURN(MatrixBool,obj2);
}
//BIND_END

//BIND_METHOD MatrixBool transpose
{
  int argn;
  argn = lua_gettop(L);
  if (argn == 0) {
    LUABIND_RETURN(MatrixBool, obj->transpose());
  }
  else {
    int d1,d2;
    LUABIND_GET_PARAMETER(1, int, d1);
    LUABIND_GET_PARAMETER(2, int, d2);
    LUABIND_RETURN(MatrixBool, obj->transpose(d1-1, d2-1));
  }
}
//BIND_END

//BIND_METHOD MatrixBool diag
{
  LUABIND_CHECK_ARGN(==,1);
  bool v;
  LUABIND_GET_PARAMETER(1, boolean, v);
  LUABIND_RETURN(MatrixBool, 
                 matDiag(obj, v));
}
//BIND_END

//BIND_METHOD MatrixBool toTable
// Permite salvar una matriz en una tabla lua
// TODO: Tener en cuenta las dimensiones de la matriz
  {
    LUABIND_CHECK_ARGN(==, 0);
    lua_createtable(L, obj->size(), 0);
    int index = 1;
    for (MatrixBool::iterator it(obj->begin()); it != obj->end(); ++it) {
      lua_pushboolean(L, *it);
      lua_rawseti(L, -2, index++);
    }
    LUABIND_RETURN_FROM_STACK(-1);
  }
//BIND_END

//BIND_METHOD MatrixBool sliding_window
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
  SlidingWindowMatrixBool *window = new SlidingWindowMatrixBool(obj,
								sub_matrix_size,
								offset,
								step,
								num_steps,
								order_step);
  LUABIND_RETURN(SlidingWindowMatrixBool, window);
  delete[] sub_matrix_size;
  delete[] offset;
  delete[] step;
  delete[] num_steps;
  delete[] order_step;
}
//BIND_END

//BIND_METHOD MatrixBool is_contiguous
{
  LUABIND_RETURN(bool, obj->getIsContiguous());
}
//BIND_END

//BIND_METHOD MatrixBool complement
{
  for (MatrixBool::iterator it(obj->begin());
       it != obj->end(); ++it) {
    if (*it) *it = false;
    else *it = true;
  }
  LUABIND_RETURN(MatrixBool, obj);
}
//BIND_END

//BIND_METHOD MatrixBool copy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 1);
  MatrixBool *mat;
  LUABIND_GET_PARAMETER(1, MatrixBool, mat);
  LUABIND_RETURN(MatrixBool, 
                 matCopy(obj, mat));
}
//BIND_END

//BIND_METHOD MatrixBool count_ones
{
  int count=0;
  for (MatrixBool::const_iterator it(obj->begin());
       it != obj->end(); ++it) {
    if (*it) ++count;
  }
  LUABIND_RETURN(int, count);
}
//BIND_END

//BIND_METHOD MatrixBool count_zeros
{
  int count=0;
  for (MatrixBool::const_iterator it(obj->begin());
       it != obj->end(); ++it) {
    if (!(*it)) ++count;
  }
  LUABIND_RETURN(int, count);
}
//BIND_END

//BIND_METHOD MatrixBool any
{
  bool result = false;
  for (MatrixBool::const_iterator it(obj->begin());
       it != obj->end() && !result; ++it) {
    result = result || (*it);
  }
  LUABIND_RETURN(boolean, result);
}
//BIND_END

//BIND_METHOD MatrixBool all
{
  bool result = true;
  for (MatrixBool::const_iterator it(obj->begin());
       it != obj->end() && result; ++it) {
    result = result && (*it);
  }
  LUABIND_RETURN(boolean, result);
}
//BIND_END

//BIND_METHOD MatrixBool map
{
  int argn;
  int N;
  argn = lua_gettop(L); // number of arguments
  N = argn-1;
  MatrixBool **v = 0;
  MatrixBool::const_iterator *list_it = 0;
  if (N > 0) {
    v = new MatrixBool*[N];
    list_it = new MatrixBool::const_iterator[N];
  }
  for (int i=0; i<N; ++i) {
    LUABIND_CHECK_PARAMETER(i+1, MatrixBool);
    LUABIND_GET_PARAMETER(i+1, MatrixBool, v[i]);
    if (!v[i]->sameDim(obj))
      LUABIND_ERROR("The given matrices must have the same dimension sizes\n");
    list_it[i] = v[i]->begin();
  }
  LUABIND_CHECK_PARAMETER(argn, function);
  for (MatrixBool::iterator it(obj->begin()); it!=obj->end(); ++it) {
    // copy the Lua function, lua_call will pop this copy
    lua_pushvalue(L, argn);
    // push the self matrix value
    lua_pushboolean(L, *it);
    // push the value of the rest of given matrices
    for (int j=0; j<N; ++j) {
      lua_pushboolean(L, *list_it[j]);
      ++list_it[j];
    }
    // CALL
    lua_call(L, N+1, 1);
    // pop the result, a number
    if (!lua_isnil(L, -1)) {
      if (!lua_isboolean(L, -1))
	LUABIND_ERROR("Incorrect returned value type, expected NIL or COMPLEX\n");
      *it = lua_toboolean(L, -1);
    }
    lua_pop(L, 1);
  }
  delete[] v;
  delete[] list_it;
  LUABIND_RETURN(MatrixBool, obj);
}
//BIND_END

//// MATRIX SERIALIZATION ////

//BIND_CLASS_METHOD MatrixBool deserialize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::
                               deserialize(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixBool read
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::
                               read(L));
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_METHOD MatrixBool convert_to
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::convert_to(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool to_index
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::to_index(L,obj));
}
//BIND_END

//BIND_METHOD MatrixBool equals
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<bool>::equals(L,obj));
}
//BIND_END
