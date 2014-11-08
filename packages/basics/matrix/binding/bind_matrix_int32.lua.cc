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
#include "bind_mtrand.h"
#include "utilMatrixInt32.h"
#include "luabindutil.h"
#include "luabindmacros.h"

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

  int sliding_window_matrixInt32_iterator_function(lua_State *L) {
    SlidingWindowMatrixInt32 *obj = lua_toSlidingWindowMatrixInt32(L,1);
    if (obj->isEnd()) {
      lua_pushnil(L);
      return 1;
    }
    MatrixInt32 *mat = obj->getMatrix();
    lua_pushMatrixInt32(L, mat);
    obj->next();
    return 1;
  }

  static int32_t april_optint(lua_State *L, int i, int32_t opt) {
    if (lua_type(L,i) == LUA_TNONE || lua_isnil(L,i)) return opt;
    return lua_toint(L,i);
  }

}

//BIND_END

//BIND_HEADER_H
#include "matrixInt32.h"
using namespace Basics;
typedef MatrixInt32::sliding_window SlidingWindowMatrixInt32;
//BIND_END

//BIND_LUACLASSNAME MatrixInt32 matrixInt32
//BIND_CPP_CLASS MatrixInt32
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixInt32 Serializable

//BIND_LUACLASSNAME SlidingWindowMatrixInt32 matrixInt32.__sliding_window__
//BIND_CPP_CLASS SlidingWindowMatrixInt32

//BIND_CONSTRUCTOR SlidingWindowMatrixInt32
{
  LUABIND_ERROR("Use matrixInt32.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 get_matrix
{
  MatrixInt32 *dest;
  LUABIND_GET_OPTIONAL_PARAMETER(1, MatrixInt32, dest, 0);
  LUABIND_RETURN(MatrixInt32, obj->getMatrix(dest));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 next
{
  LUABIND_RETURN(SlidingWindowMatrixInt32, obj->next());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 set_at_window
{
  int windex;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, int, windex);
  if (windex < 1) LUABIND_ERROR("Index must be >= 1\n");
  obj->setAtWindow(windex-1);
  LUABIND_RETURN(SlidingWindowMatrixInt32, obj);
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 num_windows
{
  LUABIND_RETURN(int, obj->numWindows());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 coords
{
  LUABIND_VECTOR_TO_NEW_TABLE(int, obj->getCoords(), obj->getNumDim());
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 is_end
{
  LUABIND_RETURN(bool, obj->isEnd());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixInt32 iterate
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,sliding_window_matrixInt32_iterator_function);
  LUABIND_RETURN(SlidingWindowMatrixInt32,obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixInt32
{
  int i,argn;
  argn = lua_gettop(L); // number of arguments
  LUABIND_CHECK_ARGN(>=, 1);
  int ndims = (!lua_isnumber(L,argn)) ? argn-1 : argn;
  int *dim;
  if (ndims == 0) { // caso matrix{valores}
    ndims = 1;
    dim = new int[ndims];
    LUABIND_TABLE_GETN(1, dim[0]);
  } else {
    dim = new int[ndims];
    for (i=1; i <= ndims; i++) {
      if (!lua_isnumber(L,i))
	// TODO: Este mensaje de error parece que no es correcto... y no se todavia por que!!!
	LUABIND_FERROR2("incorrect argument to matrix dimension (arg %d must"
			" be a number and is a %s)",
			i, lua_typename(L,i));
      dim[i-1] = (int)lua_tonumber(L,i);
      if (dim[i-1] <= 0)
	LUABIND_FERROR1("incorrect argument to matrix dimension (arg %d must be >0)",i);
    }
  }
  MatrixInt32* obj;
  obj = new MatrixInt32(ndims,dim);
  if (lua_istable(L,argn)) {
    int i=1;
    for (MatrixInt32::iterator it(obj->begin()); it != obj->end(); ++i, ++it) {
      lua_rawgeti(L,argn,i);
      int32_t v = luaL_checkint(L,-1);
      *it = v;
      lua_remove(L,-1);
    }
  }
  delete[] dim;
  LUABIND_RETURN(MatrixInt32,obj);
}
//BIND_END

//BIND_METHOD MatrixInt32 size
{
  LUABIND_RETURN(int, obj->size());
}
//BIND_END

//BIND_METHOD MatrixInt32 rewrap
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
  MatrixInt32 *new_obj = obj->rewrap(dims, ndims);
  delete[] dims;
  LUABIND_RETURN(MatrixInt32,new_obj);
}
//BIND_END

//BIND_METHOD MatrixInt32 squeeze
{
  LUABIND_RETURN(MatrixInt32,obj->squeeze());
}
//BIND_END

//BIND_METHOD MatrixInt32 get_reference_string
{
  char buff[128];
  sprintf(buff,"%p data= %p",
	  (void*)obj,
	  (void*)obj->getRawDataAccess());
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_METHOD MatrixInt32 copy_from_table
//DOC_BEGIN
// void copy_from_table(table matrix_values)
/// Permite dar valores a una matriz. Require una tabla con un numero
/// de argumentos igual al numero de elementos de la matriz.
///@param matrix_values Tabla con los elementos de la matriz.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  int veclen;
  LUABIND_TABLE_GETN(1, veclen);
  if (veclen != obj->size())
    LUABIND_FERROR2("wrong size %d instead of %d",veclen,obj->size());
  int i=1;
  for (MatrixInt32::iterator it(obj->begin()); it != obj->end(); ++i, ++it) {
    lua_rawgeti(L,1,i);
    int32_t v = luaL_checkint(L,-1);
    *it = v;
    lua_remove(L,-1);
  }
  LUABIND_RETURN(MatrixInt32, obj);
}
//BIND_END

//BIND_METHOD MatrixInt32 get
//DOC_BEGIN
// float get(coordinates)
/// Permite ver valores de una matriz. Requiere tantos indices como dimensiones tenga la matriz.
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim())
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim());
  int32_t ret;
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
  LUABIND_RETURN(int, ret);
}
//BIND_END

//BIND_METHOD MatrixInt32 set
//DOC_BEGIN
// float set(coordinates,value)
/// Permite cambiar el valor de un elemento en la matriz. Requiere
/// tantos indices como dimensiones tenga la matriz y adicionalmente
/// el valor a cambiar
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim()+1)
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim()+1);
  int32_t value;
  if (obj->getNumDim() == 1) {
    int v1;
    LUABIND_GET_PARAMETER(1,int,v1);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,int,value);
    (*obj)(v1-1) = value;
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
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,int,value);
    (*obj)(v1-1, v2-1) = value;
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
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,int,value);
    (*obj)(coords, obj->getNumDim()) = value;
    delete[] coords;
  }
  LUABIND_RETURN(MatrixInt32, obj);
}
//BIND_END

//BIND_METHOD MatrixInt32 offset
{
  LUABIND_RETURN(int, obj->getOffset());
}
//BIND_END

//BIND_METHOD MatrixInt32 raw_get
{
  int raw_pos;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  LUABIND_RETURN(int, (*obj)[raw_pos]);
}
//BIND_END

//BIND_METHOD MatrixInt32 raw_set
{
  int raw_pos;
  int value;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  LUABIND_GET_PARAMETER(2, int, value);
  (*obj)[raw_pos] = static_cast<int32_t>(value);
  LUABIND_RETURN(MatrixInt32, obj);
}
//BIND_END

//BIND_METHOD MatrixInt32 fill
//DOC_BEGIN
// void fill(float value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, string);
  int value;
  LUABIND_GET_PARAMETER(1,int,value);
  LUABIND_RETURN(MatrixInt32, AprilMath::MatrixExt::Operations::
                 matFill(obj, static_cast<int32_t>(value)));
}
//BIND_END

//BIND_METHOD MatrixInt32 zeros
{
  LUABIND_RETURN(MatrixInt32, AprilMath::MatrixExt::Operations::
                 matZeros(obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 ones
{
  LUABIND_RETURN(MatrixInt32, AprilMath::MatrixExt::Operations::
                 matOnes(obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 dim
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

//BIND_METHOD MatrixInt32 stride
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

//BIND_METHOD MatrixInt32 slice
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
  MatrixInt32 *obj2 = new MatrixInt32(obj, coords, sizes, clone);
  LUABIND_RETURN(MatrixInt32, obj2);
  delete[] coords;
  delete[] sizes;
}
//BIND_END

//BIND_METHOD MatrixInt32 select
{
  LUABIND_CHECK_ARGN(>=,2);
  LUABIND_CHECK_ARGN(<=,3);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(2, int);
  int dim, index;
  MatrixInt32 *dest;
  LUABIND_GET_PARAMETER(1, int, dim);
  LUABIND_GET_PARAMETER(2, int, index);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixInt32, dest, 0);
  MatrixInt32 *obj2 = obj->select(dim-1, index-1, dest);
  LUABIND_RETURN(MatrixInt32, obj2);
}
//BIND_END

//BIND_METHOD MatrixInt32 clone
//DOC_BEGIN
// matrix *clone()
/// Devuelve un <em>clon</em> de la matriz.
//DOC_END
{
  MatrixInt32 *obj2 = obj->clone();
  LUABIND_RETURN(MatrixInt32,obj2);
}
//BIND_END

//BIND_METHOD MatrixInt32 transpose
{
  int argn;
  argn = lua_gettop(L);
  if (argn == 0) {
    LUABIND_RETURN(MatrixInt32, obj->transpose());
  }
  else {
    int d1,d2;
    LUABIND_GET_PARAMETER(1, int, d1);
    LUABIND_GET_PARAMETER(2, int, d2);
    LUABIND_RETURN(MatrixInt32, obj->transpose(d1-1, d2-1));
  }
}
//BIND_END

//BIND_METHOD MatrixInt32 diag
{
  LUABIND_CHECK_ARGN(==,1);
  int v;
  LUABIND_GET_PARAMETER(1, int, v);
  LUABIND_RETURN(MatrixInt32, AprilMath::MatrixExt::Operations::
                 matDiag(obj, static_cast<int32_t>(v)));
}
//BIND_END

//BIND_METHOD MatrixInt32 toTable
// Permite salvar una matriz en una tabla lua
// TODO: Tener en cuenta las dimensiones de la matriz
  {
    LUABIND_CHECK_ARGN(==, 0);
    lua_createtable(L, obj->size(), 0);
    int index = 1;
    for (MatrixInt32::iterator it(obj->begin()); it != obj->end(); ++it) {
      lua_pushint(L, *it);
      lua_rawseti(L, -2, index++);
    }
    LUABIND_RETURN_FROM_STACK(-1);
  }
//BIND_END

//BIND_METHOD MatrixInt32 map
{
  int argn;
  int N;
  argn = lua_gettop(L); // number of arguments
  N = argn-1;
  MatrixInt32 **v = 0;
  MatrixInt32::const_iterator *list_it = 0;
  if (N > 0) {
    v = new MatrixInt32*[N];
    list_it = new MatrixInt32::const_iterator[N];
  }
  for (int i=0; i<N; ++i) {
    LUABIND_CHECK_PARAMETER(i+1, MatrixInt32);
    LUABIND_GET_PARAMETER(i+1, MatrixInt32, v[i]);
    if (!v[i]->sameDim(obj))
      LUABIND_ERROR("The given matrices must have the same dimension sizes\n");
    list_it[i] = v[i]->begin();
  }
  LUABIND_CHECK_PARAMETER(argn, function);
  for (MatrixInt32::iterator it(obj->begin()); it!=obj->end(); ++it) {
    // copy the Lua function, lua_call will pop this copy
    lua_pushvalue(L, argn);
    // push the self matrix value
    lua_pushint(L, *it);
    // push the value of the rest of given matrices
    for (int j=0; j<N; ++j) {
      lua_pushint(L, *list_it[j]);
      ++list_it[j];
    }
    // CALL
    lua_call(L, N+1, 1);
    // pop the result, a number
    if (!lua_isnil(L, -1)) {
      if (!lua_isint(L, -1))
	LUABIND_ERROR("Incorrect returned value type, expected NIL or INT\n");
      *it = lua_toint(L, -1);
    }
    lua_pop(L, 1);
  }
  delete[] v;
  delete[] list_it;
  LUABIND_RETURN(MatrixInt32, obj);
}
//BIND_END

//BIND_METHOD MatrixInt32 sliding_window
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
  SlidingWindowMatrixInt32 *window = new SlidingWindowMatrixInt32(obj,
								  sub_matrix_size,
								  offset,
								  step,
								  num_steps,
								  order_step);
  LUABIND_RETURN(SlidingWindowMatrixInt32, window);
  delete[] sub_matrix_size;
  delete[] offset;
  delete[] step;
  delete[] num_steps;
  delete[] order_step;
}
//BIND_END

//BIND_METHOD MatrixInt32 is_contiguous
{
  LUABIND_RETURN(bool, obj->getIsContiguous());
}
//BIND_END

//BIND_METHOD MatrixInt32 to_float
{
  LUABIND_RETURN(MatrixFloat, convertFromMatrixInt32ToMatrixFloat(obj));
}
//BIND_END

//BIND_METHOD MatrixInt32 copy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 1);
  MatrixInt32 *mat;
  LUABIND_GET_PARAMETER(1, MatrixInt32, mat);
  LUABIND_RETURN(MatrixInt32, AprilMath::MatrixExt::Operations::
                 matCopy(obj, mat));
}
//BIND_END

//BIND_METHOD MatrixInt32 uniform
{
  int lower, upper;
  MTRand *random;
  LUABIND_GET_PARAMETER(1, int, lower);
  LUABIND_GET_PARAMETER(2, int, upper);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MTRand, random, 0);
  
  if (lower > upper) {
    LUABIND_ERROR("First argument must be <= second argument");
  }
  if (random == 0) random = new MTRand();
  IncRef(random);
  for (MatrixInt32::iterator it(obj->begin()); it != obj->end(); ++it) {
    *it = random->randInt(upper - lower) + lower;
  }
  DecRef(random);
  LUABIND_RETURN(MatrixInt32, obj);
}
//BIND_END

//// MATRIX SERIALIZATION ////

//BIND_CLASS_METHOD MatrixInt32 read
{
  MAKE_READ_MATRIX_LUA_METHOD(MatrixInt32, int32_t);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

