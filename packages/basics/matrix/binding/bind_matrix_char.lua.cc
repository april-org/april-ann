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
#include "utilMatrixChar.h"
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

  int sliding_window_matrixChar_iterator_function(lua_State *L) {
    SlidingWindowMatrixChar *obj = lua_toSlidingWindowMatrixChar(L,1);
    if (obj->isEnd()) {
      lua_pushnil(L);
      return 1;
    }
    MatrixChar *mat = obj->getMatrix();
    lua_pushMatrixChar(L, mat);
    obj->next();
    return 1;
  }

  static char april_optchar(lua_State *L, int i, char opt) {
    if (lua_type(L,i) == LUA_TNONE || lua_isnil(L,i)) return opt;
    const char *str = luaL_checkstring(L,i);
    return str[0];
  }
}
//BIND_END

//BIND_HEADER_H
#include "matrixChar.h"
using namespace Basics;
typedef MatrixChar::sliding_window SlidingWindowMatrixChar;
//BIND_END

//BIND_LUACLASSNAME MatrixChar matrixChar
//BIND_CPP_CLASS MatrixChar
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixChar Serializable

//BIND_LUACLASSNAME SlidingWindowMatrixChar matrixChar.__sliding_window__
//BIND_CPP_CLASS SlidingWindowMatrixChar

//BIND_CONSTRUCTOR SlidingWindowMatrixChar
{
  LUABIND_ERROR("Use matrixChar.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar get_matrix
{
  MatrixChar *dest;
  LUABIND_GET_OPTIONAL_PARAMETER(1, MatrixChar, dest, 0);
  LUABIND_RETURN(MatrixChar, obj->getMatrix(dest));
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar next
{
  LUABIND_RETURN(SlidingWindowMatrixChar, obj->next());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar set_at_window
{
  int windex;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, int, windex);
  if (windex < 1) LUABIND_ERROR("Index must be >= 1\n");
  obj->setAtWindow(windex-1);
  LUABIND_RETURN(SlidingWindowMatrixChar, obj);
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar num_windows
{
  LUABIND_RETURN(int, obj->numWindows());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar coords
{
  LUABIND_VECTOR_TO_NEW_TABLE(int, obj->getCoords(), obj->getNumDim());
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar is_end
{
  LUABIND_RETURN(bool, obj->isEnd());
}
//BIND_END

//BIND_METHOD SlidingWindowMatrixChar iterate
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,sliding_window_matrixChar_iterator_function);
  LUABIND_RETURN(SlidingWindowMatrixChar,obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixChar
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
  MatrixChar* obj;
  obj = new MatrixChar(ndims,dim);
  if (lua_istable(L,argn)) {
    int i=1;
    for (MatrixChar::iterator it(obj->begin()); it != obj->end(); ++i) {
      lua_rawgeti(L,argn,i);
      const char *data = luaL_checkstring(L,-1);
      while(it != obj->end() && data != '\0') {
	*it = *data;
	++it;
	++data;
      }
      lua_remove(L,-1);
    }
  }
  delete[] dim;
  LUABIND_RETURN(MatrixChar,obj);
}
//BIND_END

//BIND_METHOD MatrixChar size
{
  LUABIND_RETURN(int, obj->size());
}
//BIND_END

//BIND_METHOD MatrixChar rewrap
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
  MatrixChar *new_obj = obj->rewrap(dims, ndims);
  delete[] dims;
  LUABIND_RETURN(MatrixChar,new_obj);
}
//BIND_END

//BIND_METHOD MatrixChar get_reference_string
{
  char buff[128];
  sprintf(buff,"%p data= %p",
	  (void*)obj,
	  (void*)obj->getRawDataAccess());
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_METHOD MatrixChar copy_from_table
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  int veclen;
  LUABIND_TABLE_GETN(1, veclen);
  if (veclen != obj->size())
    LUABIND_FERROR2("wrong size %d instead of %d",veclen,obj->size());
  int i=1;
  for (MatrixChar::iterator it(obj->begin()); it != obj->end(); ++i) {
    lua_rawgeti(L,1,i);
    const char *data = luaL_checkstring(L,-1);
    while(it != obj->end() && data != '\0') {
      *it = *data;
	++it;
	++data;
    }
    lua_remove(L,-1);
  }
  LUABIND_RETURN(MatrixChar, obj);
}
//BIND_END

//BIND_METHOD MatrixChar get
//DOC_BEGIN
// char get(coordinates)
/// Permite ver valores de una matriz. Requiere tantos indices como dimensiones tenga la matriz.
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim())
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim());
  char ret;
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
  LUABIND_RETURN(char, ret);
}
//BIND_END

//BIND_METHOD MatrixChar set
//DOC_BEGIN
// char set(coordinates,value)
/// Permite cambiar el valor de un elemento en la matriz. Requiere
/// tantos indices como dimensiones tenga la matriz y adicionalmente
/// el valor a cambiar
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim()+1)
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim()+1);
  const char *str;
  if (obj->getNumDim() == 1) {
    int v1;
    LUABIND_GET_PARAMETER(1,int,v1);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,string,str);
    (*obj)(v1-1) = *str;
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
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,string,str);
    (*obj)(v1-1, v2-1) = *str;
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
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,string,str);
    (*obj)(coords, obj->getNumDim()) = *str;
    delete[] coords;
  }
  LUABIND_RETURN(MatrixChar, obj);
}
//BIND_END

//BIND_METHOD MatrixChar offset
{
  LUABIND_RETURN(int, obj->getOffset());
}
//BIND_END

//BIND_METHOD MatrixChar raw_get
{
  int raw_pos;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  LUABIND_RETURN(char, (*obj)[raw_pos]);
}
//BIND_END

//BIND_METHOD MatrixChar raw_set
{
  int raw_pos;
  const char *value;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  LUABIND_GET_PARAMETER(2, string, value);
  (*obj)[raw_pos] = *value;
  LUABIND_RETURN(MatrixChar, obj);
}
//BIND_END

//BIND_METHOD MatrixChar fill
//DOC_BEGIN
// void fill(char value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, string);
  const char *value;
  LUABIND_GET_PARAMETER(1,string,value);
  LUABIND_RETURN(MatrixChar, AprilMath::MatrixExt::Operations::
                 matFill(obj, *value));
}
//BIND_END

//BIND_METHOD MatrixChar dim
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

//BIND_METHOD MatrixChar stride
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

//BIND_METHOD MatrixChar slice
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
  MatrixChar *obj2 = new MatrixChar(obj, coords, sizes, clone);
  LUABIND_RETURN(MatrixChar, obj2);
  delete[] coords;
  delete[] sizes;
}
//BIND_END

//BIND_METHOD MatrixChar select
{
  LUABIND_CHECK_ARGN(>=,2);
  LUABIND_CHECK_ARGN(<=,3);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(2, int);
  int dim, index;
  MatrixChar *dest;
  LUABIND_GET_PARAMETER(1, int, dim);
  LUABIND_GET_PARAMETER(2, int, index);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixChar, dest, 0);
  MatrixChar *obj2 = obj->select(dim-1, index-1, dest);
  LUABIND_RETURN(MatrixChar, obj2);
}
//BIND_END

//BIND_METHOD MatrixChar clone
//DOC_BEGIN
// matrix *clone()
/// Devuelve un <em>clon</em> de la matriz.
//DOC_END
{
  MatrixChar *obj2 = obj->clone();
  LUABIND_RETURN(MatrixChar,obj2);
}
//BIND_END

//BIND_METHOD MatrixChar transpose
{
  LUABIND_RETURN(MatrixChar, obj->transpose());
}
//BIND_END

//BIND_METHOD MatrixChar diag
{
  LUABIND_CHECK_ARGN(==,1);
  const char *v;
  LUABIND_GET_PARAMETER(1, string, v);
  LUABIND_RETURN(MatrixChar, AprilMath::MatrixExt::Operations::
                 matDiag(obj, *v));
}
//BIND_END

//BIND_METHOD MatrixChar toTable
// Permite salvar una matriz en una tabla lua
// TODO: Tener en cuenta las dimensiones de la matriz
  {
    LUABIND_CHECK_ARGN(==, 0);
    lua_createtable(L, obj->size(), 0);
    int index = 1;
    for (MatrixChar::iterator it(obj->begin()); it != obj->end(); ++it) {
      char aux[2] = { *it, '\0' };
      lua_pushstring(L, aux);
      lua_rawseti(L, -2, index++);
    }
    LUABIND_RETURN_FROM_STACK(-1);
  }
//BIND_END

//BIND_METHOD MatrixChar sliding_window
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
  SlidingWindowMatrixChar *window = new SlidingWindowMatrixChar(obj,
								sub_matrix_size,
								offset,
								step,
								num_steps,
								order_step);
  LUABIND_RETURN(SlidingWindowMatrixChar, window);
  delete[] sub_matrix_size;
  delete[] offset;
  delete[] step;
  delete[] num_steps;
  delete[] order_step;
}
//BIND_END

//BIND_METHOD MatrixChar is_contiguous
{
  LUABIND_RETURN(bool, obj->getIsContiguous());
}
//BIND_END

//BIND_METHOD MatrixChar to_string_table
{
  SlidingWindowMatrixChar *window = new SlidingWindowMatrixChar(obj);
  IncRef(window);
  MatrixChar *m = window->getMatrix();
  IncRef(m);
  char *str = new char[m->size()+1];
  lua_createtable(L, window->numWindows(), 0);
  int index = 1;
  do {
    window->getMatrix(m);
    int i=0;
    for (MatrixChar::const_iterator it(m->begin()); it!=m->end(); ++it)
      str[i++] = *it;
    str[i] = '\0';
    lua_pushstring(L, str);
    lua_rawseti(L, -2, index++);
    window->next();
  } while(!window->isEnd());
  DecRef(window);
  DecRef(m);
  delete[] str;
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD MatrixChar copy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 1);
  MatrixChar *mat;
  LUABIND_GET_PARAMETER(1, MatrixChar, mat);
  LUABIND_RETURN(MatrixChar, AprilMath::MatrixExt::Operations::
                 matCopy(obj, mat));
}
//BIND_END

//// MATRIX SERIALIZATION ////

//BIND_CLASS_METHOD MatrixChar read
{
  MAKE_READ_MATRIX_LUA_METHOD(MatrixChar, char);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

