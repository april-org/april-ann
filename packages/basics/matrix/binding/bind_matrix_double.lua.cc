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
#include "utilMatrixDouble.h"
#include "luabindutil.h"
#include "luabindmacros.h"

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

//BIND_END

//BIND_HEADER_H
#include "matrixDouble.h"
typedef MatrixDouble::sliding_window SlidingWindowMatrixDouble;
//BIND_END

//BIND_LUACLASSNAME MatrixDouble matrixDouble
//BIND_CPP_CLASS MatrixDouble

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
  MatrixDouble* obj;
  obj = new MatrixDouble(ndims,dim);
  if (lua_istable(L,argn)) {
    int i=1;
    for (MatrixDouble::iterator it(obj->begin()); it != obj->end(); ++i, ++it) {
      lua_rawgeti(L,argn,i);
      double v = luaL_checknumber(L,-1);
      *it = v;
      lua_remove(L,-1);
    }
  }
  delete[] dim;
  LUABIND_RETURN(MatrixDouble,obj);
}
//BIND_END

//BIND_METHOD MatrixDouble size
{
  LUABIND_RETURN(int, obj->size());
}
//BIND_END

//BIND_METHOD MatrixDouble rewrap
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
  MatrixDouble *new_obj = obj->rewrap(dims, ndims);
  delete[] dims;
  LUABIND_RETURN(MatrixDouble,new_obj);
}
//BIND_END

//BIND_METHOD MatrixDouble get_reference_string
{
  char buff[128];
  sprintf(buff,"%p data= %p",
	  (void*)obj,
	  (void*)obj->getRawDataAccess());
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_CLASS_METHOD MatrixDouble fromFilename
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, string);
  const char *filename;
  LUABIND_GET_PARAMETER(1,string,filename);
  MatrixDouble *obj;
  if ((obj = readMatrixDoubleFromFile(filename)) == 0)
    LUABIND_ERROR("bad format");
  else LUABIND_RETURN(MatrixDouble,obj);
}
//BIND_END


//BIND_CLASS_METHOD MatrixDouble fromString
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, string);
  constString cs;
  LUABIND_GET_PARAMETER(1,constString,cs);
  MatrixDouble *obj;
  if ((obj = readMatrixDoubleFromString(cs)) == 0)
    LUABIND_ERROR("bad format");
  else LUABIND_RETURN(MatrixDouble,obj);
}
//BIND_END

//BIND_METHOD MatrixDouble toFilename
{
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 2);
  const char *filename;
  constString cs;
  LUABIND_GET_PARAMETER(1, string, filename);
  LUABIND_GET_OPTIONAL_PARAMETER(2,constString,cs,constString("ascii"));
  bool is_ascii = (cs == "ascii");
  writeMatrixDoubleToFile(obj, filename, is_ascii);
}
//BIND_END

//BIND_METHOD MatrixDouble toString
{
  LUABIND_CHECK_ARGN(<=, 1);
  constString cs;
  LUABIND_GET_OPTIONAL_PARAMETER(1,constString,cs,constString("ascii"));
  bool is_ascii = (cs == "ascii");
  writeMatrixDoubleToLuaString(obj, L, is_ascii);
  LUABIND_INCREASE_NUM_RETURNS(1);
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
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  int veclen;
  LUABIND_TABLE_GETN(1, veclen);
  if (veclen != obj->size())
    LUABIND_FERROR2("wrong size %d instead of %d",veclen,obj->size());
  int i=1;
  for (MatrixDouble::iterator it(obj->begin()); it != obj->end(); ++i, ++it) {
    lua_rawgeti(L,1,i);
    double v = luaL_checknumber(L,-1);
    *it = v;
    lua_remove(L,-1);
  }
  LUABIND_RETURN(MatrixDouble, obj);
}
//BIND_END

//BIND_METHOD MatrixDouble get
//DOC_BEGIN
// double get(coordinates)
/// Permite ver valores de una matriz. Requiere tantos indices como dimensiones tenga la matriz.
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim())
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim());
  double ret;
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
  LUABIND_RETURN(double, ret);
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
  int argn = lua_gettop(L); // number of arguments
  if (argn != obj->getNumDim()+1)
    LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim()+1);
  double value;
  if (obj->getNumDim() == 1) {
    int v1;
    LUABIND_GET_PARAMETER(1,int,v1);
    if (v1<1 || v1 > obj->getDimSize(0)) {
      LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
		      v1, obj->getDimSize(0));
    }
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,double,value);
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
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,double,value);
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
    LUABIND_GET_PARAMETER(obj->getNumDim()+1,double,value);
    (*obj)(coords, obj->getNumDim()) = value;
    delete[] coords;
  }
  LUABIND_RETURN(MatrixDouble, obj);
}
//BIND_END

//BIND_METHOD MatrixDouble offset
{
  LUABIND_RETURN(int, obj->getOffset());
}
//BIND_END

//BIND_METHOD MatrixDouble raw_get
{
  int raw_pos;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  LUABIND_RETURN(double, (*obj)[raw_pos]);
}
//BIND_END

//BIND_METHOD MatrixDouble raw_set
{
  int raw_pos;
  int value;
  LUABIND_GET_PARAMETER(1, int, raw_pos);
  LUABIND_GET_PARAMETER(2, int, value);
  (*obj)[raw_pos] = static_cast<double>(value);
  LUABIND_RETURN(MatrixDouble, obj);
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
  obj->fill(static_cast<double>(value));
  LUABIND_RETURN(MatrixDouble, obj);
}
//BIND_END

//BIND_METHOD MatrixDouble dim
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

//BIND_METHOD MatrixDouble stride
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

//BIND_METHOD MatrixDouble slice
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
  LUABIND_TABLE_TO_VECTOR(1, int, coords, coords_len);
  LUABIND_TABLE_TO_VECTOR(2, int, sizes,  sizes_len);
  for (int i=0; i<coords_len; ++i) --coords[i];
  LUABIND_GET_OPTIONAL_PARAMETER(3, bool, clone, false);
  MatrixDouble *obj2 = new MatrixDouble(obj, coords, sizes, clone);
  LUABIND_RETURN(MatrixDouble, obj2);
  delete[] coords;
  delete[] sizes;
}
//BIND_END

//BIND_METHOD MatrixDouble select
{
  LUABIND_CHECK_ARGN(>=,2);
  LUABIND_CHECK_ARGN(<=,3);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(2, int);
  int dim, index;
  MatrixDouble *dest;
  LUABIND_GET_PARAMETER(1, int, dim);
  LUABIND_GET_PARAMETER(2, int, index);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixDouble, dest, 0);
  MatrixDouble *obj2 = obj->select(dim-1, index-1, dest);
  LUABIND_RETURN(MatrixDouble, obj2);
}
//BIND_END

//BIND_METHOD MatrixDouble clone
//DOC_BEGIN
// matrix *clone()
/// Devuelve un <em>clon</em> de la matriz.
//DOC_END
{
  LUABIND_CHECK_ARGN(>=, 0);
  LUABIND_CHECK_ARGN(<=, 1);
  int argn;
  argn = lua_gettop(L); // number of arguments
  MatrixDouble *obj2;
  if (argn == 0) obj2 = obj->clone();
  else {
    const char *major;
    LUABIND_GET_OPTIONAL_PARAMETER(1, string, major, "row_major");
    CBLAS_ORDER order=CblasRowMajor;
    if (strcmp(major, "col_major") == 0) order = CblasColMajor;
    else if (strcmp(major, "row_major") != 0)
      LUABIND_FERROR1("Incorrect major order char %s", major);
    obj2 = obj->clone(order);
  }
  LUABIND_RETURN(MatrixDouble,obj2);
}
//BIND_END

//BIND_METHOD MatrixDouble transpose
{
  LUABIND_RETURN(MatrixDouble, obj->transpose());
}
//BIND_END

//BIND_METHOD MatrixDouble diag
{
  LUABIND_CHECK_ARGN(==,1);
  double v;
  LUABIND_GET_PARAMETER(1, double, v);
  obj->diag(static_cast<double>(v));
  LUABIND_RETURN(MatrixDouble, obj);
}
//BIND_END

//BIND_METHOD MatrixDouble toTable
// Permite salvar una matriz en una tabla lua
// TODO: Tener en cuenta las dimensiones de la matriz
  {
    LUABIND_CHECK_ARGN(==, 0);
    lua_createtable(L, obj->size(), 0);
    int index = 1;
    for (MatrixDouble::iterator it(obj->begin()); it != obj->end(); ++it) {
      lua_pushint(L, *it);
      lua_rawseti(L, -2, index++);
    }
    LUABIND_RETURN_FROM_STACK(-1);
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

//BIND_METHOD MatrixDouble to_float
{
  bool col_major;
  LUABIND_GET_OPTIONAL_PARAMETER(1, bool, col_major, false);
  LUABIND_RETURN(MatrixFloat,
		 convertFromMatrixDoubleToMatrixFloat(obj, col_major));
}
//BIND_END

//BIND_METHOD MatrixDouble copy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 1);
  MatrixDouble *mat;
  LUABIND_GET_PARAMETER(1, MatrixDouble, mat);
  obj->copy(mat);
  LUABIND_RETURN(MatrixDouble, obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

