/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#include <cmath> // para isfinite
#include "april_assert.h"
#include "bind_april_io.h"
#include "bind_matrix.h"
#include "bind_matrix_int32.h"
#include "bind_mathcore.h"
#include "luabindmacros.h" // for lua_pushfloat and lua_pushint
#include "luabindutil.h"   // for lua_pushfloat and lua_pushint

namespace basics {
  int sparseMatrixFloatIteratorFunction(lua_State *L) {
    SparseMatrixFloatIterator *obj = lua_toSparseMatrixFloatIterator(L,1);
    if (obj->it == obj->m->end()) {
      lua_pushnil(L);
      return 1;
    }
    int c0=0,c1=0;
    obj->it.getCoords(c0,c1);
    lua_pushfloat(L,*(obj->it));
    lua_pushint(L,c0+1);
    lua_pushint(L,c1+1);
    ++(obj->it);
    return 3;
  }
}
//BIND_END

//BIND_HEADER_H
#include <cmath> // para isfinite

#include "bind_april_io.h"
#include "referenced.h"
#include "sparse_matrixFloat.h"
#include "utilLua.h"

namespace basics {

  class SparseMatrixFloatIterator : public Referenced {
  public:
    SparseMatrixFloat *m;
    SparseMatrixFloat::iterator it;
    //
    SparseMatrixFloatIterator(SparseMatrixFloat *m) : m(m), it(m->begin()) {
      IncRef(m);
    }
    virtual ~SparseMatrixFloatIterator() { DecRef(m); }
  };

#define MAKE_READ_SPARSE_MATRIX_LUA_METHOD(MatrixType, Type) do {       \
    MatrixType *obj = readSparseMatrixLuaMethod<Type>(L);               \
    if (obj == 0) {                                                     \
      luaL_error(L, "Error happens reading from file stream");          \
    }                                                                   \
    else {                                                              \
      lua_push##MatrixType(L, obj);                                     \
    }                                                                   \
  } while(false)
  
  template<typename T>
  SparseMatrix<T> *readSparseMatrixLuaMethod(lua_State *L) {
    AprilIO::StreamInterface *stream =
      lua_toAuxStreamInterface<AprilIO::StreamInterface>(L,1);
    april_utils::SharedPtr<AprilIO::StreamInterface> ptr(stream);
    if (stream == 0) {
      luaL_error(L, "Needs a stream as 1st argument");
      return 0;
    }
    april_utils::LuaTableOptions options(L,2);
    return SparseMatrix<T>::read(ptr.get(), &options);
  }

} // namespace basics

using namespace basics;

//BIND_END

//BIND_LUACLASSNAME SparseMatrixFloatIterator matrix.sparse.iterator
//BIND_CPP_CLASS SparseMatrixFloatIterator

//BIND_CONSTRUCTOR SparseMatrixFloatIterator
{
  LUABIND_ERROR("Use iterate() method in matrix.sparse object");
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME SparseMatrixFloat matrix.sparse
//BIND_CPP_CLASS SparseMatrixFloat
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF SparseMatrixFloat Serializable

//BIND_CONSTRUCTOR SparseMatrixFloat
{
  if (lua_isMatrixFloat(L,1)) {
    float near_zero;
    MatrixFloat *m;
    LUABIND_GET_PARAMETER(1, MatrixFloat, m);
    LUABIND_GET_OPTIONAL_PARAMETER(2, float, near_zero, NEAR_ZERO);
    obj = new SparseMatrixFloat(m, CSR_FORMAT, near_zero);
  }
  else {
    int rows,cols;
    FloatGPUMirroredMemoryBlock *values;
    Int32GPUMirroredMemoryBlock *indices;
    Int32GPUMirroredMemoryBlock *first_index;
    bool sort;
    LUABIND_GET_PARAMETER(1, int, rows);
    if (lua_isFloatGPUMirroredMemoryBlock(L,2)) {
      cols = rows;
      rows = 1;
      LUABIND_GET_PARAMETER(2, FloatGPUMirroredMemoryBlock, values);
      LUABIND_GET_PARAMETER(3, Int32GPUMirroredMemoryBlock, indices);
      LUABIND_GET_OPTIONAL_PARAMETER(4, Int32GPUMirroredMemoryBlock,
				     first_index, 0);
      LUABIND_GET_OPTIONAL_PARAMETER(5, boolean, sort, false);
    }
    else {
      LUABIND_GET_PARAMETER(2, int, cols);
      LUABIND_GET_PARAMETER(3, FloatGPUMirroredMemoryBlock, values);
      LUABIND_GET_PARAMETER(4, Int32GPUMirroredMemoryBlock, indices);
      LUABIND_GET_OPTIONAL_PARAMETER(5, Int32GPUMirroredMemoryBlock,
				     first_index, 0);
      LUABIND_GET_OPTIONAL_PARAMETER(6, boolean, sort, false);
    }
    obj = new SparseMatrixFloat(rows, cols,
				values, indices, first_index,
				CSR_FORMAT,
				sort);
  }
  LUABIND_RETURN(SparseMatrixFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD SparseMatrixFloat csc
{
  SparseMatrixFloat *obj;
  if (lua_isMatrixFloat(L,1)) {
    MatrixFloat *m;
    float near_zero;
    LUABIND_GET_PARAMETER(1, MatrixFloat, m);
    LUABIND_GET_OPTIONAL_PARAMETER(2, float, near_zero, NEAR_ZERO);
    obj = new SparseMatrixFloat(m, CSC_FORMAT, near_zero);
  }
  else {
    int rows,cols;
    FloatGPUMirroredMemoryBlock *values;
    Int32GPUMirroredMemoryBlock *indices;
    Int32GPUMirroredMemoryBlock *first_index;
    bool sort;
    LUABIND_GET_PARAMETER(1, int, rows);
    if (lua_isFloatGPUMirroredMemoryBlock(L,2)) {
      cols = 1;
      LUABIND_GET_PARAMETER(2, FloatGPUMirroredMemoryBlock, values);
      LUABIND_GET_PARAMETER(3, Int32GPUMirroredMemoryBlock, indices);
      LUABIND_GET_OPTIONAL_PARAMETER(4, Int32GPUMirroredMemoryBlock,
				     first_index, 0);
      LUABIND_GET_OPTIONAL_PARAMETER(5, boolean, sort, false);
    }
    else {
      LUABIND_GET_PARAMETER(2, int, cols);
      LUABIND_GET_PARAMETER(3, FloatGPUMirroredMemoryBlock, values);
      LUABIND_GET_PARAMETER(4, Int32GPUMirroredMemoryBlock, indices);
      LUABIND_GET_OPTIONAL_PARAMETER(5, Int32GPUMirroredMemoryBlock,
				     first_index, 0);
      LUABIND_GET_OPTIONAL_PARAMETER(6, boolean, sort, false);
    }
    obj = new SparseMatrixFloat(rows, cols,
				values, indices, first_index,
				CSC_FORMAT,
				sort);
  }
  LUABIND_RETURN(SparseMatrixFloat,obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat size
{
  LUABIND_RETURN(int, obj->size());
}
//BIND_END

//BIND_METHOD SparseMatrixFloat non_zero_size
{
  LUABIND_RETURN(int, obj->nonZeroSize());
}
//BIND_END

//BIND_METHOD SparseMatrixFloat get_reference_string
{
  char buff[128];
  sprintf(buff,"%p data= %p %p %p",
	  (void*)obj,
	  (void*)obj->getRawValuesAccess(),
	  (void*)obj->getRawIndicesAccess(),
	  (void*)obj->getRawFirstIndexAccess());
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_CLASS_METHOD SparseMatrixFloat fromMMap
{
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 3);
  LUABIND_CHECK_PARAMETER(1, string);
  const char *filename;
  bool write, shared;
  LUABIND_GET_PARAMETER(1,string,filename);
  LUABIND_GET_OPTIONAL_PARAMETER(2,bool,write,true);
  LUABIND_GET_OPTIONAL_PARAMETER(3,bool,shared,true);
  april_utils::MMappedDataReader *mmapped_data;
  mmapped_data = new april_utils::MMappedDataReader(filename,write,shared);
  IncRef(mmapped_data);
  SparseMatrixFloat *obj = SparseMatrixFloat::fromMMappedDataReader(mmapped_data);
  DecRef(mmapped_data);
  LUABIND_RETURN(SparseMatrixFloat,obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat toMMap
{
  LUABIND_CHECK_ARGN(==, 1);
  const char *filename;
  LUABIND_GET_PARAMETER(1, string, filename);
  april_utils::MMappedDataWriter *mmapped_data;
  mmapped_data = new april_utils::MMappedDataWriter(filename);
  IncRef(mmapped_data);
  obj->toMMappedDataWriter(mmapped_data);
  DecRef(mmapped_data);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat get
{
  int row,col;
  LUABIND_GET_PARAMETER(1, int, row);
  LUABIND_GET_PARAMETER(2, int, col);
  float v = (*obj)(row-1,col-1);
  LUABIND_RETURN(float, v);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat raw_get
{
  int idx;
  LUABIND_GET_PARAMETER(1, int, idx);
  if (idx < 1 || idx > obj->nonZeroSize())
    LUABIND_FERROR2("Index out-of-bounds [%d,%d]", 1, obj->nonZeroSize());
  SparseMatrixFloat::const_iterator it(obj->iteratorAtRawIndex(idx-1));
  LUABIND_RETURN(float, *it);
  int c0=0,c1=0;
  it.getCoords(c0,c1);
  LUABIND_RETURN(float, *it);
  LUABIND_RETURN(int, c0+1);
  LUABIND_RETURN(int, c1+1);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat raw_set
{
  float value;
  int idx;
  LUABIND_GET_PARAMETER(1, int, idx);
  LUABIND_GET_PARAMETER(2, float, value);
  if (idx < 1 || idx > obj->nonZeroSize())
    LUABIND_FERROR2("Index out-of-bounds [%d,%d]", 1, obj->nonZeroSize());
  (*obj)[idx] = value;
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat iterate
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,sparseMatrixFloatIteratorFunction);
  LUABIND_RETURN(SparseMatrixFloatIterator,new SparseMatrixFloatIterator(obj));
}
//BIND_END

//BIND_METHOD SparseMatrixFloat fill
{
  LUABIND_CHECK_ARGN(==, 1);
  float value;
  if (lua_isSparseMatrixFloat(L, 1)) {
    SparseMatrixFloat *aux;
    LUABIND_GET_PARAMETER(1,SparseMatrixFloat,aux);
    for (int i=0; i<aux->getNumDim(); ++i)
      if (aux->getDimSize(i) != 1)
	LUABIND_ERROR("Needs a float or a matrix with only one element\n");
    value = *(aux->begin());
  }
  else {
    LUABIND_CHECK_PARAMETER(1, float);
    LUABIND_GET_PARAMETER(1,float,value);
  }
  obj->fill(value);
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat zeros
//DOC_BEGIN
// void zeros(float value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  obj->zeros();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat ones
//DOC_BEGIN
// void onex(float value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  obj->ones();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat get_use_cuda
{
  LUABIND_RETURN(bool, obj->getCudaFlag());
}
//BIND_END

//BIND_METHOD SparseMatrixFloat to_dense
{
  const char *major;
  LUABIND_GET_OPTIONAL_PARAMETER(1, string, major, "row_major");
  CBLAS_ORDER order=CblasRowMajor;
  if (strcmp(major, "col_major") == 0) order = CblasColMajor;
  else if (strcmp(major, "row_major") != 0)
    LUABIND_FERROR1("Incorrect major order string %s", major);
  LUABIND_RETURN(MatrixFloat, obj->toDense(order));
}
//BIND_END

//BIND_METHOD SparseMatrixFloat set_use_cuda
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, bool);
  bool v;
  LUABIND_GET_PARAMETER(1,bool, v);
  obj->setUseCuda(v);
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat get_sparse_format
{
  if (obj->getSparseFormat() == CSR_FORMAT)
    LUABIND_RETURN(string, "csr");
  else LUABIND_RETURN(string, "csc");
}
//BIND_END

//BIND_METHOD SparseMatrixFloat dim
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

//BIND_METHOD SparseMatrixFloat slice
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
  LUABIND_GET_OPTIONAL_PARAMETER(3, bool, clone, false);
  for (int i=0; i<sizes_len; ++i)
    if (coords[i] < 0 || sizes[i] < 1 ||
	sizes[i]+coords[i] > obj->getDimSize(i))
      LUABIND_FERROR1("Incorrect size or coord at position %d\n", i+1);
  SparseMatrixFloat *obj2 = new SparseMatrixFloat(obj, coords, sizes, clone);
  LUABIND_RETURN(SparseMatrixFloat, obj2);
  delete[] coords;
  delete[] sizes;
}
//BIND_END

//BIND_METHOD SparseMatrixFloat clone
{
  LUABIND_CHECK_ARGN(>=, 0);
  LUABIND_CHECK_ARGN(<=, 1);
  int argn;
  argn = lua_gettop(L); // number of arguments
  SparseMatrixFloat *obj2;
  if (argn == 0) obj2 = obj->clone();
  else {
    const char *sparse;
    LUABIND_GET_PARAMETER(1, string, sparse);
    SPARSE_FORMAT format = CSC_FORMAT;
    if (strcmp(sparse, "csr") == 0) format = CSR_FORMAT;
    else if (strcmp(sparse, "csc") != 0)
      LUABIND_FERROR1("Incorrect sparse format string %s", sparse);
    obj2 = obj->clone(format);
  }
  LUABIND_RETURN(SparseMatrixFloat,obj2);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat transpose
{
  LUABIND_RETURN(SparseMatrixFloat, obj->transpose());
}
//BIND_END

//BIND_METHOD SparseMatrixFloat isfinite
{
  LUABIND_CHECK_ARGN(==, 0);
  bool resul=true;
  for (SparseMatrixFloat::iterator it(obj->begin()); resul && it!=obj->end(); ++it)
    //if (!isfinite(obj->data[i])) resul = 0;
    if ((*it) - (*it) != 0.0f) resul = false;
  LUABIND_RETURN(boolean,resul);
}
//BIND_END

//BIND_CLASS_METHOD SparseMatrixFloat diag
{
  SparseMatrixFloat *obj;
  int argn = lua_gettop(L); // number of arguments
  if (lua_isMatrixFloat(L,1)) {
    MatrixFloat *m;
    LUABIND_GET_PARAMETER(1, MatrixFloat, m);
    const char *sparse;
    LUABIND_GET_OPTIONAL_PARAMETER(2, string, sparse, "csr");
    SPARSE_FORMAT format = CSR_FORMAT;
    if (strcmp(sparse, "csc") == 0) format = CSC_FORMAT;
    else if (strcmp(sparse, "csr") != 0)
      LUABIND_FERROR1("Incorrect sparse format string %s", sparse);    
    obj = SparseMatrixFloat::diag(m,format);
  }
  else if (lua_isFloatGPUMirroredMemoryBlock(L,1)) {
    FloatGPUMirroredMemoryBlock *block;
    LUABIND_GET_PARAMETER(1, FloatGPUMirroredMemoryBlock, block);
    const char *sparse;
    LUABIND_GET_OPTIONAL_PARAMETER(2, string, sparse, "csr");
    SPARSE_FORMAT format = CSR_FORMAT;
    if (strcmp(sparse, "csc") == 0) format = CSC_FORMAT;
    else if (strcmp(sparse, "csr") != 0)
      LUABIND_FERROR1("Incorrect sparse format string %s", sparse);    
    obj = SparseMatrixFloat::diag(block,format);
  }
  else if (lua_istable(L,1)) {
    int i=1;
    int len;
    LUABIND_TABLE_GETN(1, len);
    FloatGPUMirroredMemoryBlock *block;
    block = new FloatGPUMirroredMemoryBlock(static_cast<unsigned int>(len));
    float *data = block->getPPALForWrite();
    for (int i=1; i<=len; ++i) {
      lua_rawgeti(L,1,i);
      if (!lua_isnumber(L, -1))
	LUABIND_FERROR1("The given table has a no number value at position %d, "
			"the table could be smaller than matrix size", i);
      data[i-1] = (float)luaL_checknumber(L, -1);
      lua_pop(L,1);
    }
    const char *sparse;
    LUABIND_GET_OPTIONAL_PARAMETER(2, string, sparse, "csr");
    SPARSE_FORMAT format = CSR_FORMAT;
    if (strcmp(sparse, "csc") == 0) format = CSC_FORMAT;
    else if (strcmp(sparse, "csr") != 0)
      LUABIND_FERROR1("Incorrect sparse format string %s", sparse);    
    obj = SparseMatrixFloat::diag(block,format);
  }
  else {
    int N;
    float v;
    LUABIND_GET_PARAMETER(1, int, N);
    LUABIND_GET_PARAMETER(2, float, v);
    const char *sparse;
    LUABIND_GET_OPTIONAL_PARAMETER(3, string, sparse, "csr");
    SPARSE_FORMAT format = CSR_FORMAT;
    if (strcmp(sparse, "csc") == 0) format = CSC_FORMAT;
    else if (strcmp(sparse, "csr") != 0)
      LUABIND_FERROR1("Incorrect sparse format string %s", sparse);  
    obj = SparseMatrixFloat::diag(N,v,format);
  }
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat min
  {
    LUABIND_CHECK_ARGN(>=,0);
    LUABIND_CHECK_ARGN(<=,3);
    int argn = lua_gettop(L);
    if (argn > 0) {
      int dim;
      MatrixFloat *dest;
      MatrixInt32 *argmin;
      LUABIND_GET_PARAMETER(1, int, dim);
      LUABIND_GET_OPTIONAL_PARAMETER(2, MatrixFloat, dest, 0);
      LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixInt32, argmin, 0);
      int *aux = 0;
      if (argmin == 0) {
	aux = new int[obj->getNumDim()];
	for (int i=0; i<obj->getNumDim(); ++i) aux[i] = obj->getDimSize(i);
	aux[dim-1] = 1;
	argmin = new MatrixInt32(obj->getNumDim(), aux);
      }
      IncRef(argmin);
      if (dim < 1 || dim > obj->getNumDim())
	LUABIND_FERROR2("Incorrect dimension, found %d, expect in [1,%d]",
			dim, obj->getNumDim());
      LUABIND_RETURN(MatrixFloat, obj->min(dim-1, dest, argmin));
      LUABIND_RETURN(MatrixInt32, argmin);
      DecRef(argmin);
      delete[] aux;
    }
    else {
      int c0, c1;
      LUABIND_RETURN(float, obj->min(c0, c1));
      LUABIND_RETURN(int, c0+1);
      LUABIND_RETURN(int, c1+1);
    }
  }
//BIND_END

//BIND_METHOD SparseMatrixFloat max
  {
    LUABIND_CHECK_ARGN(>=,0);
    LUABIND_CHECK_ARGN(<=,3);
    int argn = lua_gettop(L);
    if (argn > 0) {
      int dim;
      MatrixFloat *dest;
      MatrixInt32 *argmax;
      LUABIND_GET_PARAMETER(1, int, dim);
      LUABIND_GET_OPTIONAL_PARAMETER(2, MatrixFloat, dest, 0);
      LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixInt32, argmax, 0);
      int *aux = 0;
      if (argmax == 0) {
	aux = new int[obj->getNumDim()];
	for (int i=0; i<obj->getNumDim(); ++i) aux[i] = obj->getDimSize(i);
	aux[dim-1] = 1;
	argmax = new MatrixInt32(obj->getNumDim(), aux);
      }
      IncRef(argmax);
      if (dim < 1 || dim > obj->getNumDim())
	LUABIND_FERROR2("Incorrect dimension, found %d, expect in [1,%d]",
			dim, obj->getNumDim());
      LUABIND_RETURN(MatrixFloat, obj->max(dim-1, dest, argmax));
      LUABIND_RETURN(MatrixInt32, argmax);
      DecRef(argmax);
      delete[] aux;
    }
    else {
      int c0, c1;
      LUABIND_RETURN(float, obj->max(c0, c1));
      LUABIND_RETURN(int, c0+1);
      LUABIND_RETURN(int, c1+1);
    }
  }
//BIND_END

//BIND_METHOD SparseMatrixFloat equals
{
  SparseMatrixFloat *other;
  float epsilon;
  LUABIND_GET_PARAMETER(1, SparseMatrixFloat, other);
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, epsilon, 1e-04f);
  LUABIND_RETURN(boolean, obj->equals(other, epsilon));
}
//BIND_END

//BIND_METHOD SparseMatrixFloat sqrt
{
  obj->sqrt();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat pow
{
  float value;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, float, value);
  obj->pow(value);
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat tan
{
  obj->tan();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat tanh
{
  obj->tanh();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat atan
{
  obj->atan();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat atanh
{
  obj->atanh();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat sin
{
  obj->sin();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat sinh
{
  obj->sinh();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat asin
{
  obj->asin();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat asinh
{
  obj->asinh();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat abs
{
  obj->abs();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat sign
{
  obj->sign();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat sum
{
  LUABIND_CHECK_ARGN(>=, 0);
  LUABIND_CHECK_ARGN(<=, 2);
  int argn = lua_gettop(L); // number of arguments
  if (argn >= 1) {
    int dim;
    MatrixFloat *dest;
    LUABIND_GET_PARAMETER(1, int, dim);
    LUABIND_GET_OPTIONAL_PARAMETER(2, MatrixFloat, dest, 0);
    if (dim < 1 || dim > obj->getNumDim())
      LUABIND_FERROR2("Incorrect dimension, found %d, expect in [1,%d]",
		      dim, obj->getNumDim());
    MatrixFloat *result = obj->sum(dim-1, dest);
    LUABIND_RETURN(MatrixFloat, result);
  }
  else LUABIND_RETURN(float, obj->sum());
}
//BIND_END

//BIND_METHOD SparseMatrixFloat copy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 1);
  SparseMatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, SparseMatrixFloat, mat);
  obj->copy(mat);
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat scal
  {
    LUABIND_CHECK_ARGN(==, 1);
    float value;
    LUABIND_GET_PARAMETER(1, float, value);
    obj->scal(value);
    LUABIND_RETURN(SparseMatrixFloat, obj);
  }
//BIND_END

//BIND_METHOD SparseMatrixFloat div
  {
    LUABIND_CHECK_ARGN(==, 1);
    float value;
    LUABIND_GET_PARAMETER(1, float, value);
    obj->div(value);
    LUABIND_RETURN(SparseMatrixFloat, obj);
  }
//BIND_END
 
//BIND_METHOD SparseMatrixFloat norm2
  {
    LUABIND_RETURN(float, obj->norm2());
  }
//BIND_END

//BIND_METHOD SparseMatrixFloat get_shared_count
{
  LUABIND_RETURN(uint, obj->getSharedCount());
}
//BIND_END

//BIND_METHOD SparseMatrixFloat reset_shared_count
{
  obj->resetSharedCount();
  LUABIND_RETURN(SparseMatrixFloat, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat add_to_shared_count
{
  unsigned int count;
  LUABIND_GET_PARAMETER(1,uint,count);
  obj->addToSharedCount(count);
}
//BIND_END

//BIND_METHOD SparseMatrixFloat prune_subnormal_and_check_normal
{
  obj->pruneSubnormalAndCheckNormal();
}
//BIND_END

//BIND_METHOD SparseMatrixFloat as_vector
{
  LUABIND_RETURN(SparseMatrixFloat, obj->asVector());
}
//BIND_END

//BIND_METHOD SparseMatrixFloat is_diagonal
{
  LUABIND_RETURN(bool, obj->isDiagonal());
}
//BIND_END


//// MATRIX SERIALIZATION ////

//BIND_CLASS_METHOD SparseMatrixFloat read
{
  MAKE_READ_SPARSE_MATRIX_LUA_METHOD(SparseMatrixFloat, float);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END
