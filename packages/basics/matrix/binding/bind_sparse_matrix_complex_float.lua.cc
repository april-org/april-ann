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
#include "bind_complex.h"
#include "bind_matrix.h"
#include "bind_matrix_int32.h"
#include "bind_matrix_complex_float.h"
#include "bind_mathcore.h"
#include "cmath_overloads.h"
#include "luabindmacros.h" // for lua_pushComplexF and lua_pushint
#include "luabindutil.h"   // for lua_pushComplexF and lua_pushint

#include "matrix_ext.h"
using AprilMath::ComplexF;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Boolean;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilMath::MatrixExt::Misc;
using namespace AprilMath::MatrixExt::LAPACK;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(SparseMatrixComplexF);

namespace Basics {
  int sparseMatrixComplexFIteratorFunction(lua_State *L) {
    SparseMatrixComplexFIterator *obj = lua_toSparseMatrixComplexFIterator(L,1);
    if (obj->it == obj->m->end()) {
      lua_pushnil(L);
      return 1;
    }
    int c0=0,c1=0;
    obj->it.getCoords(c0,c1);
    lua_pushComplexF(L,*(obj->it));
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
#include "sparse_matrixComplexF.h"
#include "utilLua.h"

namespace Basics {

  class SparseMatrixComplexFIterator : public Referenced {
  public:
    SparseMatrixComplexF *m;
    SparseMatrixComplexF::iterator it;
    //
    SparseMatrixComplexFIterator(SparseMatrixComplexF *m) : m(m), it(m->begin()) {
      IncRef(m);
    }
    virtual ~SparseMatrixComplexFIterator() { DecRef(m); }
  };

#define MAKE_READ_SPARSE_MATRIX_LUA_METHOD(MatrixType, Type) do {       \
    MatrixType *obj = readSparse<Type>(L);               \
    if (obj == 0) {                                                     \
      luaL_error(L, "Error happens reading from file stream");          \
    }                                                                   \
    else {                                                              \
      lua_push##MatrixType(L, obj);                                     \
    }                                                                   \
  } while(false)
  
  template<typename T>
  SparseMatrix<T> *readSparse(lua_State *L) {
    AprilIO::StreamInterface *stream =
      lua_toAuxStreamInterface<AprilIO::StreamInterface>(L,1);
    AprilUtils::SharedPtr<AprilIO::StreamInterface> ptr(stream);
    if (stream == 0) {
      luaL_error(L, "Needs a stream as 1st argument");
      return 0;
    }
    AprilUtils::LuaTable options(L,2);
    return SparseMatrix<T>::read(ptr.get(), options);
  }

} // namespace Basics

using namespace Basics;

//BIND_END

//BIND_LUACLASSNAME SparseMatrixComplexFIterator matrixComplexF.sparse.iterator
//BIND_CPP_CLASS SparseMatrixComplexFIterator

//BIND_CONSTRUCTOR SparseMatrixComplexFIterator
{
  LUABIND_ERROR("Use iterate() method in matrixComplexF.sparse object");
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME SparseMatrixComplexF matrixComplexF.sparse
//BIND_CPP_CLASS SparseMatrixComplexF
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF SparseMatrixComplexF Serializable

//BIND_CONSTRUCTOR SparseMatrixComplexF
{
  if (lua_isMatrixComplexF(L,1)) {
    ComplexF near_zero;
    MatrixComplexF *m;
    LUABIND_GET_PARAMETER(1, MatrixComplexF, m);
    LUABIND_GET_OPTIONAL_PARAMETER(2, ComplexF, near_zero, NEAR_ZERO);
    obj = SparseMatrixComplexF::fromDenseMatrix(m, CSR_FORMAT, near_zero);
  }
  else {
    int rows,cols;
    ComplexFGPUMirroredMemoryBlock *values;
    Int32GPUMirroredMemoryBlock *indices;
    Int32GPUMirroredMemoryBlock *first_index;
    bool sort;
    LUABIND_GET_PARAMETER(1, int, rows);
    if (lua_isComplexFGPUMirroredMemoryBlock(L,2)) {
      cols = rows;
      rows = 1;
      LUABIND_GET_PARAMETER(2, ComplexFGPUMirroredMemoryBlock, values);
      LUABIND_GET_PARAMETER(3, Int32GPUMirroredMemoryBlock, indices);
      LUABIND_GET_OPTIONAL_PARAMETER(4, Int32GPUMirroredMemoryBlock,
				     first_index, 0);
      LUABIND_GET_OPTIONAL_PARAMETER(5, boolean, sort, false);
    }
    else {
      LUABIND_GET_PARAMETER(2, int, cols);
      LUABIND_GET_PARAMETER(3, ComplexFGPUMirroredMemoryBlock, values);
      LUABIND_GET_PARAMETER(4, Int32GPUMirroredMemoryBlock, indices);
      LUABIND_GET_OPTIONAL_PARAMETER(5, Int32GPUMirroredMemoryBlock,
				     first_index, 0);
      LUABIND_GET_OPTIONAL_PARAMETER(6, boolean, sort, false);
    }
    obj = new SparseMatrixComplexF(rows, cols,
				values, indices, first_index,
				CSR_FORMAT,
				sort);
  }
  LUABIND_RETURN(SparseMatrixComplexF,obj);
}
//BIND_END

//BIND_CLASS_METHOD SparseMatrixComplexF csc
{
  SparseMatrixComplexF *obj;
  if (lua_isMatrixComplexF(L,1)) {
    MatrixComplexF *m;
    ComplexF near_zero;
    LUABIND_GET_PARAMETER(1, MatrixComplexF, m);
    LUABIND_GET_OPTIONAL_PARAMETER(2, ComplexF, near_zero, NEAR_ZERO);
    obj = SparseMatrixComplexF::fromDenseMatrix(m, CSC_FORMAT, near_zero);
  }
  else {
    int rows,cols;
    ComplexFGPUMirroredMemoryBlock *values;
    Int32GPUMirroredMemoryBlock *indices;
    Int32GPUMirroredMemoryBlock *first_index;
    bool sort;
    LUABIND_GET_PARAMETER(1, int, rows);
    if (lua_isComplexFGPUMirroredMemoryBlock(L,2)) {
      cols = 1;
      LUABIND_GET_PARAMETER(2, ComplexFGPUMirroredMemoryBlock, values);
      LUABIND_GET_PARAMETER(3, Int32GPUMirroredMemoryBlock, indices);
      LUABIND_GET_OPTIONAL_PARAMETER(4, Int32GPUMirroredMemoryBlock,
				     first_index, 0);
      LUABIND_GET_OPTIONAL_PARAMETER(5, boolean, sort, false);
    }
    else {
      LUABIND_GET_PARAMETER(2, int, cols);
      LUABIND_GET_PARAMETER(3, ComplexFGPUMirroredMemoryBlock, values);
      LUABIND_GET_PARAMETER(4, Int32GPUMirroredMemoryBlock, indices);
      LUABIND_GET_OPTIONAL_PARAMETER(5, Int32GPUMirroredMemoryBlock,
				     first_index, 0);
      LUABIND_GET_OPTIONAL_PARAMETER(6, boolean, sort, false);
    }
    obj = new SparseMatrixComplexF(rows, cols,
				values, indices, first_index,
				CSC_FORMAT,
				sort);
  }
  LUABIND_RETURN(SparseMatrixComplexF,obj);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF size
{
  LUABIND_RETURN(int, obj->size());
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF non_zero_size
{
  LUABIND_RETURN(int, obj->nonZeroSize());
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF get_reference_string
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

//BIND_CLASS_METHOD SparseMatrixComplexF fromMMap
{
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 3);
  LUABIND_CHECK_PARAMETER(1, string);
  const char *filename;
  bool write, shared;
  LUABIND_GET_PARAMETER(1,string,filename);
  LUABIND_GET_OPTIONAL_PARAMETER(2,bool,write,true);
  LUABIND_GET_OPTIONAL_PARAMETER(3,bool,shared,true);
  AprilUtils::MMappedDataReader *mmapped_data;
  mmapped_data = new AprilUtils::MMappedDataReader(filename,write,shared);
  IncRef(mmapped_data);
  SparseMatrixComplexF *obj = SparseMatrixComplexF::fromMMappedDataReader(mmapped_data);
  DecRef(mmapped_data);
  LUABIND_RETURN(SparseMatrixComplexF,obj);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF toMMap
{
  LUABIND_CHECK_ARGN(==, 1);
  const char *filename;
  LUABIND_GET_PARAMETER(1, string, filename);
  AprilUtils::MMappedDataWriter *mmapped_data;
  mmapped_data = new AprilUtils::MMappedDataWriter(filename);
  IncRef(mmapped_data);
  obj->toMMappedDataWriter(mmapped_data);
  DecRef(mmapped_data);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF get
{
  int row,col;
  LUABIND_GET_PARAMETER(1, int, row);
  LUABIND_GET_PARAMETER(2, int, col);
  ComplexF v = (*obj)(row-1,col-1);
  LUABIND_RETURN(ComplexF, v);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF raw_get
{
  int idx;
  LUABIND_GET_PARAMETER(1, int, idx);
  if (idx < 1 || idx > obj->nonZeroSize())
    LUABIND_FERROR2("Index out-of-bounds [%d,%d]", 1, obj->nonZeroSize());
  SparseMatrixComplexF::const_iterator it(obj->iteratorAtRawIndex(idx-1));
  LUABIND_RETURN(ComplexF, *it);
  int c0=0,c1=0;
  it.getCoords(c0,c1);
  LUABIND_RETURN(ComplexF, *it);
  LUABIND_RETURN(int, c0+1);
  LUABIND_RETURN(int, c1+1);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF raw_set
{
  ComplexF value;
  int idx;
  LUABIND_GET_PARAMETER(1, int, idx);
  LUABIND_GET_PARAMETER(2, ComplexF, value);
  if (idx < 1 || idx > obj->nonZeroSize())
    LUABIND_FERROR2("Index out-of-bounds [%d,%d]", 1, obj->nonZeroSize());
  (*obj)[idx] = value;
  LUABIND_RETURN(SparseMatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF iterate
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,sparseMatrixComplexFIteratorFunction);
  LUABIND_RETURN(SparseMatrixComplexFIterator,new SparseMatrixComplexFIterator(obj));
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF fill
{
  LUABIND_CHECK_ARGN(==, 1);
  ComplexF value;
  if (lua_isSparseMatrixComplexF(L, 1)) {
    SparseMatrixComplexF *aux;
    LUABIND_GET_PARAMETER(1,SparseMatrixComplexF,aux);
    for (int i=0; i<aux->getNumDim(); ++i)
      if (aux->getDimSize(i) != 1)
	LUABIND_ERROR("Needs a ComplexF or a matrix with only one element\n");
    value = *(aux->begin());
  }
  else {
    LUABIND_CHECK_PARAMETER(1, ComplexF);
    LUABIND_GET_PARAMETER(1,ComplexF,value);
  }
  LUABIND_RETURN(SparseMatrixComplexF,
                 
                 matFill(obj,value));
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF zeros
//DOC_BEGIN
// void zeros(ComplexF value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  
  LUABIND_RETURN(SparseMatrixComplexF, 
                 matZeros(obj));
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF ones
//DOC_BEGIN
// void onex(ComplexF value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  LUABIND_RETURN(SparseMatrixComplexF, 
                 matOnes(obj));
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF get_use_cuda
{
  LUABIND_RETURN(bool, obj->getCudaFlag());
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF to_dense
{
  LUABIND_RETURN(MatrixComplexF, obj->toDense());
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF set_use_cuda
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, bool);
  bool v;
  LUABIND_GET_PARAMETER(1,bool, v);
  obj->setUseCuda(v);
  LUABIND_RETURN(SparseMatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF get_sparse_format
{
  if (obj->getSparseFormat() == CSR_FORMAT)
    LUABIND_RETURN(string, "csr");
  else LUABIND_RETURN(string, "csc");
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF dim
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

//BIND_METHOD SparseMatrixComplexF slice
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
  SparseMatrixComplexF *obj2 = new SparseMatrixComplexF(obj, coords, sizes, clone);
  LUABIND_RETURN(SparseMatrixComplexF, obj2);
  delete[] coords;
  delete[] sizes;
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF clone
{
  LUABIND_CHECK_ARGN(>=, 0);
  LUABIND_CHECK_ARGN(<=, 1);
  int argn;
  argn = lua_gettop(L); // number of arguments
  SparseMatrixComplexF *obj2;
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
  LUABIND_RETURN(SparseMatrixComplexF,obj2);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF transpose
{
  LUABIND_RETURN(SparseMatrixComplexF, obj->transpose());
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF isfinite
{
  LUABIND_CHECK_ARGN(==, 0);
  bool resul=true;
  for (SparseMatrixComplexF::iterator it(obj->begin()); resul && it!=obj->end(); ++it)
    //if (!isfinite(obj->data[i])) resul = 0;
    if ((*it) - (*it) != 0.0f) resul = false;
  LUABIND_RETURN(boolean,resul);
}
//BIND_END

//BIND_CLASS_METHOD SparseMatrixComplexF diag
{
  SparseMatrixComplexF *obj;
  int argn = lua_gettop(L); // number of arguments
  if (lua_isMatrixComplexF(L,1)) {
    MatrixComplexF *m;
    LUABIND_GET_PARAMETER(1, MatrixComplexF, m);
    const char *sparse;
    LUABIND_GET_OPTIONAL_PARAMETER(2, string, sparse, "csr");
    SPARSE_FORMAT format = CSR_FORMAT;
    if (strcmp(sparse, "csc") == 0) format = CSC_FORMAT;
    else if (strcmp(sparse, "csr") != 0)
      LUABIND_FERROR1("Incorrect sparse format string %s", sparse);    
    obj = SparseMatrixComplexF::diag(m,format);
  }
  else if (lua_isComplexFGPUMirroredMemoryBlock(L,1)) {
    ComplexFGPUMirroredMemoryBlock *block;
    LUABIND_GET_PARAMETER(1, ComplexFGPUMirroredMemoryBlock, block);
    const char *sparse;
    LUABIND_GET_OPTIONAL_PARAMETER(2, string, sparse, "csr");
    SPARSE_FORMAT format = CSR_FORMAT;
    if (strcmp(sparse, "csc") == 0) format = CSC_FORMAT;
    else if (strcmp(sparse, "csr") != 0)
      LUABIND_FERROR1("Incorrect sparse format string %s", sparse);    
    obj = SparseMatrixComplexF::diag(block,format);
  }
  else if (lua_istable(L,1)) {
    int N;
    LUABIND_TABLE_GETN(1, N);
    ComplexFGPUMirroredMemoryBlock *block = new ComplexFGPUMirroredMemoryBlock(N);
    const char *sparse;
    LUABIND_GET_OPTIONAL_PARAMETER(2, string, sparse, "csr");
    SPARSE_FORMAT format = CSR_FORMAT;
    if (strcmp(sparse, "csc") == 0) format = CSC_FORMAT;
    else if (strcmp(sparse, "csr") != 0)
      LUABIND_FERROR1("Incorrect sparse format string %s", sparse);
    ComplexF *mem = block->getPPALForWrite();
    lua_pushnil(L);
    int i=0;
    while(lua_next(L, 1) != 0) {
      mem[i++] = lua_toComplexF(L, -1); 
      lua_pop(L, 1);
    }
    obj = SparseMatrixComplexF::diag(block,format);
  }
  else {
    int N;
    ComplexF v;
    LUABIND_GET_PARAMETER(1, int, N);
    LUABIND_GET_PARAMETER(2, ComplexF, v);
    const char *sparse;
    LUABIND_GET_OPTIONAL_PARAMETER(3, string, sparse, "csr");
    SPARSE_FORMAT format = CSR_FORMAT;
    if (strcmp(sparse, "csc") == 0) format = CSC_FORMAT;
    else if (strcmp(sparse, "csr") != 0)
      LUABIND_FERROR1("Incorrect sparse format string %s", sparse);  
    obj = SparseMatrixComplexF::diag(N,v,format);
  }
  LUABIND_RETURN(SparseMatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF equals
{
  SparseMatrixComplexF *other;
  float epsilon;
  LUABIND_GET_PARAMETER(1, SparseMatrixComplexF, other);
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, epsilon, 1e-04f);
  LUABIND_RETURN(boolean, 
                 matEquals(obj, other, epsilon));
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF copy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 1);
  SparseMatrixComplexF *mat;
  LUABIND_GET_PARAMETER(1, SparseMatrixComplexF, mat);
  LUABIND_RETURN(SparseMatrixComplexF, 
                 matCopy(obj, mat));
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF scal
  {
    LUABIND_CHECK_ARGN(==, 1);
    ComplexF value;
    LUABIND_GET_PARAMETER(1, ComplexF, value);
    LUABIND_RETURN(SparseMatrixComplexF,
                   matScal(obj, value));
  }
//BIND_END
 
//BIND_METHOD SparseMatrixComplexF get_shared_count
{
  LUABIND_RETURN(uint, obj->getSharedCount());
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF reset_shared_count
{
  obj->resetSharedCount();
  LUABIND_RETURN(SparseMatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF add_to_shared_count
{
  unsigned int count;
  LUABIND_GET_PARAMETER(1,uint,count);
  obj->addToSharedCount(count);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF prune_subnormal_and_check_normal
{
  obj->pruneSubnormalAndCheckNormal();
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF as_vector
{
  LUABIND_RETURN(SparseMatrixComplexF, obj->asVector());
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF is_diagonal
{
  LUABIND_RETURN(bool, obj->isDiagonal());
}
//BIND_END


//// MATRIX SERIALIZATION ////

//BIND_CLASS_METHOD SparseMatrixComplexF read
{
  MAKE_READ_SPARSE_MATRIX_LUA_METHOD(SparseMatrixComplexF, ComplexF);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF num_dim
{
  LUABIND_RETURN(int, 2);
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF select
{
  LUABIND_ERROR("Not implemented method in sparse matrix instances");
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////

//BIND_METHOD SparseMatrixComplexF values
{
  LUABIND_RETURN(ComplexFGPUMirroredMemoryBlock, obj->getRawValuesAccess());
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF indices
{
  LUABIND_RETURN(Int32GPUMirroredMemoryBlock, obj->getRawIndicesAccess());
}
//BIND_END

//BIND_METHOD SparseMatrixComplexF first_index
{
  LUABIND_RETURN(Int32GPUMirroredMemoryBlock, obj->getRawFirstIndexAccess());
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME DOKBuilderSparseMatrixComplexF matrixComplexF.sparse.builders.dok
//BIND_CPP_CLASS DOKBuilderSparseMatrixComplexF

//BIND_CONSTRUCTOR DOKBuilderSparseMatrixComplexF
{
  LUABIND_RETURN(DOKBuilderSparseMatrixComplexF, new DOKBuilderSparseMatrixComplexF());
}
//BIND_END

//BIND_METHOD DOKBuilderSparseMatrixComplexF set
{
  unsigned int row, col;
  ComplexF value;
  LUABIND_GET_PARAMETER(1, uint, row);
  LUABIND_GET_PARAMETER(2, uint, col);
  LUABIND_GET_PARAMETER(3, ComplexF, value);
  if (row == 0 || col == 0) LUABIND_ERROR("Indices should be > 0");
  obj->insert(row-1, col-1, value);
  LUABIND_RETURN(DOKBuilderSparseMatrixComplexF, obj);
}
//BIND_END

//BIND_METHOD DOKBuilderSparseMatrixComplexF build
{
  unsigned int d0, d1;
  SPARSE_FORMAT format = CSR_FORMAT;
  const char *sparse;
  LUABIND_GET_OPTIONAL_PARAMETER(1, uint, d0, 0);
  LUABIND_GET_OPTIONAL_PARAMETER(2, uint, d1, 0);
  LUABIND_GET_OPTIONAL_PARAMETER(3, string, sparse, 0);
  if (sparse != 0) {
    if (strcmp(sparse, "csc") == 0) {
      format = CSC_FORMAT;
    }
    else if (strcmp(sparse, "csr") != 0) {
      LUABIND_FERROR1("Incorrect sparse format string %s", sparse);
    }
  }
  LUABIND_RETURN(SparseMatrixComplexF, obj->build(d0, d1, format));
}
//BIND_END

//////////////////////////////////////////////////////////////////////////////
