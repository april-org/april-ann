/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
 * Copyright 2012, Salvador España-Boquera
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

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "error_print.h"
#include "ignore_result.h"

template<typename T>
const unsigned int SparseMatrix<T>::MATRIX_BINARY_VERSION = 0x00000001;

template<typename T>
int SparseMatrix<T>::searchIndexOf(const int c0, const int c1) const {
  int idx;
  switch(sparse_format) {
  case CSC_FORMAT:
    if (c1 >= matrixSize[1]) idx = non_zero_size+1;
    else idx = doSearchCSCSparseIndexOf(indices, first_index, c0, c1, use_cuda);
    break;
  case CSR_FORMAT:
    if (c0 >= matrixSize[0]) idx = non_zero_size+1;
    else idx = doSearchCSRSparseIndexOf(indices, first_index, c0, c1, use_cuda);
    break;
  default:
    ERROR_EXIT1(128, "Unrecognized format %d\n", sparse_format);
  }
  if (idx == -1)
    ERROR_EXIT2(128, "Impossible to found index of (%d,%d)\n", c0, c1);
  return idx;
}

template<typename T>
int SparseMatrix<T>::searchIndexOfFirst(const int c0, const int c1) const {
  int idx;
  switch(sparse_format) {
  case CSC_FORMAT:
    if (c1 >= matrixSize[1]) idx = non_zero_size+1;
    else idx = doSearchCSCSparseIndexOfFirst(indices, first_index, c0, c1,
					     use_cuda);
    break;
  case CSR_FORMAT:
    if (c0 >= matrixSize[0]) idx = non_zero_size+1;
    else idx = doSearchCSRSparseIndexOfFirst(indices, first_index, c0, c1,
					     use_cuda);
    break;
  default:
    ERROR_EXIT1(128, "Unrecognized format %d\n", sparse_format);
  }
  return idx;
}

template <typename T>
void SparseMatrix<T>::initialize(int d0, int d1) {
  total_size    = d0*d1;
  matrixSize[0] = d0;
  matrixSize[1] = d1;
  non_zero_size = 0;
}

/// Allocation of memory for data pointer. It is Referenced for sharing.
template <typename T>
void SparseMatrix<T>::allocate_memory(int size) {
  unsigned int sz = static_cast<unsigned int>(size);
  values  = new GPUMirroredMemoryBlock<T>(sz);
  indices = new IntGPUMirroredMemoryBlock(INITIAL_NON_ZERO_SIZE);
  first_index = new IntGPUMirroredMemoryBlock(INITIAL_NON_ZERO_SIZE+1);
  ptrB = new IntGPUMirroredMemoryBlock(first_index->getPPALForRead(),
				       INITIAL_NON_ZERO_SIZE);
  ptrE = new IntGPUMirroredMemoryBlock(first_index->getPPALForRead()+1,
				       INITIAL_NON_ZERO_SIZE);
  IncRef(values);
  IncRef(indices);
  IncRef(first_index);
  IncRef(ptrB);
  IncRef(ptrE);
}

/// Release of the memory allocated for data pointer.
template <typename T>
void SparseMatrix<T>::release_memory() {
  DecRef(values);
  DecRef(indices);
  DecRef(first_index);
  DecRef(ptrB);
  DecRef(ptrE);
}

/// Default constructor
template <typename T>
SparseMatrix<T>::SparseMatrix(const int d0, const int d1,
			      const SPARSE_FORMAT sparse_format) :
  Referenced(), shared_count(0), mmapped_data(0),
  sparse_format(sparse_format), use_cuda(false),
  end_iterator(), end_const_iterator() {
  initialize(d0, d1);
  allocate_memory(total_size);
}

/// Constructor for sub-matrix building
template <typename T>
SparseMatrix<T>::SparseMatrix(const SparseMatrix<T> *other,
			      const int coord0, const int coord1,
			      const int size0, const int size1) :
  Referenced(),
  shared_count(0), mmapped_data(0),
  sparse_format(other->sparse_format),
  use_cuda(other->use_cuda),
  end_iterator(), end_const_iterator(), end_best_span_iterator() {
  if (size0 + coord0 > other->matrixSize[0])
    ERROR_EXIT3(128, "Size+coordinates are out of dimension size: %d+%d>%d\n",
		size0, coord0, other->matrixSize[0]);
  if (size1 + coord1 > other->matrixSize[1])
    ERROR_EXIT3(128, "Size+coordinates are out of dimension size: %d+%d>%d\n",
		size1, coord1, other->matrixSize[1]);
  initialize(size0, size1);
  allocate_memory(total_size);
  switch(sparse_format) {
  case CSC_FORMAT:
    for (int c1=0; c1<size1; ++c1) {
      const_iterator it(other->iteratorAtFirst(coord0, c1));
      const_iterator end_it(other->iteratorAtFirst(coord0 + size0, c1));
      pushBack(it,end_it,coord0,coord1);
    }
    break;
  case CSR_FORMAT:
    for (int c0=0; c0<size0; ++c0) {
      const_iterator it(other->iteratorAtFirst(c0, coord1));
      const_iterator end_it(other->iteratorAtFirst(c0, coord1 + size1));
      pushBack(it,end_it,coord0,coord1);
    }
    break;
  default:
    ERROR_EXIT1(128, "Unrecognized format %d\n", sparse_format);
  }
}

template <typename T>
SparseMatrix<T> *SparseMatrix<T>::fromMMappedDataReader(april_utils::MMappedDataReader
							*mmapped_data) {
  SparseMatrix<T> *obj = new SparseMatrix();
  //
  obj->values  = GPUMirroredMemoryBlock<T>::fromMMappedDataReader(mmapped_data);
  obj->indices = IntGPUMirroredMemoryBlock::fromMMappedDataReader(mmapped_data);
  obj->first_index = IntGPUMirroredMemoryBlock::fromMMappedDataReader(mmapped_data);
  IncRef(values);
  IncRef(indices);
  IncRef(first_index);
  //
  ptrB = new IntGPUMirroredMemoryBlock(first_index->getPPALForRead(),
				       INITIAL_NON_ZERO_SIZE);
  ptrE = new IntGPUMirroredMemoryBlock(first_index->getPPALForRead()+1,
				       INITIAL_NON_ZERO_SIZE);
  IncRef(ptrB);
  IncRef(ptrE);
  //
  unsigned int binary_version = *(mmapped_data->get<unsigned int>());
  if (binary_version != MATRIX_BINARY_VERSION)
    ERROR_EXIT1(128,
		"Incorrect binary matrix version from commit number %d\n",
		mmapped_data->getCommitNumber());
  int *aux_size = mmapped_data->get<int>(2);
  obj->matrixSize[0] = aux_size[0];
  obj->matrixSize[1] = aux_size[1];
  obj->total_size    = *(mmapped_data->get<int>());
  obj->non_zero_size = *(mmapped_data->get<int>());
  obj->sparse_format = *(mmapped_data->get<SPARSE_FORMAT>());
  // NON MAPPED DATA
  obj->use_cuda      = false;
  obj->shared_count  = 0;
  // THE MMAP POINTER
  obj->mmapped_data  = mmapped_data;
  IncRef(obj->mmapped_data);
  //
  return obj;
}

template <typename T>
void SparseMatrix<T>::toMMappedDataWriter(april_utils::MMappedDataWriter
					  *mmapped_data) const {
  values->toMMappedDataWriter(mmapped_data);
  indices->toMMappedDataWriter(mmapped_data);
  first_index->toMMappedDataWriter(mmapped_data);
  mmapped_data->put(&MATRIX_BINARY_VERSION);
  mmapped_data->put(matrixSize, 2);
  mmapped_data->put(&total_size);
  mmapped_data->put(&non_zero_size);
  mmapped_data->put(&sparse_format);
}

template <typename T>
SparseMatrix<T>::~SparseMatrix() {
  release_memory();
  if (mmapped_data != 0) DecRef(mmapped_data);
}

template <typename T>
SparseMatrix<T> *SparseMatrix<T>::rewrap(const int *new_dims, int len) {
  if (!getIsContiguous())
    ERROR_EXIT(128, "Impossible to re-wrap non contiguous matrix, "
	       "clone it first\n");
  bool equal = true;
  int new_size = 1;
  for (int i=0; i<len; ++i) {
    if (i>=numDim || new_dims[i] != matrixSize[i]) equal=false;
    new_size *= new_dims[i];
  }
  if (len==numDim && equal) return this;
  if (new_size != size())
    ERROR_EXIT2(128, "Incorrect size, expected %d, and found %d\n",
		size(), new_size);
  SparseMatrix<T> *obj = new SparseMatrix<T>(len, new_dims, major_order, data, offset);
#ifdef USE_CUDA
  obj->setUseCuda(use_cuda);
#endif
  return obj;
}

template<typename T>
SparseMatrix<T> *SparseMatrix<T>::transpose() {
  SparseMatrix<T> *result;
  if (this->numDim > 1) {
    result = this->shallow_copy();
    result->transposed = !result->transposed;
    for (int i=0,j=numDim-1; i<numDim; ++i,--j) {
      result->stride[j]     = this->stride[i];
      result->matrixSize[j] = this->matrixSize[i];
    }
  }
  else result = this;
  return result;
}

template <typename T>
SparseMatrix<T>* SparseMatrix<T>::cloneOnlyDims() const {
  SparseMatrix<T> *obj = new SparseMatrix<T>(numDim, matrixSize, major_order);
#ifdef USE_CUDA
  obj->setUseCuda(use_cuda);
#endif
  return obj;
}

template<typename T>
SparseMatrix<T> *SparseMatrix<T>::clone(CBLAS_ORDER major_order) {
  SparseMatrix<T> *resul;
  if (this->major_order != major_order) {
    resul = new SparseMatrix<T>(numDim, matrixSize, major_order);
#ifdef USE_CUDA
    resul->setUseCuda(use_cuda);
#endif
    iterator resul_it(resul->begin());
    const_iterator this_it(begin());
    while(resul_it != resul->end()) {
      *resul_it = *this_it;
      ++resul_it;
      ++this_it;
    }
  }
  else resul = this->clone();
  return resul;
}

template <typename T>
SparseMatrix<T>* SparseMatrix<T>::clone() {
  return new SparseMatrix<T>(this,true);
}

template <typename T>
SparseMatrix<T>* SparseMatrix<T>::shallow_copy() {
  return new SparseMatrix<T>(this,false);
}

template <typename T>
T& SparseMatrix<T>::operator[] (int i) {
  return data->get(static_cast<unsigned int>(i));
}

template <typename T>
const T& SparseMatrix<T>::operator[] (int i) const {
  return data->get(static_cast<unsigned int>(i));
}

template <typename T>
T& SparseMatrix<T>::operator() (int i) {
  april_assert(numDim == 1);
  int raw_pos = computeRawPos(&i);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
T& SparseMatrix<T>::operator() (int row, int col) {
  april_assert(numDim == 2);
  int pos[2]={row,col};
  int raw_pos = computeRawPos(pos);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
T& SparseMatrix<T>::operator() (int coord0, int coord0, int coord1, ...) {
  april_assert(numDim >= 3);
  int *aux_coords = new int[numDim];
  aux_coords[0] = coord0;
  aux_coords[1] = coord0;
  aux_coords[2] = coord1;
  va_list ap;
  va_start(ap, coord1);
  for(int i=3; i<numDim; i++) {
    int coordn = va_arg(ap, int);
    aux_coords[i] = coordn;
  }
  va_end(ap);
  int raw_pos = computeRawPos(aux_coords);
  delete[] aux_coords;
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
T& SparseMatrix<T>::operator() (int *coords, int sz) {
  UNUSED_VARIABLE(sz);
  april_assert(numDim == sz);
  int raw_pos = computeRawPos(coords);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
const T& SparseMatrix<T>::operator() (int i) const {
  april_assert(numDim == 1);
  int raw_pos = computeRawPos(&i);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
const T& SparseMatrix<T>::operator() (int row, int col) const {
  april_assert(numDim == 2);
  int pos[2]={row,col};
  int raw_pos = computeRawPos(pos);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
const T& SparseMatrix<T>::operator() (int coord0, int coord0, int coord1, ...) const {
  april_assert(numDim >= 3);
  int *aux_coords = new int[numDim];
  aux_coords[0] = coord0;
  aux_coords[1] = coord0;
  aux_coords[2] = coord1;
  va_list ap;
  va_start(ap, coord1);
  for(int i=3; i<numDim; i++) {
    int coordn = va_arg(ap, int);
    aux_coords[i] = coordn;
  }
  va_end(ap);
  int raw_pos = computeRawPos(aux_coords);
  delete[] aux_coords;
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
const T& SparseMatrix<T>::operator() (int *coords, int sz) const {
  UNUSED_VARIABLE(sz);
  april_assert(numDim == sz);
  int raw_pos = computeRawPos(coords);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
bool SparseMatrix<T>::getCol(int col, T* vec, int vecsize) {
  // If it is not a 2D matrix, error
  if (numDim != 2) return false;
  // If the column is out of range, error
  if ((col < 0) || (col >= matrixSize[1])) return false;
  // If the array length is different to the size of the matrix columns, error
  if (vecsize != matrixSize[0]) return false;
  const T *d = data->getPPALForRead();
  for (int row = 0; row < matrixSize[0]; row++) {
    int coords[2] = { row, col };
    vec[row] = d[computeRawPos(coords)];
  }
  return true;
}

template <typename T>
bool SparseMatrix<T>::putCol(int col, T* vec, int vecsize) {
  // If it is not a 2D matrix, error
  if (numDim != 2) return false;
  // If the column is out of range, error
  if ((col < 0) || (col >= matrixSize[1])) return false;
  // If the array length is different to the size of the matrix columns, error
  if (vecsize != matrixSize[0]) return false;
  T *d = data->getPPALForWrite();
  for (int row = 0; row < matrixSize[0]; row++) {
    int coords[2] = { row, col };
    d[computeRawPos(coords)] = vec[row];
  }
  return true;
}

template <typename T>
bool SparseMatrix<T>::putSubCol(int col, int first_row, T* vec, int vecsize) {
  // If it is not a 2D matrix, error
  if (numDim != 2) return false;
  // If the column is out of range, error
  if ((col < 0) || (col >= matrixSize[1])) return false;
  // If the first row is out of range, error
  if ((first_row < 0) || (first_row >= matrixSize[0])) return false;
  // If the array is out of range, error
  if ((first_row < 0) || (first_row+vecsize > matrixSize[0])) return false;
  T *d = data->getPPALForWrite();
  for (int row = first_row; row < first_row+vecsize; row++) {
    int coords[2] = { row, col };
    d[computeRawPos(coords)] = vec[row];
  }
  return true;
}

template <typename T>
template <typename O>
bool SparseMatrix<T>::sameDim(const SparseMatrix<O> *other) const {
  return sameDim(other->getDimPtr(), other->getNumDim());
}

template <typename T>
bool SparseMatrix<T>::sameDim(const int *dims, const int len) const {
  if (numDim != len) return false;
  switch(numDim) {
  default:
    for (int i=0; i<numDim; ++i)
      if (matrixSize[i] != dims[i]) return false;
    break;
#define CASE(i,j) case i: if (matrixSize[j] != dims[j]) return false
    CASE(6,5);
    CASE(5,4);
    CASE(4,3);
    CASE(3,2);
    CASE(2,1);
    CASE(1,0);
    break;
#undef CASE
  }
  return true;
}

template<typename T>
SparseMatrix<T> *SparseMatrix<T>::select(int dim, int index, SparseMatrix<T> *dest) {
  if (numDim == 1)
    ERROR_EXIT(128, "Not possible to execute select for numDim=1\n");
  if (dim >= numDim)
    ERROR_EXIT(128, "Select for a dimension which doesn't exists\n");
  if (index >= matrixSize[dim])
    ERROR_EXIT(128, "Select for an index out of the matrix\n");
  SparseMatrix<T> *result;
  if (dest == 0) {
    result = new SparseMatrix();
    int d = numDim - 1;
    // Data initialization
    result->transposed   = this->transposed;
    result->use_cuda     = use_cuda;
    result->numDim       = d;
    result->matrixSize   = new int[d];
    result->stride       = new int[d];
    result->major_order  = major_order;
    result->offset       = offset + index*stride[dim]; // the select implies an offset
    result->last_raw_pos = result->offset;
    result->data         = data;
    result->mmapped_data = 0;
    IncRef(data);
    for(int i=0; i<dim; ++i) {
      result->stride[i]      = stride[i];
      result->matrixSize[i]  = matrixSize[i];
      result->last_raw_pos  += (matrixSize[i]-1)*stride[i];
    }
    for(int i=dim+1; i<numDim; ++i) {
      result->stride[i-1]      = stride[i];
      result->matrixSize[i-1]  = matrixSize[i];
      result->last_raw_pos    += (matrixSize[i]-1)*stride[i];
    }
    result->total_size = total_size/matrixSize[dim];
  }
  else {
    //
    april_assert(dest->total_size == total_size/matrixSize[dim]);
    april_assert(dest->numDim == numDim-1);
    //
    int dest_offset = offset + index*stride[dim];
    int dest_last_raw_pos = dest_offset;
    for(int i=0; i<dim; ++i)
      dest_last_raw_pos += (matrixSize[i]-1)*stride[i];
    for(int i=dim+1; i<numDim; ++i)
      dest_last_raw_pos += (matrixSize[i]-1)*stride[i];
    dest->changeSubSparseMatrixData(dest_offset, dest_last_raw_pos);
    //
    result = dest;
  }
  return result;
}

/***** COORDINATES METHODS *****/

template <typename T>
bool SparseMatrix<T>::nextCoordVectorRowOrder(int *coords, int &raw_pos) const {
  return nextCoordVectorRowOrder(coords, raw_pos, matrixSize, stride, numDim,
				 last_raw_pos);
}

template <typename T>
bool SparseMatrix<T>::nextCoordVectorColOrder(int *coords, int &raw_pos) const {
  return nextCoordVectorColOrder(coords, raw_pos, matrixSize, stride, numDim,
				 last_raw_pos);
}

template <typename T>
bool SparseMatrix<T>::nextCoordVectorRowOrder(int *coords, int &raw_pos,
					      const int *sizes,
					      const int *strides,
					      const int numDim,
					      const int last_raw_pos) {
  bool ret = true;
  switch(numDim) {
  case 1:
    coords[0] = (coords[0]+1) % sizes[0];
    if (coords[0] == 0) {
      raw_pos = last_raw_pos + 1;
      ret = false;
    }
    else raw_pos += strides[0];
    break;
  case 2:
    coords[1] = (coords[1]+1) % sizes[1];
    if (coords[1] == 0) {
      coords[0] = (coords[0]+1) % sizes[0];
      if (coords[0] == 0) {
	raw_pos = last_raw_pos + 1;
	ret = false;
      }
      else raw_pos += strides[0] - (sizes[1]-1)*strides[1];
    }
    else raw_pos += strides[1];
    break;
  default:
    int j = numDim;
    do {
      --j;
      coords[j] = (coords[j]+1) % sizes[j];
      if (coords[j] == 0) raw_pos -= (sizes[j]-1) * strides[j];
      else raw_pos += strides[j];
    } while(j>0 && coords[j] == 0);
    if (j == 0 && coords[0] == 0) {
      raw_pos = last_raw_pos + 1;
      ret = false;
    }
  }
  return ret;
}

template <typename T>
bool SparseMatrix<T>::nextCoordVectorColOrder(int *coords, int &raw_pos,
					      const int *sizes,
					      const int *strides,
					      const int numDim,
					      const int last_raw_pos) {
  bool ret = true;
  switch(numDim) {
  case 1:
    coords[0] = (coords[0]+1) % sizes[0];
    if (coords[0] == 0) {
      raw_pos = last_raw_pos + 1;
      ret = false;
    }
    else raw_pos += strides[0];
    break;
  case 2:
    coords[0] = (coords[0]+1) % sizes[0];
    if (coords[0] == 0) {
      coords[1] = (coords[1]+1) % sizes[1];
      if (coords[1] == 0) {
	raw_pos = last_raw_pos + 1;
	ret = false;
      }
      else raw_pos += strides[1] - (sizes[0]-1)*strides[0];
    }
    else raw_pos += strides[0];
    break;
  default:
    int j = 0;
    do {
      coords[j] = (coords[j]+1) % sizes[j];
      if (coords[j] == 0) {
	if (sizes[j] > 1) raw_pos -= (sizes[j]-1) * strides[j];
      }
      else raw_pos += strides[j];
    } while(coords[j++] == 0 && j<numDim);
    if (j == numDim && coords[numDim-1] == 0) {
      raw_pos = last_raw_pos + 1;
      ret = false;
    }
  }
  return ret;
}

template <typename T>
bool SparseMatrix<T>::nextCoordVectorRowOrder(int *coords) const {
  return nextCoordVectorRowOrder(coords, matrixSize, numDim);
}

template <typename T>
bool SparseMatrix<T>::nextCoordVectorColOrder(int *coords) const {
  return nextCoordVectorColOrder(coords, matrixSize, numDim);
}

template <typename T>
bool SparseMatrix<T>::nextCoordVectorRowOrder(int *coords,
					      const int *sizes,
					      int numDim) {
  int j = numDim;
  do {
    --j;
    coords[j] = (coords[j]+1) % sizes[j];
  } while(j>0 && coords[j] == 0);
  if (j == 0 && coords[0] == 0) return false;
  return true;
}

template <typename T>
bool SparseMatrix<T>::nextCoordVectorColOrder(int *coords,
					      const int *sizes,
					      int numDim) {
  int j = 0;
  do {
    coords[j] = (coords[j]+1) % sizes[j];
  } while(coords[j++] == 0 && j<numDim);
  if (j == numDim && coords[numDim-1] == 0) return false;
  return true;
}

template <typename T>
int SparseMatrix<T>::computeRawPos(const int *coords) const {
  int raw_pos;
  switch(numDim) {
  case 1:
    april_assert(coords[0] < matrixSize[0]);
    raw_pos = coords[0]*stride[0];
    break;
  case 2:
    april_assert(coords[0] < matrixSize[0]);
    april_assert(coords[1] < matrixSize[1]);
    raw_pos = coords[0]*stride[0]+coords[1]*stride[1];
    break;
  default:
    raw_pos=0;
    for(int i=0; i<numDim; i++) {
      april_assert(coords[i] < matrixSize[i]);
      raw_pos += stride[i]*coords[i];
    }
  }
  return raw_pos + offset;
}

template <typename T>
void SparseMatrix<T>::computeCoords(const int raw_pos, int *coords) const {
  int R = raw_pos - offset;
  switch(numDim) {
  case 1: coords[0] = R / stride[0]; break;
  case 2:
    switch(major_order) {
    case CblasRowMajor:
      coords[0] =  R / stride[0];
      coords[1] = (R % stride[0]) / stride[1];
      break;
    case CblasColMajor:
      coords[1] =  R / stride[1];
      coords[0] = (R % stride[1]) / stride[0];
      break;
    }
    break;
  default:
    switch(major_order) {
    case CblasRowMajor:
      for (int i=0; i<numDim; ++i) {
	coords[i] = R / stride[i];
	R = R % stride[i];
      }
      break;
    case CblasColMajor:
      for (int i=numDim-1; i>=0; --i) {
	coords[i] = R / stride[i];
	R = R % stride[i];
      }
      break;
    }
  }
}

template <typename T>
bool SparseMatrix<T>::getIsContiguous() const {
  if (is_contiguous != NONE) return (is_contiguous==CONTIGUOUS);
  if (major_order == CblasRowMajor) {
    int aux = 1;
    for (int i=numDim-1; i>=0; --i) {
      if(stride[i] != aux) {
	is_contiguous = NONCONTIGUOUS;
	return false;
      }
      else aux *= matrixSize[i];
    }
  }
  else {
    int aux = 1;
    for (int i=0; i<numDim; ++i) {
      if(stride[i] != aux) {
	is_contiguous = NONCONTIGUOUS;
	return false;
      }
      else aux *= matrixSize[i];
    }
  }
  is_contiguous = CONTIGUOUS;
  return true;
}

// expands current matrix to a diagonal matrix
template <typename T>
SparseMatrix<T> *SparseMatrix<T>::diagonalize() const {
  if (numDim != 1)
    ERROR_EXIT(128, "Only one-dimensional matrix is allowed\n");
  const int dims[2] = { matrixSize[0], matrixSize[0] };
  SparseMatrix<T> *resul  = new SparseMatrix<T>(2, dims, major_order);
  // resul_diag is a submatrix of resul, build to do a diagonal traverse
  const int stride  = matrixSize[0] + 1;
  SparseMatrix<T> *resul_diag = new SparseMatrix<T>(1, &stride, 0, dims, dims[0],
						    resul->last_raw_pos, resul->data,
						    resul->major_order,
						    resul->use_cuda,
						    resul->transposed);
  resul->zeros();
  resul_diag->copy(this);
  delete resul_diag;
  return resul;
}

template <typename T>
void SparseMatrix<T>::pruneSubnormalAndCheckNormal() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

 template <typename T>
 void SparseMatrix<T>::pushBack(const_iterator &b, const_iterator &e,
				int offset0, int offset1) {
   const_iterator it(b);
   while(it != e) {
     int x0,x1;
     it->getCoords(x0, x1);
     pushBack(x0-offset0, x1-offset1, *it);
     ++it;
   }
 }
