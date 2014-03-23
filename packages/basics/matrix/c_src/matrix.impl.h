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
const unsigned int Matrix<T>::MATRIX_BINARY_VERSION = 0x00000001;

template <typename T>
void Matrix<T>::initialize(const int *dim) {
  total_size=1;
  switch(major_order) {
  case CblasRowMajor:
    for(int i=numDim-1; i>=0; --i) {
      stride[i] = total_size;
      total_size *= dim[i];
      matrixSize[i] = dim[i];
      april_assert(matrixSize[i] > 0);
    }
    break;
  case CblasColMajor:
    for(int i=0; i<numDim; ++i) {
      stride[i] = total_size;
      total_size *= dim[i];
      matrixSize[i] = dim[i];
      april_assert(matrixSize[i] > 0);
    }
    break;
  default:
    ERROR_EXIT(128, "Incorrect major order!!!\n");
  }
  last_raw_pos = total_size-1;
}

/// Allocation of memory for data pointer. It is Referenced for sharing.
template <typename T>
void Matrix<T>::allocate_memory(int size) {
  data = new GPUMirroredMemoryBlock<T>(static_cast<unsigned int>(size));
  IncRef(data);
}

/// Release of the memory allocated for data pointer.
template <typename T>
void Matrix<T>::release_memory() {
  DecRef(data);
}

/// Null constructor
template <typename T>
Matrix<T>::Matrix(int numDim, const int *stride, const int offset,
		  const int *matrixSize,
		  const int total_size,
		  const int last_raw_pos,
		  GPUMirroredMemoryBlock<T> *data,
		  const CBLAS_ORDER major_order,
		  const bool use_cuda,
		  const bool transposed,
		  april_utils::MMappedDataReader *mmapped_data) :
  Referenced(), shared_count(0), transposed(transposed),
  numDim(numDim), stride(new int[numDim]), offset(offset),
  matrixSize(new int[numDim]), total_size(total_size),
  last_raw_pos(last_raw_pos), data(data), mmapped_data(mmapped_data),
  major_order(major_order),
  use_cuda(use_cuda),
  is_contiguous(NONE),
  end_iterator(), end_const_iterator(), end_best_span_iterator() {
  IncRef(data);
  if (mmapped_data) IncRef(mmapped_data);
  for (int i=0; i<numDim; ++i) {
    this->stride[i] = stride[i];
    this->matrixSize[i] = matrixSize[i];
  }
}

/// Default constructor
template <typename T>
Matrix<T>::Matrix(int numDim,
		  const int* dim,
		  CBLAS_ORDER major_order,
		  GPUMirroredMemoryBlock<T> *data,
		  int offset,
		  bool transposed) :
  Referenced(), shared_count(0), transposed(transposed),
  numDim(numDim),
  offset(offset),
  mmapped_data(0),
  major_order(major_order),
  use_cuda(false),
  is_contiguous(CONTIGUOUS),
  end_iterator(), end_const_iterator(), end_best_span_iterator() {
  stride     = new int[numDim];
  matrixSize = new int[numDim];
  initialize(dim);
  last_raw_pos += offset;
  if (data == 0) allocate_memory(total_size);
  else {
    if (static_cast<int>(data->getSize()) < offset + size())
      ERROR_EXIT2(128, "Data pointer size doesn't fit, expected %d, found %d\n",
		  size(), data->getSize());
    this->data = data;
    IncRef(data);
  }
}

/// Constructor for sub-matrix building
template <typename T>
Matrix<T>::Matrix(Matrix<T> *other,
		  const int* coords, const int *sizes,
		  bool clone) :
  Referenced(),
  shared_count(0), transposed(other->transposed),
  numDim(other->numDim),
  offset(0),
  mmapped_data(0),
  major_order(other->major_order),
  use_cuda(other->use_cuda),
  is_contiguous(NONE),
  end_iterator(), end_const_iterator(), end_best_span_iterator() {
  for (int i=0; i<numDim; i++) {
    if (sizes[i] + coords[i] > other->matrixSize[i])
      ERROR_EXIT3(128, "Size+coordinates are out of dimension size: %d+%d>%d\n",
		  sizes[i], coords[i], other->matrixSize[i]);
  }
  stride     = new int[numDim];
  matrixSize = new int[numDim];
  if (clone) {
    transposed    = false;
    is_contiguous = CONTIGUOUS;
    initialize(sizes);
    allocate_memory(total_size);
    int other_raw_pos = other->computeRawPos(coords);
    const T *other_data = other->data->getPPALForRead();
    int *aux_coords = new int[numDim];
    for (int i=0; i<numDim; ++i) aux_coords[i] = 0;
    if (major_order == CblasRowMajor) {
      for (iterator it(begin()); it!=end(); ++it) {
	*it = other_data[other_raw_pos];
	nextCoordVectorRowOrder(aux_coords, other_raw_pos,
				sizes, other->stride, numDim,
				other->last_raw_pos);
      }
    }
    else {
      for (col_major_iterator it(begin()); it!=end(); ++it) {
	*it = other_data[other_raw_pos];
	nextCoordVectorColOrder(aux_coords, other_raw_pos,
				sizes, other->stride, numDim,
				other->last_raw_pos);
      }
    }
    delete[] aux_coords;
  }
  else {
    int *aux_coords = new int[numDim];
    total_size = 1;
    for (int i=0; i<numDim; i++) {
      stride[i]     = other->stride[i];
      matrixSize[i] = sizes[i];
      total_size    = total_size * sizes[i];
      aux_coords[i] = sizes[i]-1;
    }
    offset = other->computeRawPos(coords);
    data   = other->data;
    IncRef(data);
    last_raw_pos = computeRawPos(aux_coords);
    delete[] aux_coords;
  }
}


/// Constructor with variable arguments
template <typename T>
Matrix<T>::Matrix(int numDim, int d1, ...) :
  Referenced(), shared_count(0), transposed(false),
  numDim(numDim),
  offset(0),
  mmapped_data(0),
  major_order(CblasRowMajor),
  is_contiguous(CONTIGUOUS),
  end_iterator(), end_const_iterator(), end_best_span_iterator() {
  int *dim   = new int[numDim];
  stride     = new int[numDim];
  matrixSize = new int[numDim];
  va_list ap;
  va_start(ap, d1);
  dim[0]=d1;
  for (int i=1; i<numDim; i++) {
    int di = va_arg(ap, int);
    dim[i] = di;
  }
  va_end(ap);
  initialize(dim);
  allocate_memory(total_size);
  delete[] dim;
}


/// Constructor for copy or clone other given matrix
template <typename T>
Matrix<T>::Matrix(Matrix<T> *other, bool clone) :
  Referenced(),
  shared_count(0), transposed(other->transposed),
  numDim(other->numDim),
  offset(0),
  mmapped_data(0),
  major_order(other->major_order),
  use_cuda(other->use_cuda),
  is_contiguous(other->is_contiguous),
  end_iterator(), end_const_iterator(), end_best_span_iterator() {
  stride       = new int[numDim];
  matrixSize   = new int[numDim];
  total_size   = other->total_size;
  last_raw_pos = other->last_raw_pos;
  if (clone) {
    transposed = false;
    initialize(other->matrixSize);
    allocate_memory(total_size);
    copy(other);
    is_contiguous = CONTIGUOUS;
  }
  else {
    offset       = other->offset;
    data         = other->data;
    IncRef(data);
    for (int i=0; i<numDim; ++i) {
      stride[i]     = other->stride[i];
      matrixSize[i] = other->matrixSize[i];
    }
  }
}

template <typename T>
Matrix<T> *Matrix<T>::fromMMappedDataReader(april_utils::MMappedDataReader
					    *mmapped_data) {
  Matrix<T> *obj = new Matrix();
  //
  obj->data = GPUMirroredMemoryBlock<T>::fromMMappedDataReader(mmapped_data);
  IncRef(obj->data);
  //
  unsigned int binary_version = *(mmapped_data->get<unsigned int>());
  if (binary_version != MATRIX_BINARY_VERSION)
    ERROR_EXIT1(128,
		"Incorrect binary matrix version from commit number %d\n",
		mmapped_data->getCommitNumber());
  int N = *(mmapped_data->get<int>());
  obj->numDim        = N;
  obj->stride        = mmapped_data->get<int>(N);
  obj->offset        = *(mmapped_data->get<int>());
  obj->matrixSize    = mmapped_data->get<int>(N);
  obj->total_size    = *(mmapped_data->get<int>());
  obj->last_raw_pos  = *(mmapped_data->get<int>());
  obj->major_order   = *(mmapped_data->get<CBLAS_ORDER>());
  obj->transposed    = *(mmapped_data->get<bool>());
  // NON MAPPED DATA
  obj->use_cuda      = false;
  obj->shared_count  = 0;
  obj->is_contiguous = NONE;
  // THE MMAP POINTER
  obj->mmapped_data  = mmapped_data;
  IncRef(obj->mmapped_data);
  //
  return obj;
}

template <typename T>
void Matrix<T>::toMMappedDataWriter(april_utils::MMappedDataWriter
				    *mmapped_data) const {
  data->toMMappedDataWriter(mmapped_data);
  mmapped_data->put(&MATRIX_BINARY_VERSION);
  mmapped_data->put(&numDim);
  mmapped_data->put(stride, numDim);
  mmapped_data->put(&offset);
  mmapped_data->put(matrixSize, numDim);
  mmapped_data->put(&total_size);
  mmapped_data->put(&last_raw_pos);
  mmapped_data->put(&major_order);
  mmapped_data->put(&transposed);
}

template <typename T>
Matrix<T>::~Matrix() {
  release_memory();
  if (mmapped_data == 0) {
    delete[] stride;
    delete[] matrixSize;
  }
  else DecRef(mmapped_data);
}

template <typename T>
Matrix<T> *Matrix<T>::rewrap(const int *new_dims, int len) {
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
  Matrix<T> *obj = new Matrix<T>(len, new_dims, major_order, data, offset);
#ifdef USE_CUDA
  obj->setUseCuda(use_cuda);
#endif
  return obj;
}

template<typename T>
Matrix<T> *Matrix<T>::transpose() {
  Matrix<T> *result;
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

template<typename T>
Matrix<T> *Matrix<T>::changeOrder(CBLAS_ORDER new_major_order) {
  Matrix<T> *result;
  if (new_major_order == major_order) result = this;
  else result = new Matrix<T>(numDim, stride, offset, matrixSize, total_size,
			      last_raw_pos, data, new_major_order, use_cuda,
			      transposed, mmapped_data);
  return result;
}

template <typename T>
Matrix<T>* Matrix<T>::cloneOnlyDims() const {
  Matrix<T> *obj = new Matrix<T>(numDim, matrixSize, major_order);
#ifdef USE_CUDA
  obj->setUseCuda(use_cuda);
#endif
  return obj;
}

template<typename T>
Matrix<T> *Matrix<T>::clone(CBLAS_ORDER major_order) {
  Matrix<T> *resul;
  if (this->major_order != major_order) {
    resul = new Matrix<T>(numDim, matrixSize, major_order);
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
Matrix<T>* Matrix<T>::clone() {
  return new Matrix<T>(this,true);
}

template <typename T>
Matrix<T>* Matrix<T>::shallow_copy() {
  return new Matrix<T>(this,false);
}

template <typename T>
T& Matrix<T>::operator[] (int i) {
  return data->get(static_cast<unsigned int>(i));
}

template <typename T>
const T& Matrix<T>::operator[] (int i) const {
  return data->get(static_cast<unsigned int>(i));
}

template <typename T>
T& Matrix<T>::operator() (int i) {
  april_assert(numDim == 1);
  int raw_pos = computeRawPos(&i);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
T& Matrix<T>::operator() (int row, int col) {
  april_assert(numDim == 2);
  int pos[2]={row,col};
  int raw_pos = computeRawPos(pos);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
T& Matrix<T>::operator() (int coord0, int coord1, int coord2, ...) {
  april_assert(numDim >= 3);
  int *aux_coords = new int[numDim];
  aux_coords[0] = coord0;
  aux_coords[1] = coord1;
  aux_coords[2] = coord2;
  va_list ap;
  va_start(ap, coord2);
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
T& Matrix<T>::operator() (int *coords, int sz) {
  UNUSED_VARIABLE(sz);
  april_assert(numDim == sz);
  int raw_pos = computeRawPos(coords);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
const T& Matrix<T>::operator() (int i) const {
  april_assert(numDim == 1);
  int raw_pos = computeRawPos(&i);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
const T& Matrix<T>::operator() (int row, int col) const {
  april_assert(numDim == 2);
  int pos[2]={row,col};
  int raw_pos = computeRawPos(pos);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
const T& Matrix<T>::operator() (int coord0, int coord1, int coord2, ...) const {
  april_assert(numDim >= 3);
  int *aux_coords = new int[numDim];
  aux_coords[0] = coord0;
  aux_coords[1] = coord1;
  aux_coords[2] = coord2;
  va_list ap;
  va_start(ap, coord2);
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
const T& Matrix<T>::operator() (int *coords, int sz) const {
  UNUSED_VARIABLE(sz);
  april_assert(numDim == sz);
  int raw_pos = computeRawPos(coords);
  return data->get(static_cast<unsigned int>(raw_pos));
}

template <typename T>
bool Matrix<T>::getCol(int col, T* vec, int vecsize) {
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
bool Matrix<T>::putCol(int col, T* vec, int vecsize) {
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
bool Matrix<T>::putSubCol(int col, int first_row, T* vec, int vecsize) {
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
bool Matrix<T>::sameDim(const Matrix<O> *other) const {
  return sameDim(other->getDimPtr(), other->getNumDim());
}

template <typename T>
bool Matrix<T>::sameDim(const int *dims, const int len) const {
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
Matrix<T> *Matrix<T>::select(int dim, int index, Matrix<T> *dest) {
  if (numDim == 1)
    ERROR_EXIT(128, "Not possible to execute select for numDim=1\n");
  if (dim >= numDim)
    ERROR_EXIT(128, "Select for a dimension which doesn't exists\n");
  if (index >= matrixSize[dim])
    ERROR_EXIT(128, "Select for an index out of the matrix\n");
  Matrix<T> *result;
  if (dest == 0) {
    result = new Matrix();
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
    dest->changeSubMatrixData(dest_offset, dest_last_raw_pos);
    //
    result = dest;
  }
  return result;
}

/***** COORDINATES METHODS *****/

template <typename T>
bool Matrix<T>::nextCoordVectorRowOrder(int *coords, int &raw_pos) const {
  return nextCoordVectorRowOrder(coords, raw_pos, matrixSize, stride, numDim,
				 last_raw_pos);
}

template <typename T>
bool Matrix<T>::nextCoordVectorColOrder(int *coords, int &raw_pos) const {
  return nextCoordVectorColOrder(coords, raw_pos, matrixSize, stride, numDim,
				 last_raw_pos);
}

template <typename T>
bool Matrix<T>::nextCoordVectorRowOrder(int *coords, int &raw_pos,
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
bool Matrix<T>::nextCoordVectorColOrder(int *coords, int &raw_pos,
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
bool Matrix<T>::nextCoordVectorRowOrder(int *coords) const {
  return nextCoordVectorRowOrder(coords, matrixSize, numDim);
}

template <typename T>
bool Matrix<T>::nextCoordVectorColOrder(int *coords) const {
  return nextCoordVectorColOrder(coords, matrixSize, numDim);
}

template <typename T>
bool Matrix<T>::nextCoordVectorRowOrder(int *coords,
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
bool Matrix<T>::nextCoordVectorColOrder(int *coords,
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
int Matrix<T>::computeRawPos(const int *coords) const {
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
void Matrix<T>::computeCoords(const int raw_pos, int *coords) const {
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
bool Matrix<T>::getIsContiguous() const {
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
Matrix<T> *Matrix<T>::diagonalize() const {
  if (numDim != 1)
    ERROR_EXIT(128, "Only one-dimensional matrix is allowed\n");
  const int dims[2] = { matrixSize[0], matrixSize[0] };
  Matrix<T> *resul  = new Matrix<T>(2, dims, major_order);
  // resul_diag is a submatrix of resul, build to do a diagonal traverse
  const int stride  = matrixSize[0] + 1;
  Matrix<T> *resul_diag = new Matrix<T>(1, &stride, 0, dims, dims[0],
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
void Matrix<T>::pruneSubnormalAndCheckNormal() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}
