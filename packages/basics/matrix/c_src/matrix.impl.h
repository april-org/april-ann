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
#ifndef MATRIX_IMPL_H
#define MATRIX_IMPL_H
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "april_print.h"
#include "cmath_overloads.h"
#include "error_print.h"
#include "ignore_result.h"
#include "matrix.h"
#include "omp_utils.h"

// Must be defined in this order.
#include "matrix_operations.h"

namespace Basics {

  template<typename T>
  const unsigned int Matrix<T>::MATRIX_BINARY_VERSION = 0x00000001;

  template <typename T>
  void Matrix<T>::initialize(const int *dim) {
    total_size=1;
    for(int i=numDim-1; i>=0; --i) {
      stride[i] = total_size;
      total_size *= dim[i];
      matrixSize[i] = dim[i];
      april_assert(matrixSize[i] > 0);
    }
    last_raw_pos = total_size-1;
  }

  /// Allocation of memory for data pointer. It is Referenced for sharing.
  template <typename T>
  void Matrix<T>::allocate_memory(int size) {
    data.reset( new AprilMath::GPUMirroredMemoryBlock<T>(static_cast<unsigned int>(size)) );
#ifndef NDEBUG
    // Initialization to NaN allows to find not initialized memory blocks.
    AprilMath::MatrixExt::Operations::
      matFill(this, AprilMath::Limits<T>::quiet_NaN());
#endif
  }

  /// Release of the memory allocated for data pointer.
  template <typename T>
  void Matrix<T>::release_memory() {
    data.reset();
  }

  /// Null constructor
  template <typename T>
  Matrix<T>::Matrix(int numDim, const int *stride, const int offset,
                    const int *matrixSize,
                    const int total_size,
                    const int last_raw_pos,
                    AprilMath::GPUMirroredMemoryBlock<T> *data,
                    const bool use_cuda,
                    AprilUtils::MMappedDataReader *mmapped_data) :
    AprilIO::Serializable(), shared_count(0),
    numDim(numDim), stride(new int[numDim]), offset(offset),
    matrixSize(new int[numDim]), total_size(total_size),
    last_raw_pos(last_raw_pos), data(data), mmapped_data(mmapped_data),
    use_cuda(use_cuda),
    is_contiguous(NONE),
    end_iterator(), end_const_iterator(), end_span_iterator_() {
    for (int i=0; i<numDim; ++i) {
      this->stride[i] = stride[i];
      this->matrixSize[i] = matrixSize[i];
    }
    april_assert(offset >= 0);
  }

  /// Default constructor
  template <typename T>
  Matrix<T>::Matrix(int numDim,
                    const int* dim,
                    AprilMath::GPUMirroredMemoryBlock<T> *data,
                    int offset) :
    AprilIO::Serializable(), shared_count(0),
    numDim(numDim),
    offset(offset),
    data(data),
    use_cuda(AprilMath::GPUMirroredMemoryBlockBase::USE_CUDA_DEFAULT),
    is_contiguous(NONE),
    end_iterator(), end_const_iterator(), end_span_iterator_() {
    stride     = new int[numDim];
    matrixSize = new int[numDim];
    initialize(dim);
    last_raw_pos += offset;
    if (this->data.empty()) allocate_memory(total_size);
    else {
      if (static_cast<int>(this->data->getSize()) < offset + size())
        ERROR_EXIT2(128, "Data pointer size doesn't fit, expected %d, found %d\n",
                    size(), data->getSize());
    }
    april_assert(offset >= 0);
  }

  /// Constructor for sub-matrix building
  template <typename T>
  Matrix<T>::Matrix(Matrix<T> *other,
                    const int* coords, const int *sizes,
                    bool clone) :
    AprilIO::Serializable(),
    shared_count(0),
    numDim(other->numDim),
    offset(0),
    use_cuda(other->use_cuda),
    is_contiguous(NONE),
    end_iterator(), end_const_iterator(), end_span_iterator_() {
    for (int i=0; i<numDim; i++) {
      if (sizes[i] + coords[i] > other->matrixSize[i])
        ERROR_EXIT3(128, "Size+coordinates are out of dimension size: %d+%d>%d\n",
                    sizes[i], coords[i], other->matrixSize[i]);
    }
    stride     = new int[numDim];
    matrixSize = new int[numDim];
    if (clone) {
      initialize(sizes);
      allocate_memory(total_size);
      span_iterator it(this);
      const int *dims_order = it.getDimOrder();
      const int first_dim = dims_order[0];
      const int this_stride = it.getStride();
      const int other_stride = other->getStrideSize(first_dim);
      const int N = it.numberOfIterations();
      for (int i=0; i<N; ++i) {
        april_assert(it != this->end_span_iterator());
        const int other_raw_pos = other->computeRawPos(it.getCoordinates(), coords);
        doCopy(it.getSize(),
               other->data.get(), other_stride, other_raw_pos,
               this->data.get(), this_stride, it.getOffset(),
               this->getCudaFlag());
        ++it;
      }
      april_assert(it == this->end_span_iterator());
    } // if (clone)
    else { // !clone
      AprilUtils::UniquePtr<int []> aux_coords( new int[numDim] );
      total_size = 1;
      for (int i=0; i<numDim; i++) {
        stride[i]     = other->stride[i];
        matrixSize[i] = sizes[i];
        total_size    = total_size * sizes[i];
        aux_coords[i] = sizes[i]-1;
      }
      offset = other->computeRawPos(coords);
      data = other->data;
      last_raw_pos = computeRawPos(aux_coords.get());
    } // !clone
    april_assert(offset >= 0);
  }

  // Constructor for sub-matrix building from a CONST matrix
  template <typename T>
  Matrix<T>::Matrix(const Matrix<T> *other,
                    const int* coords, const int *sizes,
                    bool clone) :
    AprilIO::Serializable(),
    shared_count(0),
    numDim(other->numDim),
    offset(0),
    use_cuda(other->use_cuda),
    is_contiguous(NONE),
    end_iterator(), end_const_iterator(), end_span_iterator_() {
    for (int i=0; i<numDim; i++) {
      if (sizes[i] + coords[i] > other->matrixSize[i])
        ERROR_EXIT3(128, "Size+coordinates are out of dimension size: %d+%d>%d\n",
                    sizes[i], coords[i], other->matrixSize[i]);
    }
    stride     = new int[numDim];
    matrixSize = new int[numDim];
    if (clone) {
      initialize(sizes);
      allocate_memory(total_size);
      span_iterator it(this), it_other(other, it.getDimOrder());
      const int *dims_order = it_other.getDimOrder();
      const int first_dim = dims_order[0];
      int diff_raw_pos = other->getStrideSize(first_dim) * coords[first_dim];
      int other_first_iteration = 0;
      if (numDim > 1) {
        // compute it_other traversal length until the coordinates of the slice
        // (except first_dim)
        AprilUtils::UniquePtr<int []> aux_stride( new int[numDim] );
        aux_stride[dims_order[0]] = 1;
        aux_stride[dims_order[1]] = 1;
        for (int i=2; i<numDim; ++i) {
          aux_stride[dims_order[i]] = aux_stride[dims_order[i-1]] * other->matrixSize[i-1];
        }
        other_first_iteration = 0;
        for (int i=1; i<numDim; ++i){ 
          other_first_iteration += aux_stride[dims_order[i]] * coords[dims_order[i]];
        }
      }
      const int N = it.numberOfIterations();
#ifndef NO_OMP
      if (N > OMPUtils::get_num_threads()) {
#pragma omp parallel for firstprivate(it) firstprivate(it_other)
        for (int i=0; i<N; ++i) {
          it.setAtIteration(i);
          it_other.setAtIteration(other_first_iteration + i);
          april_assert(it != this->end_span_iterator());
          april_assert(it_other != other->end_span_iterator());
          doCopy(it.getSize(),
                 other->data.get(),
                 it_other.getStride(),
                 diff_raw_pos + it_other.getOffset(),
                 this->data.get(),
                 it.getStride(),
                 it.getOffset(),
                 this->getCudaFlag());
        }
      }
      else {
#else
        it_other.setAtIteration(other_first_iteration);
        for (int i=0; i<N; ++i) {
          april_assert(it != this->end_span_iterator());
          april_assert(it_other != other->end_span_iterator());
          doCopy(it.getSize(),
                 other->data.get(), it_other.getStride(),
                 diff_raw_pos + it_other.getOffset(),
                 this->data.get(), it.getStride(), it.getOffset(),
                 this->getCudaFlag());
          ++it;
          ++it_other;
        }
#endif
#ifndef NO_OMP
      }
#endif
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
      data.reset( new AprilMath::
                  GPUMirroredMemoryBlock<T>(other->size(),
                                            other->data->getPPALForRead()) );
      last_raw_pos = computeRawPos(aux_coords);
      delete[] aux_coords;
    }
    april_assert(offset >= 0);
  }

  /// Constructor with variable arguments
  template <typename T>
  Matrix<T>::Matrix(int numDim, int d1, ...) :
    AprilIO::Serializable(), shared_count(0),
    numDim(numDim),
    offset(0),
    use_cuda(AprilMath::GPUMirroredMemoryBlockBase::USE_CUDA_DEFAULT),
    is_contiguous(NONE),
    end_iterator(), end_const_iterator(), end_span_iterator_() {
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
    AprilIO::Serializable(),
    shared_count(0),
    numDim(other->numDim),
    offset(0),
    use_cuda(other->use_cuda),
    is_contiguous(NONE),
    end_iterator(), end_const_iterator(), end_span_iterator_() {
    stride       = new int[numDim];
    matrixSize   = new int[numDim];
    total_size   = other->total_size;
    last_raw_pos = other->last_raw_pos;
    if (clone) {
      initialize(other->matrixSize);
      allocate_memory(total_size);
      AprilMath::MatrixExt::Operations::matCopy(this, other);
    }
    else {
      offset       = other->offset;
      data         = other->data;
      for (int i=0; i<numDim; ++i) {
        stride[i]     = other->stride[i];
        matrixSize[i] = other->matrixSize[i];
      }
    }
    april_assert(offset >= 0);
  }

  template <typename T>
  Matrix<T> *Matrix<T>::fromMMappedDataReader(AprilUtils::MMappedDataReader
                                              *mmapped_data) {
    Matrix<T> *obj = new Matrix();
    //
    obj->data.reset( AprilMath::GPUMirroredMemoryBlock<T>::
                     fromMMappedDataReader(mmapped_data) );
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
    CBLAS_ORDER dummy1 = *(mmapped_data->get<CBLAS_ORDER>());
    bool dummy2        = *(mmapped_data->get<bool>()); // legacy transposed flag
    UNUSED_VARIABLE(dummy1);
    UNUSED_VARIABLE(dummy2);
    // NON MAPPED DATA
    obj->use_cuda      = AprilMath::GPUMirroredMemoryBlockBase::USE_CUDA_DEFAULT;
    obj->shared_count  = 0;
    obj->is_contiguous = NONE;
    // THE MMAP POINTER
    obj->mmapped_data.reset( mmapped_data );
    //
    april_assert(obj->offset >= 0);
    return obj;
  }

  template <typename T>
  void Matrix<T>::toMMappedDataWriter(AprilUtils::MMappedDataWriter
                                      *mmapped_data) const {
    data->toMMappedDataWriter(mmapped_data);
    mmapped_data->put(&MATRIX_BINARY_VERSION);
    mmapped_data->put(&numDim);
    mmapped_data->put(stride, numDim);
    mmapped_data->put(&offset);
    mmapped_data->put(matrixSize, numDim);
    mmapped_data->put(&total_size);
    mmapped_data->put(&last_raw_pos);
    CBLAS_ORDER dummy1=CblasRowMajor;
    mmapped_data->put(&dummy1);
    bool dummy2=false; // legacy transposed flag
    mmapped_data->put(&dummy2);
  }

  template <typename T>
  Matrix<T>::~Matrix() {
    release_memory();
    if (mmapped_data.empty()) {
      delete[] stride;
      delete[] matrixSize;
    }
  }

  template <typename T>
  void Matrix<T>::print() const {
    for (Matrix<T>::const_iterator it(begin()); it != end(); ++it) {
      AprilUtils::aprilPrint(*it);
      printf(" ");
    }
    printf("\n");
  }

  template <typename T>
  Matrix<T> *Matrix<T>::rewrap(const int *new_dims, int len,
                               bool clone_if_not_contiguous) {
    bool need_clone = false;
    Matrix<T> * obj;
    if (!getIsContiguous()) {
      if (!clone_if_not_contiguous) {
        ERROR_EXIT(128, "Impossible to re-wrap non contiguous matrix, "
                   "clone it first\n");
      }
      else {
        need_clone = true;
      }
    }
    bool equal   = true;
    int new_size = 1;
    for (int i=0; i<len; ++i) {
      if (i>=numDim || new_dims[i] != matrixSize[i]) equal=false;
      new_size *= new_dims[i];
    }
    if (len==numDim && equal) return this;
    if (new_size != size()) {
      ERROR_EXIT2(128, "Incorrect size, expected %d, and found %d\n",
                  size(), new_size);
    }
    if (need_clone) {
      AprilMath::GPUMirroredMemoryBlock<T> *new_data =
        new AprilMath::GPUMirroredMemoryBlock<T>(new_size);
      obj = new Matrix<T>(len, new_dims, new_data);
      AprilUtils::SharedPtr< Matrix<T> > aux( obj->rewrap(this->getDimPtr(),
                                                           this->getNumDim()) );
      AprilMath::MatrixExt::Operations::matCopy(aux.get(),this);
    }
    else {
      obj = new Matrix<T>(len, new_dims, data.get(), offset);
    }
#ifdef USE_CUDA
    obj->setUseCuda(use_cuda);
#endif
    return obj;
  }
  
  template <typename T>
  Matrix<T> *Matrix<T>::squeeze() {
    int len = 0;
    AprilUtils::UniquePtr<int []> sizes(new int[getNumDim()]);
    AprilUtils::UniquePtr<int []> strides(new int[getNumDim()]);
    for (int i=0; i<getNumDim(); ++i) {
      int sz = getDimSize(i);
      if (sz > 1) {
        strides[len] = getStrideSize(i);
        sizes[len++] = sz;
      }
    }
    // matrices with 1x1x1x...x1 dimensions need the following sanity check
    if (len == 0) {
      strides[len] = 1;
      sizes[len++] = 1;
    }
    // return this in case len==numDim, rewrap in other case
    Matrix<T> *obj = (len==numDim) ?
      this : new Matrix<T>(len, strides.get(), getOffset(), sizes.get(),
                           size(), last_raw_pos, data.get(),
                           use_cuda, mmapped_data.get());
#ifdef USE_CUDA
    obj->setUseCuda(use_cuda);
#endif
    return obj;
  }

  template<typename T>
  Matrix<T> *Matrix<T>::transpose() {
    Matrix<T> *result;
    if (this->numDim > 1) {
      result = this->shallowCopy();
      for (int i=0,j=numDim-1; i<numDim; ++i,--j) {
        result->stride[j]     = this->stride[i];
        result->matrixSize[j] = this->matrixSize[i];
      }
      result->is_contiguous = NONE;
    }
    else result = this;
    return result;
  }

  /// Symbolic transposition, changes strides order
  template<typename T>
  Matrix<T>* Matrix<T>::transpose(int dim1, int dim2) {
    if (dim1 == dim2) return this;
    if (dim1 < 0 || dim1 >= numDim ||
        dim2 < 0 || dim2 >= numDim) {
      ERROR_EXIT4(128, "Incorrect dimensions, exepected to be "
                  "in range [%d,%d], given dim1=%d, dim2=%d\n",
                  0, numDim-1, dim1, dim2);
    }
    Matrix<T> *result;
    result = this->shallowCopy();
    AprilUtils::swap(result->stride[dim1], result->stride[dim2]);
    AprilUtils::swap(result->matrixSize[dim1], result->matrixSize[dim2]);
    return result;
  }
  
  template <typename T>
  Matrix<T>* Matrix<T>::cloneOnlyDims() const {
    Matrix<T> *obj = new Matrix<T>(numDim, matrixSize);
#ifdef USE_CUDA
    obj->setUseCuda(use_cuda);
#endif
    return obj;
  }

  template <typename T>
  Matrix<T>* Matrix<T>::clone() const {
    Matrix<T> *result = this->cloneOnlyDims();
    AprilMath::MatrixExt::Operations::matCopy(result,this);
    return result;
  }

  template <typename T>
  Matrix<T>* Matrix<T>::shallowCopy() {
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
      result->use_cuda     = use_cuda;
      result->numDim       = d;
      result->matrixSize   = new int[d];
      result->stride       = new int[d];
      result->offset       = offset + index*stride[dim]; // the select implies an offset
      result->last_raw_pos = result->offset;
      result->data         = data;
      // Not needed: result->mmapped_data = 0;
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
    april_assert(result->offset >= 0);
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
  int Matrix<T>::computeRawPos(const int *coords, const int *offset) const {
    int raw_pos;
    switch(numDim) {
    case 1:
      april_assert(coords[0]+offset[0] < matrixSize[0]);
      raw_pos = (coords[0]+offset[0])*stride[0];
      break;
    case 2:
      april_assert(coords[0]+offset[0] < matrixSize[0]);
      april_assert(coords[1]+offset[1] < matrixSize[1]);
      raw_pos = (coords[0]+offset[0])*stride[0]+(coords[1]+offset[1])*stride[1];
      break;
    default:
      raw_pos=0;
      for(int i=0; i<numDim; i++) {
        april_assert(coords[i]+offset[i] < matrixSize[i]);
        raw_pos += stride[i]*(coords[i]+offset[i]);
      }
    }
    return raw_pos + this->offset;
  }

  /// FIXME: Change it to compute coords traversing strides in descending order?
  template <typename T>
  void Matrix<T>::computeCoords(const int raw_pos, int *coords) const {
    int R = raw_pos - offset;
    switch(numDim) {
    case 1: coords[0] = R / stride[0]; break;
    case 2:
      coords[0] =  R / stride[0];
      coords[1] = (R % stride[0]) / stride[1];
      break;
    default:
      for (int i=0; i<numDim; ++i) {
        coords[i] = R / stride[i];
        R = R % stride[i];
      }
    }
  }

  template <typename T>
  bool Matrix<T>::getIsContiguous() const {
    if (is_contiguous != NONE) return (is_contiguous==CONTIGUOUS);
    int aux = 1;
    for (int i=numDim-1; i>=0; --i) {
      if(stride[i] != aux) {
        is_contiguous = NONCONTIGUOUS;
        return false;
      }
      else aux *= matrixSize[i];
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
    Matrix<T> *resul  = new Matrix<T>(2, dims);
    // resul_diag is a submatrix of resul, build to do a diagonal traverse
    const int stride  = matrixSize[0] + 1;
    Matrix<T> *resul_diag = new Matrix<T>(1, &stride, 0, dims, dims[0],
                                          resul->last_raw_pos, resul->data.get(),
                                          resul->use_cuda);
    AprilMath::MatrixExt::Operations::matZeros(resul);
    AprilMath::MatrixExt::Operations::matCopy(resul_diag, this);
    delete resul_diag;
    return resul;
  }

  template <typename T>
  void Matrix<T>::pruneSubnormalAndCheckNormal() {
    ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  }

  template <typename T>
  Matrix<T> *Matrix<T>::padding(int *begin_padding, int *end_padding,
                                T default_value) const {
    int *result_sizes = new int[getNumDim()];
    int *matrix_pos = new int[getNumDim()];
    for (int i=0; i<getNumDim(); ++i) {
      result_sizes[i] = getDimSize(i) + begin_padding[i] + end_padding[i];
      matrix_pos[i] = begin_padding[i];
    }
    Matrix<T> *result = new Matrix<T>(getNumDim(), result_sizes);
    // FIXME: implement fill by several submatrices for large matrix sizes with
    // small padding sizes
    AprilMath::MatrixExt::Operations::matFill(result, default_value);
    // take submatrix where data will be located
    Matrix<T> *result_data = new Matrix<T>(result, matrix_pos, getDimPtr(),
                                           false);
    // copy data to the submatrix
    AprilMath::MatrixExt::Operations::matCopy(result_data, this);
    //
    delete result_data;
    delete[] result_sizes;
    delete[] matrix_pos;
    return result;
  }

  template <typename T>
  Matrix<T> *Matrix<T>::padding(int pad_value, T default_value) const {
    int *result_sizes = new int[getNumDim()];
    int *matrix_pos = new int[getNumDim()];
    for (int i=0; i<getNumDim(); ++i) {
      result_sizes[i] = getDimSize(i) + pad_value*2;
      matrix_pos[i] = pad_value;
    }
    Matrix<T> *result = new Matrix<T>(getNumDim(), result_sizes);
    // FIXME: implement fill by several submatrices for large matrix sizes with
    // small padding sizes
    AprilMath::MatrixExt::Operations::matFill(result, default_value);
    // take submatrix where data will be located
    Matrix<T> *result_data = new Matrix<T>(result, matrix_pos, getDimPtr(),
                                           false);
    // copy data to the submatrix
    AprilMath::MatrixExt::Operations::matCopy(result_data, this);
    //
    delete result_data;
    delete[] result_sizes;
    delete[] matrix_pos;
    return result;
  }
  
} // namespace Basics

#endif // MATRIX_IMPL_H
