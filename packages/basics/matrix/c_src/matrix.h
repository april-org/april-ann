/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
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
#ifndef MATRIX_H
#define MATRIX_H

#include <cmath>
#include <cassert>
#include <cstdarg>
#include <new> // surprisingly, placement new doesn't work without this
#include "cblas_headers.h"
#include "wrapper.h"
#include "gpu_mirrored_memory_block.h"
#include "referenced.h"
#include "constants.h"
#include "clamp.h"
#include "aligned_memory.h"
#include "swap.h"

template <typename T>
class Matrix : public Referenced {
protected:
  /// Number of dimensions
  int numDim;
  /// Size of each dimension
  int *stride;
  /// Indication for sub-matrices
  bool is_submatrix;
  /// Offset for sub-matrix
  int offset;
  /// Sub-matrix size of each dimension
  int *matrixSize;
  /// Total size of the matrix (number of elements)
  int total_size;
  int last_raw_pos;
  /// Pointer to data
  GPUMirroredMemoryBlock<T> *data;
  /// Major type (only when numDim=2)
  CBLAS_ORDER major_order;
  /// For CUDA purposes
  bool use_cuda;
  /// Auxiliary coordinates array
  mutable int *aux_coords;

  /// Constructor... -> Integer array with the size of each dimension
  /*
    Matrix(int numDim, const int* dim, T* data_vector,
    CBLAS_ORDER major_order = CblasRowMajor);
  */
  /// Computes the position at data array given it coordinates
  int  computeRawPos(const int *coords) const;
  /// Indicates if the matrix is a sub-matrix
  bool getIsSubMatrix() const;
  /// Returns the data pointer for read and write
  T *getData() { return data->getPPALForReadAndWrite(); }
  /// Returns the data pointer for read
  const T *getData() const { return data->getPPALForRead(); }
  /// Returns the offset of first data value (sub-matrix)
  int getOffset() const { return offset; }
  /// Updates with the following coordinates vector
  static bool nextCoordVectorRowMajor(int *coords, const int *sizes,
				      int numDim);
  static bool nextCoordVectorColMajor(int *coords, const int *sizes,
				      int numDim);
  int getLastRawPos() const { return last_raw_pos; }
public:
  /********* Iterators for Matrix traversal *********/
  // forward declaration
  class const_iterator;
  class iterator {
    friend class const_iterator;
    friend class Matrix;
    Matrix *m;
    int raw_pos;
    int *coords;
    T *data;
    iterator(Matrix *m);
    iterator(Matrix *m, int raw_pos);
  public:
    iterator();
    iterator(const iterator &other);
    ~iterator();
    iterator &operator=(const iterator &other);
    bool      operator==(const iterator &other) const;
    bool      operator!=(const iterator &other) const;
    iterator &operator++();
    T &operator*();
    int getRawPos() const;
  };
  /*******************************************************/
  class const_iterator {
    friend class Matrix;
    const Matrix *m;
    int raw_pos;
    int *coords;
    const T *data;
    const_iterator(const Matrix *m);
    const_iterator(const Matrix *m, int raw_pos);
  public:
    const_iterator();
    const_iterator(const const_iterator &other);
    const_iterator(const iterator &other);
    /*const_iterator(const iterator &other);*/
    ~const_iterator();
    const_iterator &operator=(const const_iterator &other);
    const_iterator &operator=(const iterator &other);
    bool            operator==(const const_iterator &other) const;
    bool            operator==(const iterator &other) const;
    bool            operator!=(const const_iterator &other) const;
    bool            operator!=(const iterator &other) const;
    const_iterator &operator++();
    const T &operator*() const;
    int getRawPos() const;
  };
  
  /********** Constructors ***********/
  /// Full constructor given numDim, dim, default_value and major_order
  Matrix(int numDim, const int* dim, T default_value=T(),
	 CBLAS_ORDER major_order = CblasRowMajor);
  
  /// Constructor with T() values and CblasRowMajor order
  Matrix(int numDim, int d1, ...);
  
  /// Constructor given other matrix, it does a shallow or deep copy (clone). By
  /// default is a deep copy, some code pieces expect this behavior, don't
  /// change it.
  Matrix(Matrix<T> *other, bool clone=true);
  /// Sub-matrix constructor
  Matrix(Matrix<T> *other,
	 const int* coords, const int *sizes,
	 bool clone=true);
  /// Destructor
  virtual ~Matrix();
  /* Getters and setters */
  int getNumDim() const { return numDim; }
  const int *getDimPtr() const { return matrixSize; }
  const int *getStridePtr() const { return stride; }
  int getDimSize(int i) const { return matrixSize[i]; }
  int getStrideSize(int i) const { return stride[i]; }
  int size() const { return total_size; }
  CBLAS_ORDER getMajorOrder() const { return major_order; }
  void setUseCuda(bool v) { use_cuda = v; }
  bool isSimple() {
    bool is_simple=(!is_submatrix)&&(offset==0)&&(major_order==CblasRowMajor);
    int aux=1;
    for(int i=numDim-1; i>=0 && is_simple; --i) {
      is_simple=(stride[i]==aux);
      aux=aux*matrixSize[i];
    }
    return is_simple;
  }
  /**********************/
  iterator begin() { return iterator(this); }
  iterator end() { return iterator(this, last_raw_pos+1); }
  const_iterator begin() const { return const_iterator(this); }
  const_iterator end() const { return const_iterator(this, last_raw_pos+1); }

  /// Transposition
  Matrix<T>* transpose();
  /// Deep copy
  Matrix<T>* clone();
  /// Deep copy with different major_order
  Matrix<T> *clone(CBLAS_ORDER major_order);
  /// Shallow copy
  Matrix<T>* copy();
  T& operator[] (int i);
  const T& operator[] (int i) const;
  // Access to independent elements, one and two dimensions are special cases
  T& operator() (int i);
  T& operator() (int row, int col);
  T& operator() (int coord0, int coord1, int coord2, ...);
  T& operator() (int *coords, int sz);
  const T& operator() (int i) const;
  const T& operator() (int row, int col) const;
  const T& operator() (int coord0, int coord1, int coord2, ...) const;
  const T& operator() (int *coords, int sz) const;
  
  /// Function to obtain RAW access to data pointer. Be careful with it, because
  /// you are losing sub-matrix abstraction, and the major order.
  GPUMirroredMemoryBlock<T> *getRawDataAccess() { return data; }
  
  bool getCol(int col, T* vec, int vecsize);
  bool putCol(int col, T *vec, int vecsize);
  bool putSubCol(int col, int first_row, T *vec, int vecsize);

  void clamp(T lower, T upper);

  // Returns true if they have the same dimension
  bool sameDim(const Matrix *other) const;

  // Returns a new matrix with the sum, assuming they have the same dimension
  // Crashes otherwise
  Matrix<T>* addition(const Matrix<T> *other);

  // The same as addition but substracting
  Matrix<T>* substraction(const Matrix<T> *other);
  
  // Matrices must be NxK and KxM, the result is NxM
  Matrix<T>* multiply(const Matrix<T> *other) const;

  /**** BLAS OPERATIONS ****/

  // AXPY BLAS operation this = this + alpha * other
  void axpy(T alpha, const Matrix<T> *other);
  
  // GEMM BLAS operation this = alpha * op(A)*op(B) + beta*this
  void gemm(CBLAS_TRANSPOSE trans_A,
	    CBLAS_TRANSPOSE trans_B,
	    T alpha,
	    const Matrix<T> *otherA,
	    const Matrix<T> *otherB,
	    T beta);
  
  void scal(T value);
  
  T norm2() const;
  T min() const;
  T max() const;
  void minAndMax(T &min, T &max) const;
  
private:
  void allocate_memory(int size);
  void release_memory();
  void initialize(const int *dim);
};

template <typename T>
void Matrix<T>::initialize(const int *dim) {
  total_size=1;
  if (major_order == CblasRowMajor) {
    for(int i=numDim-1; i>=0; --i) {
      stride[i] = total_size;
      total_size *= dim[i];
      matrixSize[i] = dim[i];
    }
  }
  else {
    for(int i=0; i<numDim; ++i) {
      stride[i] = total_size;
      total_size *= dim[i];
      matrixSize[i] = dim[i];
    }
  }
  last_raw_pos = total_size-1;
}

/// Allocation of memory for data pointer. It is Referenced for sharing.
template <typename T>
void Matrix<T>::allocate_memory(int size) {
  data = new GPUMirroredMemoryBlock<T>(size, true);
  IncRef(data);
}

/// Release of the memory allocated for data pointer.
template <typename T>
void Matrix<T>::release_memory() {
  DecRef(data);
}

/// Constructor with given default_value initialization
template <typename T>
Matrix<T>::Matrix(int numDim,
		  const int* dim,
		  T default_value,
		  CBLAS_ORDER major_order) : numDim(numDim),
					     is_submatrix(false),
					     offset(0),
					     major_order(major_order),
					     use_cuda(false) {
  if (major_order == CblasColMajor && numDim != 2)
    ERROR_EXIT(128, "ColMajor order is only allowed when numDim=2\n");
  stride     = new int[numDim];
  matrixSize = new int[numDim];
  aux_coords = new int[numDim];
  initialize(dim);
  allocate_memory(total_size);
  T *d = data->getPPALForWrite();
  for (int i=0; i<total_size; ++i) d[i] = default_value;
}

/// Constructor for sub-matrix building
template <typename T>
Matrix<T>::Matrix(Matrix<T> *other,
		  const int* coords, const int *sizes,
		  bool clone) : numDim(other->numDim),
				is_submatrix(true),
				offset(0),
				major_order(other->major_order),
				use_cuda(other->use_cuda) {
  for (int i=0; i<numDim; i++) {
    if (sizes[i] + coords[i] > other->matrixSize[i])
      ERROR_EXIT3(128, "Size+coordinates are out of dimension size: %d+%d>%d\n",
		  sizes[i], coords[i], other->matrixSize[i]);
  }
  stride     = new int[numDim];
  matrixSize = new int[numDim];
  aux_coords = new int[numDim];
  if (clone) {
    initialize(sizes);
    allocate_memory(total_size);
    int other_offset = other->computeRawPos(coords);
    const T *other_data = other->data->getPPALForRead();
    for (int i=0; i<numDim; ++i) aux_coords[i] = 0;
    for (iterator it(begin()); it!=end(); ++it) {
      int other_raw_pos = other_offset + other->computeRawPos(aux_coords);
      *it = other_data[other_raw_pos];
      nextCoordVectorRowMajor(aux_coords, sizes, numDim);
    }
  }
  else {
    total_size = 1;
    for (int i=0; i<numDim; i++) {
      stride[i]     = other->stride[i];
      matrixSize[i] = sizes[i];
      total_size    = total_size * sizes[i];
    }
    offset = other->computeRawPos(coords);
    data   = other->data;
    IncRef(data);
    last_raw_pos = offset + other->computeRawPos(sizes);
  }
}


/// Constructor with T() default value initialization
template <typename T>
Matrix<T>::Matrix(int numDim, int d1, ...) : numDim(numDim),
					     is_submatrix(false),
					     offset(0),
					     major_order(CblasRowMajor) {
  int *dim   = new int[numDim];
  stride     = new int[numDim];
  matrixSize = new int[numDim];
  aux_coords = new int[numDim];
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
  T default_value=T();
  T *d = data->getPPALForWrite();
  for (int i=0; i<total_size; ++i) d[i] = default_value;
  delete[] dim;
}


/// Constructor for copy or clone other given matrix
template <typename T>
Matrix<T>::Matrix(Matrix<T> *other, bool clone) : numDim(other->numDim),
						  is_submatrix(false),
						  offset(0),
						  major_order(other->major_order),
						  use_cuda(other->use_cuda) {
  stride       = new int[numDim];
  matrixSize   = new int[numDim];
  aux_coords   = new int[numDim];
  total_size   = other->total_size;
  last_raw_pos = other->last_raw_pos;
  if (clone) {
    initialize(other->matrixSize);
    allocate_memory(total_size);
    iterator       this_it(begin());
    const_iterator other_it(other->begin());
    while(this_it != end()) {
      *this_it = *other_it;
      ++this_it;
      ++other_it;
    }
  }
  else {
    offset       = other->offset;
    is_submatrix = other->is_submatrix;
    data         = other->data;
    IncRef(data);
    for (int i=0; i<numDim; ++i) {
      stride[i]     = other->stride[i];
      matrixSize[i] = other->matrixSize[i];
    }
  }
}

template <typename T>
Matrix<T>::~Matrix() {
  release_memory();
  delete[] stride;
  delete[] matrixSize;
  delete[] aux_coords;
}

template<typename T>
Matrix<T> *Matrix<T>::transpose() {
  int *aux_matrix_size = new int[numDim];
  for (int i=0; i<numDim; ++i) aux_matrix_size[i] = matrixSize[numDim-i-1];
  Matrix<T> *resul = new Matrix<T>(numDim, aux_matrix_size, T(), major_order);
  const T *d = data->getPPALForRead();
  for (int i=0; i<numDim; ++i) aux_coords[i] = 0;
  for (iterator resul_it(resul->begin()); resul_it!=resul->end(); ++resul_it) {
    *resul_it = d[computeRawPos(aux_coords)];
    nextCoordVectorColMajor(aux_coords, matrixSize, numDim);
  }
  delete[] aux_matrix_size;
  return resul;
}

template<typename T>
Matrix<T> *Matrix<T>::clone(CBLAS_ORDER major_order) {
  Matrix<T> *resul;
  if (numDim != 2) ERROR_EXIT(128, "Major type not availabe when numDim!=2\n");
  if (this->major_order != major_order) {
    resul = new Matrix<T>(numDim, matrixSize, T(), major_order);
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
Matrix<T>* Matrix<T>::copy() {
  return new Matrix<T>(this,false);
}

template <typename T>
T& Matrix<T>::operator[] (int i) {
  return data->get(i);
}

template <typename T>
const T& Matrix<T>::operator[] (int i) const {
  return data->get(i);
}

template <typename T>
T& Matrix<T>::operator() (int i) {
  assert(numDim == 1);
  int raw_pos = computeRawPos(&i);
  return data->get(raw_pos);
}

template <typename T>
T& Matrix<T>::operator() (int row, int col) {
  assert(numDim == 2);
  int pos[2]={row,col};
  int raw_pos = computeRawPos(pos);
  return data->get(raw_pos);
}

template <typename T>
T& Matrix<T>::operator() (int coord0, int coord1, int coord2, ...) {
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
  return data->get(raw_pos);
}

template <typename T>
T& Matrix<T>::operator() (int *coords, int sz) {
  assert(numDim == sz);
  int raw_pos = computeRawPos(coords);
  return data->get(raw_pos);
}

template <typename T>
const T& Matrix<T>::operator() (int i) const {
  assert(numDim == 1);
  int raw_pos = computeRawPos(&i);
  return data->get(raw_pos);
}

template <typename T>
const T& Matrix<T>::operator() (int row, int col) const {
  assert(numDim == 2);
  int pos[2]={row,col};
  int raw_pos = computeRawPos(pos);
  return data->get(raw_pos);
}

template <typename T>
const T& Matrix<T>::operator() (int coord0, int coord1, int coord2, ...) const {
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
  return data->get(raw_pos);
}

template <typename T>
const T& Matrix<T>::operator() (int *coords, int sz) const {
  assert(numDim == sz);
  int raw_pos = computeRawPos(coords);
  return data->get(raw_pos);
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
void Matrix<T>::clamp(T lower, T upper) {
  T *d = data->getPPALForReadAndWrite();
  for (int i=0; i<total_size; ++i)
    d[i] = april_utils::clamp(d[i],lower,upper);
}

template <typename T>
bool Matrix<T>::sameDim(const Matrix<T> *other) const {
  if (numDim != other->numDim) return false;
  for (int i=0; i<numDim; ++i)
    if (matrixSize[i] != other->matrixSize[i]) return false;
  return true;
}

template <typename T>
Matrix<T>* Matrix<T>::addition(const Matrix<T> *other) {
  Matrix<T> *resul = this->clone();
  resul->axpy(1.0f, other);
  return resul;
}

template <typename T>
Matrix<T>* Matrix<T>::substraction(const Matrix<T> *other) {
  Matrix<T> *resul = this->clone();
  resul->axpy(-1.0f, other);
  return resul;
}

template <typename T>
Matrix<T>* Matrix<T>::multiply(const Matrix<T> *other) const {
  if (numDim != 2 || other->numDim != 2 ||
      matrixSize[1] != other->matrixSize[0]) return 0;
  int M = matrixSize[0];
  int K = matrixSize[1];
  int N = other->matrixSize[1];
  int dim[2] = {M, N};
  Matrix<T> *resul = new Matrix<T>(2,dim,T(),major_order);
  resul->gemm(CblasNoTrans, CblasNoTrans,
	      1.0f, this, other, 0.0f);
  return resul;
}

/**** BLAS OPERATIONS ****/

template <typename T>
void Matrix<T>::axpy(T alpha, const Matrix<T> *other) {
  if (size() != other->size())
    ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d",
		size(), other->size());
  if (major_order != other->major_order)
    ERROR_EXIT(128, "Matrices with different major orders");
  if (!is_submatrix && !other->is_submatrix)
    doSaxpy(total_size,
	    alpha, other->data, 0, 1,
	    data, 0, 1,
	    use_cuda);
  else {
    for (int i=0; i<numDim; ++i) aux_coords[i] = 0;
    if (major_order == CblasRowMajor) {
      do {
	int this_pos  = computeRawPos(aux_coords);
	int other_pos = other->computeRawPos(aux_coords);
	doSaxpy(matrixSize[numDim-1],
		alpha, other->data, other_pos, other->stride[numDim-1],
		data, this_pos, stride[numDim-1],
		use_cuda);
	aux_coords[numDim-1] = matrixSize[numDim-1]-1;
      } while(nextCoordVectorRowMajor(aux_coords, matrixSize, numDim));
    }
    else {
      do {
	int this_pos  = computeRawPos(aux_coords);
	int other_pos = other->computeRawPos(aux_coords);
	doSaxpy(matrixSize[0],
		alpha, other->data, other_pos, other->stride[0],
		data, this_pos, stride[0],
		use_cuda);
	aux_coords[0] = matrixSize[0]-1;
      } while(nextCoordVectorColMajor(aux_coords, matrixSize, numDim));
    }
  }
}

template <typename T>
void Matrix<T>::gemm(CBLAS_TRANSPOSE trans_A,
		     CBLAS_TRANSPOSE trans_B,
		     T alpha,
		     const Matrix<T> *otherA,
		     const Matrix<T> *otherB,
		     T beta) {
  if (numDim != 2 || otherA->numDim != 2 || otherB->numDim != 2)
    ERROR_EXIT(128,"Incorrect number of dimensions, only allowed for numDim=2");
  int row_idx_A = 0, col_idx_A = 1, row_idx_B = 0, col_idx_B = 1;
  if (trans_A == CblasTrans) april_utils::swap(row_idx_A, col_idx_A);
  if (trans_B == CblasTrans) april_utils::swap(row_idx_B, col_idx_B);
  if (matrixSize[0] != otherA->matrixSize[row_idx_A] ||
      matrixSize[1] != otherB->matrixSize[col_idx_B] ||
      otherA->matrixSize[col_idx_A] != otherB->matrixSize[row_idx_B])
    ERROR_EXIT6(128, "Incorrect matrixes dimensions: %dx%d + %dx%d * %dx%d\n",
		matrixSize[0], matrixSize[1],
		otherA->matrixSize[row_idx_A], otherA->matrixSize[col_idx_A],
		otherB->matrixSize[row_idx_B], otherB->matrixSize[col_idx_B]);
  if (major_order != otherA->major_order ||
      otherA->major_order != otherB->major_order)
    ERROR_EXIT(128, "Matrices with different major orders");
  
  int M=matrixSize[0], N=matrixSize[1], K=otherA->matrixSize[col_idx_A];
  int lda=(major_order==CblasRowMajor)?otherA->stride[0]:otherA->stride[1];
  int ldb=(major_order==CblasRowMajor)?otherB->stride[0]:otherB->stride[1];
  int ldc=(major_order==CblasRowMajor)?stride[0]:stride[1];
  doSgemm(major_order, trans_A, trans_B,
	  M, N, K,
	  alpha, otherA->data, lda,
	  otherB->data, ldb,
	  beta, data, ldc,
	  otherA->offset, otherB->offset, offset,
	  use_cuda);
}

template <typename T>
void Matrix<T>::scal(T value) {
  if (!is_submatrix) doSscal(total_size, value, data, 0, 1, use_cuda);
  else {
    for (int i=0; i<numDim; ++i) aux_coords[i] = 0;
    if (major_order == CblasRowMajor) {
      do {
	int pos  = computeRawPos(aux_coords);
	doSscal(matrixSize[numDim-1], value,
		data, pos, stride[numDim-1],
		use_cuda);
	aux_coords[numDim-1] = matrixSize[numDim-1]-1;
      } while(nextCoordVectorRowMajor(aux_coords, matrixSize, numDim));
    }
    else {
      do {
	int pos  = computeRawPos(aux_coords);
	doSscal(matrixSize[numDim-1], value,
		data, pos, stride[numDim-1],
		use_cuda);
	aux_coords[0] = matrixSize[0]-1;
      } while(nextCoordVectorColMajor(aux_coords, matrixSize, numDim));
    }
  }
}

template <typename T>
T Matrix<T>::norm2() const {
  T v;
  if (!is_submatrix) v=doSnrm2(total_size, data, 0, 1, use_cuda);
  else {
    v = 0.0f;
    for (int i=0; i<numDim; ++i) aux_coords[i] = 0;
    if (major_order == CblasRowMajor) {
      do {
	int pos   = computeRawPos(aux_coords);
	T aux = doSnrm2(matrixSize[numDim-1],
			data, pos, stride[numDim-1],
			use_cuda);
	v += aux*aux;
	aux_coords[numDim-1] = matrixSize[numDim-1]-1;
      } while(nextCoordVectorRowMajor(aux_coords, matrixSize, numDim));
    }
    else {
      do {
	int pos   = computeRawPos(aux_coords);
	T aux = doSnrm2(matrixSize[0],
			data, pos, stride[0],
			use_cuda);
	v += aux*aux;
	aux_coords[0] = matrixSize[0]-1;
      } while(nextCoordVectorColMajor(aux_coords, matrixSize, numDim));
    }
    v = (T)sqrtf(v);
  }
  return v;
}

template <typename T>
T Matrix<T>::min() const {
  const_iterator it(begin());
  T min = *it;
  for (; it!=end(); ++it) if (*it < min) min = *it;
  return min;
}

template <typename T>
T Matrix<T>::max() const {
  const_iterator it(begin());
  T max = *it;
  for (; it!=end(); ++it) if (*it > max) max = *it;
  return max;
}

template <typename T>
void Matrix<T>::minAndMax(T &min, T &max) const {
  const_iterator it(begin());
  min = *it;
  max = *it;
  for (; it!=end(); ++it) {
    if (*it < min) min = *it;
    if (*it > max) max = *it;
  }
}

/***** PRIVATE METHODS *****/

template <typename T>
bool Matrix<T>::nextCoordVectorRowMajor(int *coords, const int *sizes,
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
bool Matrix<T>::nextCoordVectorColMajor(int *coords, const int *sizes,
					int numDim) {
  int j = 0;
  do {
    coords[j] = (coords[j]+1) % sizes[j];
  } while(j<numDim-1 && coords[j++] == 0);
  if (j == numDim-1 && coords[numDim-1] == 0) return false;
  return true;
}

template <typename T>
int Matrix<T>::computeRawPos(const int *coords) const {
  int raw_pos;
  switch(numDim) {
  case 1:
    assert(coords[0] < matrixSize[0]);
    raw_pos = coords[0];
    break;
  case 2:
    assert(coords[0] < matrixSize[0]);
    assert(coords[1] < matrixSize[1]);
    raw_pos = coords[0]*stride[0]+coords[1]*stride[1];
    break;
  default:
    raw_pos=0;
    for(int i=0; i<numDim; i++) {
      assert(coords[i] < matrixSize[i]);
      raw_pos += stride[i]*coords[i];
    }
  }
  return raw_pos + offset;
}

template <typename T>
bool Matrix<T>::getIsSubMatrix() const {
  return is_submatrix;
}

/***** ITERATORS *****/

template <typename T>
Matrix<T>::iterator::iterator(Matrix *m) : m(m), raw_pos(0) {
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
  raw_pos = m->getOffset();
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::iterator::iterator(Matrix *m, int raw_pos) :
  m(m), raw_pos(raw_pos) {
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::iterator::iterator() : m(0), raw_pos(0), coords(0) { }

template <typename T>
Matrix<T>::iterator::iterator(const iterator &other) :
  m(other.m),
  raw_pos(other.raw_pos) {
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::iterator:: ~iterator() {
  delete[] coords;
  // if (m) DecRef(m);
}

template <typename T>
typename Matrix<T>::iterator &Matrix<T>::iterator::operator=(const Matrix<T>::iterator &other) {
  delete[] coords;
  // if (m) DecRef(m);
  m = other.m;
  // IncRef(m);
  raw_pos = other.raw_pos;
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
  data = m->getData();
  return *this;
}

template <typename T>
bool Matrix<T>::iterator::operator==(const Matrix<T>::iterator &other) const {
  return m==other.m && raw_pos == other.raw_pos;
}

template <typename T>
bool Matrix<T>::iterator::operator!=(const Matrix<T>::iterator &other) const {
  return !( (*this) == other );
}

template <typename T>
typename Matrix<T>::iterator &Matrix<T>::iterator::operator++() {
  if (m->getIsSubMatrix() || m->getMajorOrder()==CblasColMajor) {
    const int *dims    = m->getDimPtr();
    // const int *strides = m->getStridePtr();
    if (!Matrix<T>::nextCoordVectorRowMajor(coords, dims, m->getNumDim()))
      raw_pos = m->getLastRawPos()+1;
    else raw_pos = m->computeRawPos(coords);
  }
  else ++raw_pos;
  return *this;
}

template <typename T>
T &Matrix<T>::iterator::operator*() {
  return data[raw_pos];
}

template <typename T>
int Matrix<T>::iterator::getRawPos() const {
  return raw_pos;
}

/*******************************************************************/

template <typename T>
Matrix<T>::const_iterator::const_iterator(const Matrix *m) : m(m), raw_pos(0) {
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
  raw_pos = m->getOffset();
  data = m->getData();
}

template <typename T>
Matrix<T>::const_iterator::const_iterator(const Matrix *m, int raw_pos) :
  m(m), raw_pos(raw_pos) {
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = 0;
  data = m->getData();
}

template <typename T>
Matrix<T>::const_iterator::const_iterator() : m(0), raw_pos(0), coords(0) { }

template <typename T>
Matrix<T>::const_iterator::const_iterator(const Matrix<T>::const_iterator &other) :
  m(other.m),
  raw_pos(other.raw_pos) {
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
  data = m->getData();
}

template <typename T>
Matrix<T>::const_iterator::const_iterator(const Matrix<T>::iterator &other) :
  m(other.m),
  raw_pos(other.raw_pos) {
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
  data = m->getData();
}

/*
template <typename T>
Matrix<T>::const_iterator::const_iterator(const iterator &other) :
  m(other.m),
  raw_pos(m.raw_pos) {
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
  data = m->getData();
}
*/

template <typename T>
Matrix<T>::const_iterator::~const_iterator() {
  delete[] coords;
}

template <typename T>
typename Matrix<T>::const_iterator &Matrix<T>::const_iterator::operator=(const typename Matrix<T>::const_iterator &other) {
  delete[] coords;
  m = other.m;
  raw_pos = other.raw_pos;
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
  return *this;
}

template <typename T>
typename Matrix<T>::const_iterator &Matrix<T>::const_iterator::operator=(const typename Matrix<T>::iterator &other) {
  delete[] coords;
  m = other.m;
  raw_pos = other.raw_pos;
  coords = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) coords[i] = other.coords[i];
  return *this;
}

template <typename T>
bool Matrix<T>::const_iterator::operator==(const Matrix<T>::const_iterator &other) const {
  return m==other.m && raw_pos == other.raw_pos;
}

template <typename T>
bool Matrix<T>::const_iterator::operator==(const Matrix<T>::iterator &other) const {
  return m==other.m && raw_pos == other.raw_pos;
}

template <typename T>
bool Matrix<T>::const_iterator::operator!=(const Matrix<T>::const_iterator &other) const {
  return !( (*this) == other );
}

template <typename T>
bool Matrix<T>::const_iterator::operator!=(const Matrix<T>::iterator &other) const {
  return !( (*this) == other );
}

template <typename T>
typename Matrix<T>::const_iterator &Matrix<T>::const_iterator::operator++() {
  if (m->getIsSubMatrix() || m->getMajorOrder()==CblasColMajor) {
    const int *dims = m->getDimPtr();
    if (!Matrix<T>::nextCoordVectorRowMajor(coords, dims, m->getNumDim()))
      raw_pos = m->getLastRawPos()+1;
    else raw_pos = m->computeRawPos(coords);
  }
  else ++raw_pos;
  return *this;
}

template <typename T>
const T &Matrix<T>::const_iterator::operator*() const {
  return data[raw_pos];
}

template <typename T>
int Matrix<T>::const_iterator::getRawPos() const {
  return raw_pos;
}

#endif // MATRIX_H
