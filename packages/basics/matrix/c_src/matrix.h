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

#include "cblas_headers.h"
#include "wrapper.h"
#include "gpu_mirrored_memory_block.h"
#include "referenced.h"
#include "constants.h"
#include "clamp.h"
#include "aligned_memory.h"
#include <cassert>
#include <cstdarg>
#include <new> // surprisingly, placement new doesn't work without this

template <typename T>
class Matrix : public Referenced {
  /// Number of dimensions
  int numDim;
  /// Size of each dimension
  int *matrixSize;
  /// Indication for sub-matrices
  bool is_submatrix;
  /// Last raw position (for iterator)
  int last_raw_position;
  /// Offset for sub-matrix
  int offset;
  /// Sub-matrix size of each dimension
  int *subMatrixSize;
  /// Total size of the matrix (number of elements)
  int total_size;
  /// Pointer to data
  GPUMirroredMemoryBlock<T> *data;
  /// Major type (only when numDim=2)
  CBLAS_ORDER major_order;
  /// Constructor... -> Integer array with the size of each dimension
  /*
    Matrix(int numDim, const int* dim, T* data_vector,
    CBLAS_ORDER major_order = CblasRowMajor);
  */
  int  computeRawPos(const int *dim_pos) const;
  bool getIsSubMatrix() const;
  int  getEndRawPos() const;
  T *getData() { return data->getPPALForReadAndWrite(); }
  const T *getData() const { return data->getPPALForRead(); }
public:
  /*******************************************************/
  class const_iterator;
  class iterator {
    friend class const_iterator;
    friend class Matrix;
    Matrix *m;
    int raw_pos;
    int *dim_pos;
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
  };
  /*******************************************************/
  class const_iterator {
    friend class Matrix;
    const Matrix *m;
    int raw_pos;
    int *dim_pos;
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
  };
  /*******************************************************/
  Matrix(int numDim, const int* dim, T default_value=T(),
	 CBLAS_ORDER major_order = CblasRowMajor);
  /// Constructor with void values and CblasRowMajor
  Matrix(int numDim, int d1, ...);
  Matrix(Matrix<T> *other, bool clone=true);
  /*
 /// Sub-matrix constructor
 Matrix(Matrix<T> *other, const int* pos, const int *size, bool clone=true,
 CBLAS_ORDER major_order = CblasRowMajor);
  */
  // Destructor...
  virtual ~Matrix();
  /* Getters and setters */
  int getOffset() const { return offset; }
  int getNumDim() const { return numDim; }
  const int *getMatrixDimPtr() const { return subMatrixSize; }
  int getMatrixDimSize(int i) const { return subMatrixSize[i]; }
  int getSize() const { return total_size; }
  int size() const { return total_size; }
  CBLAS_ORDER getMajorType() const { return major_order; }
  /**********************/
  iterator begin() { return iterator(this); }
  iterator end() { return iterator(this, getEndRawPos()); }
  const_iterator begin() const { return const_iterator(this); }
  const_iterator end() const { return const_iterator(this, getEndRawPos()); }

  /// Changes the major type: CblasRowMajor, CblasColMajor
  void setMajorType(CBLAS_ORDER major_order, bool transpose);
  /// Deep copy
  Matrix<T>* clone();
  /// Deep copy with different major_order
  Matrix<T> *clone(CBLAS_ORDER major_order, bool transpose);
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
  
  bool getCol(int col, T* vec, int vecsize);
  bool putCol(int col, T *vec, int vecsize);
  bool putSubCol(int col, int first_row, T *vec, int vecsize);

  void clamp(T lower, T upper);

  // Returns true if they have the same dimension
  bool sameDim(const Matrix *other) const;

  // Returns a new matrix with the sum, assuming they have the same dimension
  // Crashes otherwise
  Matrix<T>* addition(const Matrix<T> *other, float alpha=1.0f);

  // The same as the previous one but accumulates the result in the matrix
  void accumulate_addition(const Matrix<T> *other, float alpha=1.0f);

  // The same as addition but substracting
  Matrix<T>* substraction(const Matrix<T> *other);

  // The same as the previous one but accumulates the result in the matrix
  void accumulate_substraction(const Matrix<T> *other);

  // Matrices must be NxK and KxM, the result is NxM
  Matrix<T>* multiply(const Matrix<T> *other) const;

  // Matrices must be NxK and KxM, the result is NxM (this pointer)
  void accumulate_multiply(float alpha,
			   const Matrix<T> *otherA,
			   const Matrix<T> *otherB,
			   float beta);
  
  void multiply_by_scalar(T value);
  
  T norm2() const;
  T min() const;
  T max() const;
  void minAndMax(T &min, T &max) const;
private:
  void allocate_memory(int size);
  void release_memory();
};

template <typename T>
void Matrix<T>::allocate_memory(int size) {
  data = new GPUMirroredMemoryBlock<T>(size, true);
  IncRef(data);
}

template <typename T>
void Matrix<T>::release_memory() {
  DecRef(data);
}

template <typename T>
Matrix<T>::Matrix(int numDim,
		  const int* dim,
		  T default_value,
		  CBLAS_ORDER major_order) : numDim(numDim),
					    is_submatrix(false),
					    offset(0),
					    major_order(major_order) {
  matrixSize=new int[numDim];
  subMatrixSize=new int[numDim];
  total_size=1;
  for(int i=0; i<numDim; i++) {
    total_size *= dim[i];
    matrixSize[i] = subMatrixSize[i] = dim[i];
  }
  
  allocate_memory(total_size);
  
  for (int i=0; i<total_size; ++i)
    data->get(i) = default_value;
  
  last_raw_position = total_size-1;
}

/*
  template <typename T>
  Matrix<T>::Matrix(int numDim, const int* dim, T* data_vector,
  CBLAS_ORDER major_order) : numDim(numDim),
  major_order(major_order) {
  matrixSize=new int[numDim];
  size=1;
  for(int i=0; i<numDim; i++) {
  size *= dim[i];
  matrixSize[i] = dim[i];
  }
  data = data_vector;
  }
*/

template <typename T>
Matrix<T>::Matrix(int numDim, int d1, ...) : numDim(numDim),
					     is_submatrix(false),
					     offset(0),
					     major_order(CblasRowMajor) {
  matrixSize = new int[numDim];
  subMatrixSize = new int[numDim];

  va_list ap;
  va_start(ap, d1);
  matrixSize[0] = d1;
  total_size = d1;
  for (int i=1; i<numDim; i++) {
    int di = va_arg(ap, int);
    matrixSize[i] = subMatrixSize[i] = di;
    total_size *= di;
  }
  va_end(ap);

  allocate_memory(total_size); // init with default value for type T
  T default_value=T();
  for (int i=0; i<total_size; ++i)
    data->get(i) = default_value;

  last_raw_position = total_size-1;
}


template <typename T>
Matrix<T>::Matrix(Matrix<T> *other, bool clone) : numDim(other->numDim),
						  is_submatrix(false),
						  offset(0),
						  major_order(other->major_order) {
  matrixSize = new int[numDim];
  subMatrixSize = new int[numDim];
  if (clone) {
    for (int i=0; i<numDim; i++) {
      matrixSize[i] = subMatrixSize[i] = other->subMatrixSize[i];
    }
    total_size = other->total_size;


    allocate_memory(total_size);
    doScopy(total_size, other->data, 0, 1, data, 0, 1, other->data->getCudaFlag());
  }
  else {
    data = other->data;
    IncRef(data);
  }
  
  last_raw_position = total_size-1;
}

template <typename T>
Matrix<T>::~Matrix() {
  release_memory();
  delete[] matrixSize;
  delete[] subMatrixSize;
}

template<typename T>
Matrix<T> *Matrix<T>::clone(CBLAS_ORDER major_order, bool transpose) {
  Matrix<T> *resul;
  if (numDim != 2) ERROR_EXIT(128, "Major type not availabe when numDim!=2\n");
  if (this->major_order != major_order && transpose) {
    resul = new Matrix<T>(numDim, matrixSize, T(), major_order);
    if (transpose)
      for (int i=0; i<matrixSize[0]; ++i)
	for (int j=0; j<matrixSize[1]; ++j)
	  (*resul)(i,j) = (*this)(i,j);
  }
  else {
    resul = new Matrix<T>(this);
    resul->major_order = major_order;
  }
  return resul;
}

template <typename T>
Matrix<T>* Matrix<T>::clone() {
  return new Matrix<T>(this);
}

template <typename T>
Matrix<T>* Matrix<T>::copy() {
  return new Matrix<T>(this, false);
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
const T& Matrix<T>::operator() (int *coords, int sz) const {
  assert(numDim == sz);
  int raw_pos = computeRawPos(coords);
  return data->get(raw_pos);
}

template <typename T>
bool Matrix<T>::getCol(int col, T* vec, int vecsize) {
  // If it is not a 2D matrix, error
  if (numDim != 2) 
    return false;
  // If the column is out of range, error
  if ((col < 0) || (col >= matrixSize[1]))
    return false;
  // If the array length is different to the size of the matrix columns, error
  if (vecsize != matrixSize[0]) 
    return false;
  
  if (major_order == CblasRowMajor)
    for (int row = 0; row < matrixSize[0]; row++) {
      vec[row] = data->get(matrixSize[1]*row+col);
    }
  else {
    int colpos = matrixSize[0]*col;
    for (int row = 0; row < matrixSize[0]; row++) {
      vec[row] = data->get(colpos + row);
    }
  }
  return true;
}

template <typename T>
bool Matrix<T>::putCol(int col, T* vec, int vecsize) {
  // If it is not a 2D matrix, error
  if (numDim != 2) 
    return false;
  // If the column is out of range, error
  if ((col < 0) || (col >= matrixSize[1]))
    return false;
  // If the array length is different to the size of the matrix columns, error
  if (vecsize != matrixSize[0]) 
    return false;
  
  if (major_order == CblasRowMajor)
    for (int row = 0; row < matrixSize[0]; row++) {
      data->get(matrixSize[1]*row+col) = vec[row];
    }
  else {
    int colpos = matrixSize[0]*col;
    for (int row = 0; row < matrixSize[0]; row++) {
      data->get(colpos + row) = vec[row];
    }
  }

  return true;
}

template <typename T>
bool Matrix<T>::putSubCol(int col, int first_row, T* vec, int vecsize) {
  // If it is not a 2D matrix, error
  if (numDim != 2) 
    return false;
  // If the column is out of range, error
  if ((col < 0) || (col >= matrixSize[1]))
    return false;
  // If the first row is out of range, error
  if ((first_row < 0) || (first_row >= matrixSize[0]))
    return false;
  // If the array is out of range, error
  if ((first_row < 0) || (first_row+vecsize > matrixSize[0]))
    return false;

  if (major_order == CblasRowMajor)
    for (int row = first_row; row < first_row+vecsize; row++) {
      data[matrixSize[1]*row+col] = vec[row];
    }
  else {
    int colpos = matrixSize[0]*col;
    for (int row = first_row; row < first_row+vecsize; row++) {
      data[colpos + row] = vec[row];
    }
  }
  
  return true;
}

template <typename T>
void Matrix<T>::clamp(T lower, T upper) {
  float *d = data->getPPALForReadAndWrite();
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
Matrix<T>* Matrix<T>::addition(const Matrix<T> *other, float alpha) {
  Matrix<T> *resul = new Matrix<T>(this);
  resul->accumulate_addition(other, alpha);
  return resul;
}

template <typename T>
void Matrix<T>::accumulate_addition(const Matrix<T> *other, float alpha) {
  doSaxpy(total_size,
	  alpha, other->data, 0, 1,
	  data, 0, 1,
	  data->getCudaFlag());
}

template <typename T>
Matrix<T>* Matrix<T>::substraction(const Matrix<T> *other) {
  Matrix<T> *resul = new Matrix<T>(this);
  resul->accumulate_substraction(other);
  return resul;
}

template <typename T>
void Matrix<T>::accumulate_substraction(const Matrix<T> *other) {
  accumulate_addition(other, -1.0f);
}

template <typename T>
Matrix<T>* Matrix<T>::multiply(const Matrix<T> *other) const {
  if (numDim != 2 || other->numDim != 2 ||
      matrixSize[1] != other->matrixSize[0]) return 0;
  int N = matrixSize[0];
  int K = matrixSize[1];
  int M = other->matrixSize[1];
  int dim[2] = {N, M};
  Matrix<T> *resul = new Matrix<T>(2,dim);
  resul->accumulate_multiply(1.0f, this, other, 0.0f);
  return resul;
}

template <typename T>
void Matrix<T>::accumulate_multiply(float alpha,
				    const Matrix<T> *otherA,
				    const Matrix<T> *otherB,
				    float beta) {
  if (numDim != 2 || otherA->numDim != 2 || otherB->numDim != 2 ||
      matrixSize[0] != otherA->matrixSize[0] ||
      matrixSize[1] != otherB->matrixSize[1] ||
      otherA->matrixSize[1] != otherB->matrixSize[0] ||
      major_order != otherA->major_order ||
      otherA->major_order != otherB->major_order)
    ERROR_EXIT6(128, "Incorrect matrixes dimensions or different major types: "
		"%dx%d + %dx%d * %dx%d\n",
		matrixSize[0], matrixSize[1],
		otherA->matrixSize[0], otherA->matrixSize[1],
		otherB->matrixSize[0], otherB->matrixSize[1]);
  int N=matrixSize[0], M=matrixSize[1], K=otherA->matrixSize[1];
  doSgemm(major_order, CblasNoTrans, CblasNoTrans,
	  matrixSize[0], matrixSize[1], otherA->matrixSize[1],
	  alpha, otherA->data, (major_order==CblasRowMajor)?K:N,
	  otherB->data, (major_order==CblasRowMajor)?M:K,
	  beta, data, (major_order==CblasRowMajor)?M:N,
	  0, 0, 0,
	  data->getCudaFlag());
}

template <typename T>
void Matrix<T>::multiply_by_scalar(T value) {
  doSscal(total_size, value, data, 0, 1, data->getCudaFlag());
}

template <typename T>
T Matrix<T>::norm2() const {
  return doSnrm2(total_size, data, 0, 1, data->getCudaFlag());
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

template <typename T>
int Matrix<T>::computeRawPos(const int *dim_pos) const {
  int raw_pos;
  switch(numDim) {
  case 1:
    assert(dim_pos[0] < subMatrixSize[0]);
    raw_pos = dim_pos[0];
    break;
  case 2:
    assert(dim_pos[0] < subMatrixSize[0]);
    assert(dim_pos[1] < subMatrixSize[1]);
    if (major_order == CblasRowMajor)
      raw_pos = dim_pos[0]*matrixSize[1]+dim_pos[1];
    else raw_pos = dim_pos[1]*matrixSize[0]+dim_pos[0];
    break;
  default:
    assert(dim_pos[0] < subMatrixSize[0]);
    assert(dim_pos[1] < subMatrixSize[1]);
    assert(dim_pos[2] < subMatrixSize[2]);
    raw_pos=(dim_pos[0]*matrixSize[1]+dim_pos[1])*matrixSize[2]+dim_pos[2];
    for(int i=3; i<numDim; i++) {
      assert(dim_pos[i] < subMatrixSize[i]);
      raw_pos *= matrixSize[i];
      raw_pos += dim_pos[i];
    }
  }
  return raw_pos + offset;
}

template <typename T>

bool Matrix<T>::getIsSubMatrix() const {
  return is_submatrix;
}

template <typename T>
int Matrix<T>::getEndRawPos() const {
  return last_raw_position + 1;
}

/*******************************************************************/

template <typename T>
Matrix<T>::iterator::iterator(Matrix *m) : m(m), raw_pos(0) {
  dim_pos = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) dim_pos[i] = 0;
  raw_pos = m->getOffset();
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::iterator::iterator(Matrix *m, int raw_pos) :
  m(m), raw_pos(raw_pos) {
  dim_pos = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) dim_pos[i] = 0;
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::iterator::iterator() : m(0), raw_pos(0), dim_pos(0) { }

template <typename T>
Matrix<T>::iterator::iterator(const iterator &other) :
  m(other.m),
  raw_pos(other.raw_pos) {
  dim_pos = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) dim_pos[i] = other.dim_pos[i];
  // IncRef(m);
  data = m->getData();
}

template <typename T>
Matrix<T>::iterator:: ~iterator() {
  delete[] dim_pos;
  // if (m) DecRef(m);
}

template <typename T>
typename Matrix<T>::iterator &Matrix<T>::iterator::operator=(const Matrix<T>::iterator &other) {
  delete[] dim_pos;
  // if (m) DecRef(m);
  m = other.m;
  // IncRef(m);
  raw_pos = other.raw_pos;
  dim_pos = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) dim_pos[i] = other.dim_pos[i];
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
  if (m->getIsSubMatrix()) {
    const int *dims = m->getMatrixDimPtr();
    int j = m->getNumDim();
    do {
      --j;
      dim_pos[j] = (dim_pos[j]+1) % dims[j];
    } while(dim_pos[j] == 0);
    if (j == 0 && dim_pos[0] == 0) raw_pos = m->getEndRawPos();
  }
  else ++raw_pos;
  return *this;
}

template <typename T>
T &Matrix<T>::iterator::operator*() {
  return data[raw_pos];
}

/*******************************************************************/

template <typename T>
Matrix<T>::const_iterator::const_iterator(const Matrix *m) : m(m), raw_pos(0) {
  dim_pos = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) dim_pos[i] = 0;
  raw_pos = m->getOffset();
  data = m->getData();
}

template <typename T>
Matrix<T>::const_iterator::const_iterator(const Matrix *m, int raw_pos) :
  m(m), raw_pos(raw_pos) {
  dim_pos = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) dim_pos[i] = 0;
  data = m->getData();
}

template <typename T>
Matrix<T>::const_iterator::const_iterator() : m(0), raw_pos(0), dim_pos(0) { }

template <typename T>
Matrix<T>::const_iterator::const_iterator(const Matrix<T>::const_iterator &other) :
  m(other.m),
  raw_pos(other.raw_pos) {
  dim_pos = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) dim_pos[i] = other.dim_pos[i];
  data = m->getData();
}

template <typename T>
Matrix<T>::const_iterator::const_iterator(const Matrix<T>::iterator &other) :
  m(other.m),
  raw_pos(other.raw_pos) {
  dim_pos = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) dim_pos[i] = other.dim_pos[i];
  data = m->getData();
}

/*
template <typename T>
Matrix<T>::const_iterator::const_iterator(const iterator &other) :
  m(other.m),
  raw_pos(m.raw_pos) {
  dim_pos = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) dim_pos[i] = other.dim_pos[i];
  data = m->getData();
}
*/

template <typename T>
Matrix<T>::const_iterator::~const_iterator() {
  delete[] dim_pos;
}

template <typename T>
typename Matrix<T>::const_iterator &Matrix<T>::const_iterator::operator=(const typename Matrix<T>::const_iterator &other) {
  delete[] dim_pos;
  m = other.m;
  raw_pos = other.raw_pos;
  dim_pos = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) dim_pos[i] = other.dim_pos[i];
  return *this;
}

template <typename T>
typename Matrix<T>::const_iterator &Matrix<T>::const_iterator::operator=(const typename Matrix<T>::iterator &other) {
  delete[] dim_pos;
  m = other.m;
  raw_pos = other.raw_pos;
  dim_pos = new int[m->getNumDim()];
  for (int i=0; i<m->getNumDim(); ++i) dim_pos[i] = other.dim_pos[i];
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
  if (m->getIsSubMatrix()) {
    const int *dims = m->getMatrixDimPtr();
    int j = m->getNumDim();
    do {
      --j;
      dim_pos[j] = (dim_pos[j]+1) % dims[j];
    } while(dim_pos[j] == 0);
    if (j == 0 && dim_pos[0] == 0) raw_pos = m->getEndRawPos();
    else raw_pos = m->computeRawPos(dim_pos);
  }
  else ++raw_pos;
  return *this;
}

template <typename T>
const T &Matrix<T>::const_iterator::operator*() const {
  return data[raw_pos];
}

#endif // MATRIX_H
