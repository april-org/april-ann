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
public:
  /// Number of dimensions
  int numDim;
  /// Size of each dimension
  int *matrixSize;
  /// Total size of the matrix (number of elements)
  int size;
  /// Pointer to data
  GPUMirroredMemoryBlock<T> *data;
  /// Major type (for bidimensional matrixes)
  CBLAS_ORDER major_type;
  /// Constructor... -> Integer array with the size of each dimension
  /*
    Matrix(int numDim, const int* dim, T* data_vector,
    CBLAS_ORDER major_type = CblasRowMajor);
  */
  Matrix(int numDim, const int* dim, T default_value=T(),
	 CBLAS_ORDER major_type = CblasRowMajor);
  /// Constructor with void values and CblasRowMajor
  Matrix(int numDim, int d1, ...);
  Matrix(Matrix<T> *other, bool clone=true);
  /*
 /// Sub-matrix constructor
 Matrix(Matrix<T> *other, const int* pos, const int *size, bool clone=true,
 CBLAS_ORDER major_type = CblasRowMajor);
  */
  // Destructor...
  virtual ~Matrix();
  /// Changes the major type: CblasRowMajor, CblasColMajor
  void setMajorType(CBLAS_ORDER major_type, bool transpose);
  /// Deep copy
  Matrix<T>* clone();
  /// Deep copy with different major_type
  Matrix<T> *clone(CBLAS_ORDER major_type, bool transpose);
  /// Shallow copy
  Matrix<T>* copy();
  // Access to independent elements
  T& operator() (int i);
  T& operator() (int row, int col);
  T& operator() (int coord0, int coord1, int coord2, ...);
  const T& operator() (int i) const;
  const T& operator() (int row, int col) const;
  const T& operator() (int coord0, int coord1, int coord2, ...) const;

  T *getData();
  
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
  
  T norm2();
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
Matrix<T>::Matrix(int numDim, const int* dim, T default_value,
		  CBLAS_ORDER major_type) : numDim(numDim),
					    major_type(major_type) {
  matrixSize=new int[numDim];
  size=1;
  for(int i=0; i<numDim; i++) {
    size *= dim[i];
    matrixSize[i] = dim[i];
  }

  allocate_memory(size);
  
  for (int i=0; i<size; ++i)
    data->get(i) = default_value;
					    }

/*
  template <typename T>
  Matrix<T>::Matrix(int numDim, const int* dim, T* data_vector,
  CBLAS_ORDER major_type) : numDim(numDim),
  major_type(major_type) {
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
					     major_type(CblasRowMajor) {
  matrixSize = new int[numDim];

  va_list ap;
  va_start(ap, d1);
  matrixSize[0] = d1;
  size = d1;
  for (int i=1; i<numDim; i++) {
    int di = va_arg(ap, int);
    matrixSize[i] = di;
    size *= di;
  }
  va_end(ap);

  allocate_memory(size); // init with default value for type T
  T default_value=T();
  for (int i=0; i<size; ++i)
    data->get(i) = default_value;
}


template <typename T>
Matrix<T>::Matrix(Matrix<T> *other, bool clone) :
  numDim(other->numDim),
  major_type(other->major_type) {
  matrixSize = new int[numDim];
  for (int i=0; i<numDim; i++) {
    matrixSize[i] = other->matrixSize[i];
  }
  size = other->size;

  if (clone) {
    allocate_memory(size);
    doScopy(size, other->data, 0, 1, data, 0, 1, other->data->getCudaFlag());
  }
  else {
    data = other->data;
    IncRef(data);
  }
}

template <typename T>
Matrix<T>::~Matrix() {
  release_memory();
  delete[] matrixSize;
}

template<typename T>
Matrix<T> *Matrix<T>::clone(CBLAS_ORDER major_type, bool transpose) {
  Matrix<T> *resul;
  if (numDim != 2) ERROR_EXIT(128, "Major type not availabe when numDim!=2\n");
  if (this->major_type != major_type && transpose) {
    resul = new Matrix<T>(numDim, matrixSize, major_type);
    if (transpose)
      for (int i=0; i<matrixSize[0]; ++i)
	for (int j=0; i<matrixSize[1]; ++j)
	  (*resul)(i,j) = (*this)(i,j);
  }
  else {
    resul = new Matrix<T>(this);
    resul->major_type = major_type;
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
T& Matrix<T>::operator() (int i) {
  return data->get(i);
}

template <typename T>
T& Matrix<T>::operator() (int row, int col) {
  if (major_type == CblasRowMajor)
    return data->get(row*matrixSize[1]+col);
  else return data->get(col*matrixSize[0]+row);
}

template <typename T>
T& Matrix<T>::operator() (int coord0, int coord1, int coord2, ...) {
  int index = (coord0*matrixSize[1]+coord1)*matrixSize[2]+coord2;
  va_list ap;
  va_start(ap, coord2);
  for(int i=3; i<numDim; i++) {
    index *= matrixSize[i];
    int coordn = va_arg(ap, int);
    index += coordn;
  }
  va_end(ap);

  return data->get(index);
}

template <typename T>
const T& Matrix<T>::operator() (int i) const {
  return data->get(i);
}

template <typename T>
const T& Matrix<T>::operator() (int row, int col) const {
  if (major_type == CblasRowMajor)
    return data->get(row*matrixSize[1]+col);
  else return data->get(col*matrixSize[0]+row);
}

template <typename T>
const T& Matrix<T>::operator() (int coord0, int coord1, int coord2, ...) const {
  int index = (coord0*matrixSize[1]+coord1)*matrixSize[2]*coord2;
  va_list ap;
  va_start(ap, coord2);
  for(int i=3; i<numDim; i++) {
    index *= matrixSize[i];
    int coordn = va_arg(ap, int);
    index += coordn;
  }
  va_end(ap);

  return data->get(index);
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
  
  if (major_type == CblasRowMajor)
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
  
  if (major_type == CblasRowMajor)
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

  if (major_type == CblasRowMajor)
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
  for (int i=0; i<size; ++i)
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
  doSaxpy(size, alpha, other->data, 0, 1, data, 0, 1,
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
  int dim[2];
  dim[0] = N; dim[1] = M;
  Matrix<T> *resul = new Matrix<T>(2,dim);
  resul->accumulate_multiply(0.0f, this, other, 1.0f);
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
      major_type != otherA->major_type ||
      otherA->major_type != otherB->major_type)
    ERROR_EXIT6(128, "Incorrect matrixes dimensions or different major types: "
		"%dx%d + %dx%d * %dx%d\n",
		matrixSize[0], matrixSize[1],
		otherA->matrixSize[0], otherA->matrixSize[1],
		otherB->matrixSize[0], otherB->matrixSize[1]);
  doSgemm(major_type, CblasNoTrans, CblasNoTrans,
	  matrixSize[0], matrixSize[1], otherA->matrixSize[1],
	  alpha, data, 0,
	  otherA->data, 0,
	  beta,
	  otherB->data, 0,
	  0, 0, 0,
	  data->getCudaFlag());
}

template <typename T>
void Matrix<T>::multiply_by_scalar(T value) {
  doSscal(size, value, data, 0, 0, data->getCudaFlag());
}

template <typename T>
T Matrix<T>::norm2() {
  return doSnrm2(size, data, 0, 0, data->getCudaFlag());
}

template <typename T>
T *Matrix<T>::getData() {
  return data->getPPALForReadAndWrite();
}
#endif // MATRIX_H
