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
  // Number of dimensions
  int numDim;
  // Size of each dimension
  int *matrixSize;
  // Total size of the matrix (number of elements)
  int size;
  //datos
  T *data; ///< Pointer used after taking into account the alignment
  // Constructor... -> Integer array with the size of each dimension
  Matrix(int numDim, const int* dim, T* data_vector);
  Matrix(int numDim, const int* dim, T default_value=T());
  Matrix(int numDim, int d1, ...);
  Matrix(Matrix<T> *other);
  // Destructor...
  virtual ~Matrix();
  // Loader
  Matrix<T>* clone();

  // Access to independent elements
  T& operator() (int i);
  T& operator() (int row, int col);
  T& operator() (int coord0, int coord1, int coord2, ...);
  const T& operator() (int i) const;
  const T& operator() (int row, int col) const;
  const T& operator() (int coord0, int coord1, int coord2, ...) const;

  bool getCol(int col, T* vec, int vecsize);
  bool putCol(int col, T *vec, int vecsize);
  bool putSubCol(int col, int first_row, T *vec, int vecsize);

  void clamp(T lower, T upper);

  // Returns true if they have the same dimension
  bool sameDim(const Matrix *other) const;

  // Returns a new matrix with the sum, assuming they have the same dimension
  // Crashes otherwise
  Matrix<T>* addition(const Matrix<T> *other);

  // The same as the previous one but accumulates the result in the matrix
  void accumulate_addition(const Matrix<T> *other);

  // The same as addition but substracting
  Matrix<T>* substraction(const Matrix<T> *other);

  // The same as the previous one but accumulates the result in the matrix
  void accumulate_substraction(const Matrix<T> *other);

  // Matrices must be squared, returns the product of this and other
  Matrix<T>* multiply(const Matrix<T> *other) const;

  void multiply_by_scalar(T value);
private:
  void allocate_memory(int size);
  void release_memory(T *ptr);
};

template <typename T>
void Matrix<T>::allocate_memory(int size) {
  // previously:
  //data = new T[size];
  data = aligned_malloc<T>(size);
  for (int i=0; i<size; ++i) new(data+i) T();
}

template <typename T>
void Matrix<T>::release_memory(T *ptr) {
  for (int i=0; i<size; ++i) data[i].~T();
  aligned_free(ptr);
}

template <typename T>
Matrix<T>::Matrix(int numDim, const int* dim, T default_value) {
  this->numDim=numDim;
  matrixSize=new int[numDim];
  size=1;
  for(int i=0; i<numDim; i++) {
    size *= dim[i];
    matrixSize[i] = dim[i];
  }

  allocate_memory(size);
  
  for (int i=0; i<size; ++i)
    data[i] = default_value;
}

template <typename T>
Matrix<T>::Matrix(int numDim, const int* dim, T* data_vector) {
  this->numDim=numDim;
  matrixSize=new int[numDim];
  size=1;
  for(int i=0; i<numDim; i++) {
    size *= dim[i];
    matrixSize[i] = dim[i];
  }
  data = data_vector;
}

template <typename T>
Matrix<T>::Matrix(int numDim, int d1, ...) {
    this->numDim=numDim;
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
        data[i] = default_value;
}


template <typename T>
Matrix<T>::Matrix(Matrix<T> *other) {
  numDim = other->numDim;
  matrixSize = new int[numDim];
  for (int i=0; i<numDim; i++) {
    matrixSize[i] = other->matrixSize[i];
  }
  size = other->size;

  allocate_memory(size);

  for (int i=0; i<size; ++i)
    data[i] = other->data[i];
}

template <typename T>
Matrix<T>::~Matrix() {
  release_memory(data);
  delete[] matrixSize;
}

template <typename T>
Matrix<T>* Matrix<T>::clone() {
  return new Matrix<T>(this);
}

template <typename T>
T& Matrix<T>::operator() (int i) {
    return data[i];
}

template <typename T>
T& Matrix<T>::operator() (int row, int col) {
    return data[row*matrixSize[1]+col];
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

    return data[index];
}

template <typename T>
const T& Matrix<T>::operator() (int i) const {
    return data[i];
}

template <typename T>
const T& Matrix<T>::operator() (int row, int col) const {
    return data[row*matrixSize[1]+col];
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

    return data[index];
}

template <typename T>
bool Matrix<T>::getCol(int col, T* vec, int vecsize) {
  // If the column is out of range, error
  if ((col < 0) || (col >= matrixSize[1]))
    return false;
  // If the array length is different to the size of the matrix columns, error
  if (vecsize != matrixSize[0]) 
    return false;
  // If it is not a 2D matrix, error
  if (numDim != 2) 
    return false;

  for (int row = 0; row < matrixSize[0]; row++) {
    vec[row] = data[matrixSize[1]*row+col];
  }
  return true;
}

template <typename T>
bool Matrix<T>::putCol(int col, T* vec, int vecsize) {
  // If the column is out of range, error
  if ((col < 0) || (col >= matrixSize[1]))
    return false;
  // If the array length is different to the size of the matrix columns, error
  if (vecsize != matrixSize[0]) 
    return false;
  // If it is not a 2D matrix, error
  if (numDim != 2) 
    return false;

  for (int row = 0; row < matrixSize[0]; row++) {
    data[matrixSize[1]*row+col] = vec[row];
  }
  return true;
}

template <typename T>
bool Matrix<T>::putSubCol(int col, int first_row, T* vec, int vecsize) {
  // If the column is out of range, error
  if ((col < 0) || (col >= matrixSize[1]))
    return false;
  // If the first row is out of range, error
  if ((first_row < 0) || (first_row >= matrixSize[0]))
    return false;
  // If the array is out of range, error
  if ((first_row < 0) || (first_row+vecsize > matrixSize[0]))
    return false;
  // If it is not a 2D matrix, error
  if (numDim != 2) 
    return false;

  for (int row = first_row; row < first_row+vecsize; row++) {
    data[matrixSize[1]*row+col] = vec[row];
  }
  return true;
}

template <typename T>
void Matrix<T>::clamp(T lower, T upper) {
  for (int i=0; i<size; ++i)
    data[i] = april_utils::clamp(data[i],lower,upper);
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
  Matrix<T> *resul = new Matrix<T>(numDim,matrixSize);
  for (int i=0; i<size; ++i)
    resul->data[i] = data[i] + other->data[i];
  return resul;
}

template <typename T>
void Matrix<T>::accumulate_addition(const Matrix<T> *other) {
  for (int i=0; i<size; ++i)
    data[i] += other->data[i];
}

template <typename T>
Matrix<T>* Matrix<T>::substraction(const Matrix<T> *other) {
  Matrix<T> *resul = new Matrix<T>(numDim,matrixSize);
  for (int i=0; i<size; ++i)
    resul->data[i] = data[i] - other->data[i];
  return resul;
}

template <typename T>
void Matrix<T>::accumulate_substraction(const Matrix<T> *other) {
  for (int i=0; i<size; ++i)
    data[i] -= other->data[i];
}

template <typename T>
Matrix<T>* Matrix<T>::multiply(const Matrix<T> *other) const {
  if (numDim != 2 || other->numDim != 2 ||
      matrixSize[1] != other->matrixSize[0]) return 0;
  int dim[2];
  int N= matrixSize[1];
  dim[0] = matrixSize[0]; dim[1] = other->matrixSize[1];
  Matrix<T> *resul = new Matrix<T>(2,dim);
  for (int i=0; i<dim[0]; ++i) {
    T *fila = data+i*N;
    for (int j=0; j<dim[1]; ++j) {
      T *col = other->data+j;
      T aux = T();
      for (int k=0; k<N; ++k)
	aux += fila[k]*col[dim[1]*k];
      resul->data[i*dim[1]+j] = aux;
    }
  }
  return resul;
}

template <typename T>
void Matrix<T>::multiply_by_scalar(T value) {
  for (int i=0; i<size; ++i)
    data[i] *= value;
}

#endif // MATRIX_H
