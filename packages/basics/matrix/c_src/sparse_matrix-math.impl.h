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

#include "maxmin.h"
#include "clamp.h"
#include "matrix_generic_math_templates.h"

template <typename T>
void SparseMatrix<T>::fill(T value) {
  for (iterator it(begin()); it!=end(); ++it)
    *it = value;
}

template <typename T>
void SparseMatrix<T>::clamp(T lower, T upper) {
  for (iterator it(begin()); it!=end(); ++it)
    *it = april_utils::clamp(*it, lower, upper);
}


template <typename T>
void SparseMatrix<T>::zeros() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::ones() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
SparseMatrix<T>* SparseMatrix<T>::addition(const SparseMatrix<T> *other) {
  UNUSED_VARIABLE(other);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0;
}

template <typename T>
SparseMatrix<T>* SparseMatrix<T>::substraction(const SparseMatrix<T> *other) {
  UNUSED_VARIABLE(other);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0;
}

template <typename T>
SparseMatrix<T>* SparseMatrix<T>::multiply(const SparseMatrix<T> *other) const {
  UNUSED_VARIABLE(other);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0;
}

template <typename T>
T SparseMatrix<T>::sum() const {
  T result = T();
  for (const_iterator it(begin()); it!=end(); ++it)
    result += (*it);
  return result;
}

// the argument indicates over which dimension the sum must be performed
template <typename T>
Matrix<T>* SparseMatrix<T>::sum(int dim, Matrix<T> *dest) {
  if (dim != 0 && dim != 1)
    ERROR_EXIT1(128, "Incorrect given dimension %d\n", dim);
  int ndim = (dim==0)?(1):(0);
  if (dest) {
    if (dest->getDimSize(dim) != 1 || dest->getDimSize(ndim) != matrixSize[ndim])
      ERROR_EXIT(128, "Incorrect matrix sizes\n");
  }
  else {
    int result_dims[2] = { matrixSize[0], matrixSize[1] };
    result_dims[dim] = 1;
    dest = new Matrix<T>(1, result_dims);
  }
  dest->zeros();
  typename Matrix<T>::random_access_iterator dest_it(dest);
  int aux_dims[2] = { 0, 0 };
  for (const_iterator it(begin()); it!=end(); ++it) {
    int coords[2];
    it.getCoords(coords[0],coords[1]);
    aux_dims[ndim] = coords[ndim];
    dest_it(aux_dims[0],aux_dims[1]) += (*it);
  }
  return dest;
}

/**** COMPONENT WISE OPERATIONS ****/

template <typename T>
void SparseMatrix<T>::scalarAdd(T s) {
  for (iterator it(begin()); it!=end(); ++it)
    (*it) += s;
}

template <typename T>
void SparseMatrix<T>::copy(const SparseMatrix<T> *other) {
  UNUSED_VARIABLE(other);
  if (!sameDim(other))
    ERROR_EXIT(128, "Not equal matrix dimensions or format\n");
  const_iterator it_orig(other->begin());
  iterator it_dest(this->begin());
  while(it_orig != other->end()) {
    *it_dest = *it_orig;
    ++it_orig;
    ++it_dest;
  }
}

template <typename T>
bool SparseMatrix<T>::equals(const SparseMatrix<T> *other, float epsilon) const {
  UNUSED_VARIABLE(other);
  UNUSED_VARIABLE(epsilon);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return false;
}

template <typename T>
void SparseMatrix<T>::plogp() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::log() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::log1p() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::exp() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::sqrt() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::pow(T value) {
  UNUSED_VARIABLE(value);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::tan() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::tanh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::atan() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::atanh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::cos() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::cosh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::acos() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::acosh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::sin() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::sinh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::asin() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::asinh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::abs() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::complement() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::sign() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
SparseMatrix<T> *SparseMatrix<T>::cmul(const SparseMatrix<T> *other) const {
  UNUSED_VARIABLE(other);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
T SparseMatrix<T>::dot(const SparseMatrix<T> *other) const {
  UNUSED_VARIABLE(other);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0.0f;
}

template <typename T>
void SparseMatrix<T>::scal(T value) {
  UNUSED_VARIABLE(value);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void SparseMatrix<T>::div(T value) {
  UNUSED_VARIABLE(value);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
float SparseMatrix<T>::norm2() const {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0.0f;
}
 
template <typename T>
T SparseMatrix<T>::min(int &arg_min, int &arg_min_raw_pos) const {
  UNUSED_VARIABLE(arg_min);
  UNUSED_VARIABLE(arg_min_raw_pos);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return T();
}
 
template <typename T>
T SparseMatrix<T>::max(int &arg_max, int &arg_max_raw_pos) const {
  UNUSED_VARIABLE(arg_max);
  UNUSED_VARIABLE(arg_max_raw_pos);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return T();
}
 
template <typename T>
void SparseMatrix<T>::minAndMax(T &min, T &max) const {
  UNUSED_VARIABLE(min);
  UNUSED_VARIABLE(max);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

// the argument indicates over which dimension the max must be performed
template <typename T>
Matrix<T>* SparseMatrix<T>::max(int dim, Matrix<T> *dest,
				Matrix<int32_t> *argmax) {
  if (dim != 0 && dim != 1)
    ERROR_EXIT1(128, "Incorrect given dimension %d\n", dim);
  int ndim = (dim==0)?(1):(0);
  if (dest) {
    if (dest->getDimSize(dim) != 1 || dest->getDimSize(ndim) != matrixSize[ndim])
      ERROR_EXIT(128, "Incorrect matrix sizes\n");
  }
  else {
    int result_dims[2] = { matrixSize[0], matrixSize[1] };
    result_dims[dim] = 1;
    dest = new Matrix<T>(1, result_dims);
  }
  if (argmax) {
    if (argmax->getDimSize(dim) != 1 ||
	argmax->getDimSize(ndim) != matrixSize[ndim])
      ERROR_EXIT(128, "Incorrect matrix sizes\n");
    argmax->zeros();
  }
  dest->zeros();
  typename Matrix<T>::random_access_iterator dest_it(dest);
  int aux_dims[2] = { 0, 0 };
  if (argmax == 0) {
    for (const_iterator it(begin()); it!=end(); ++it) {
      int coords[2];
      it.getCoords(coords[0],coords[1]);
      aux_dims[ndim] = coords[ndim];
      dest_it(aux_dims[0],aux_dims[1]) =
	april_utils::max(dest_it(aux_dims[0],aux_dims[1]),(*it));
    }
  }
  else {
    typename Matrix<int32_t>::random_access_iterator argmax_it(argmax);
    for (const_iterator it(begin()); it!=end(); ++it) {
      int coords[2];
      it.getCoords(coords[0],coords[1]);
      aux_dims[ndim] = coords[ndim];
      if (dest_it(aux_dims[0],aux_dims[1]) < *it) {
	dest_it(aux_dims[0],aux_dims[1]) = *it;
	argmax_it(aux_dims[0],aux_dims[1]) = aux_dims[ndim];
      }
    }
  }
  return dest;
}

// the argument indicates over which dimension the sum must be performed
template <typename T>
Matrix<T>* SparseMatrix<T>::min(int dim, Matrix<T> *dest,
				Matrix<int32_t> *argmin) {
  if (dim != 0 && dim != 1)
    ERROR_EXIT1(128, "Incorrect given dimension %d\n", dim);
  int ndim = (dim==0)?(1):(0);
  if (dest) {
    if (dest->getDimSize(dim) != 1 || dest->getDimSize(ndim) != matrixSize[ndim])
      ERROR_EXIT(128, "Incorrect matrix sizes\n");
  }
  else {
    int result_dims[2] = { matrixSize[0], matrixSize[1] };
    result_dims[dim] = 1;
    dest = new Matrix<T>(1, result_dims);
  }
  if (argmin) {
    if (argmin->getDimSize(dim) != 1 ||
	argmin->getDimSize(ndim) != matrixSize[ndim])
      ERROR_EXIT(128, "Incorrect matrix sizes\n");
    argmin->zeros();
  }
  dest->zeros();
  typename Matrix<T>::random_access_iterator dest_it(dest);
  int aux_dims[2] = { 0, 0 };
  if (argmin == 0) {
    for (const_iterator it(begin()); it!=end(); ++it) {
      int coords[2];
      it.getCoords(coords[0],coords[1]);
      aux_dims[ndim] = coords[ndim];
      dest_it(aux_dims[0],aux_dims[1]) =
	april_utils::min(dest_it(aux_dims[0],aux_dims[1]),(*it));
    }
  }
  else {
    typename Matrix<int32_t>::random_access_iterator argmin_it(argmin);
    for (const_iterator it(begin()); it!=end(); ++it) {
      int coords[2];
      it.getCoords(coords[0],coords[1]);
      aux_dims[ndim] = coords[ndim];
      if (*it < dest_it(aux_dims[0],aux_dims[1])) {
	dest_it(aux_dims[0],aux_dims[1]) = *it;
	argmin_it(aux_dims[0],aux_dims[1]) = aux_dims[ndim];
      }
    }
  }
  return dest;
}

template <typename T>
void SparseMatrix<T>::adjustRange(T rmin, T rmax) {
  UNUSED_VARIABLE(rmin);
  UNUSED_VARIABLE(rmax);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}
