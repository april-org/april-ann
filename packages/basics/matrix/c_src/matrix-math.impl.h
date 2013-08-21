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

template <typename T>
void Matrix<T>::fill(T value) {
  if (major_order == CblasRowMajor)
    for (iterator it(begin()); it!=end(); ++it) {
      *it = value;
    }
  else
    for (col_major_iterator it(begin()); it!=end(); ++it) {
      *it = value;
    }
}

template <typename T>
void Matrix<T>::clamp(T lower, T upper) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}


template <typename T>
void Matrix<T>::zeros() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::ones() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::diag(T value) {
  if (use_cuda) ERROR_PRINT("WARNING! DIAG OPERATION NOT IMPLENTED FOR CUDA\n");
  for (int i=1; i<numDim; ++i)
    if (matrixSize[i] != matrixSize[i-1])
      ERROR_EXIT(128, "Only allowed for squared matrices\n");
  T *d = data->getPPALForWrite();
  int *aux_coords = new int[numDim];
  for (int i=0; i<matrixSize[0]; ++i) {
    for (int j=0; j<numDim; ++j) aux_coords[j] = i;
    d[computeRawPos(aux_coords)] = value;
  }
  delete[] aux_coords;
}


template <typename T>
Matrix<T>* Matrix<T>::addition(const Matrix<T> *other) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0;
}

template <typename T>
Matrix<T>* Matrix<T>::substraction(const Matrix<T> *other) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0;
}

template <typename T>
Matrix<T>* Matrix<T>::multiply(const Matrix<T> *other) const {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0;
}

template <typename T>
T Matrix<T>::sum() const {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return T();
}

// the argument indicates over which dimension the sum must be performed
template <typename T>
Matrix<T>* Matrix<T>::sum(int dim) {
  int *result_dims = new int[numDim];
  /**** THIS sliding window ****/
  int *this_w_size = new int[numDim];
  int *this_w_num_steps   = new int[numDim];
  for (int i=0; i<dim; ++i) {
    this_w_size[i] = 1;
    result_dims[i] = this_w_num_steps[i] = matrixSize[i];
  }
  result_dims[dim] = 1;
  this_w_size[dim] = matrixSize[dim];
  this_w_num_steps[dim] = 1;
  for (int i=dim+1; i<numDim; ++i) {
    this_w_size[i] = 1;
    result_dims[i] = this_w_num_steps[i] = matrixSize[i];
  }
  sliding_window this_w(this,this_w_size,0,0,this_w_num_steps,0);
  Matrix<T> *slice = this_w.getMatrix();
  IncRef(slice);
  /******************************/
  Matrix<T> *result = new Matrix<T>(numDim, result_dims, major_order);
  // traverse in row major order
  for (iterator it(result->begin()); it!=result->end(); ++it) {
    this_w.getMatrix(slice);
    *it = slice->sum();
    this_w.next();
  }
  DecRef(slice);
  delete[] this_w_size;
  delete[] this_w_num_steps;
  return result;
}

/**** COMPONENT WISE OPERATIONS ****/

template <typename T>
void Matrix<T>::scalarAdd(T s) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::copy(const Matrix<T> *other) {
  if (!sameDim(other))
    ERROR_EXIT(128, "Not equal matrix dimensions\n");
  const_iterator it_orig(other->begin());
  iterator it_dest(this->begin());
  while(it_orig != other->end()) {
    *it_dest = *it_orig;
    ++it_orig;
    ++it_dest;
  }
}

template <typename T>
bool Matrix<T>::equals(const Matrix<T> *other, float epsilon) const {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return false;
}

template <typename T>
void Matrix<T>::plogp() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::log() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::log1p() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::exp() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::sqrt() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::pow(T value) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::tanh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::cos() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::sin() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
Matrix<T> *Matrix<T>::cmul(const Matrix<T> *other) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0;
}

template <typename T>
void Matrix<T>::axpy(T alpha, const Matrix<T> *other) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::gemm(CBLAS_TRANSPOSE trans_A,
		     CBLAS_TRANSPOSE trans_B,
		     T alpha,
		     const Matrix<T> *otherA,
		     const Matrix<T> *otherB,
		     T beta) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::gemv(CBLAS_TRANSPOSE trans_A,
		     T alpha,
		     const Matrix<T> *otherA,
		     const Matrix<T> *otherX,
		     T beta) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::ger(T alpha,
		    const Matrix<T> *otherX,
		    const Matrix<T> *otherY) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
T Matrix<T>::dot(const Matrix<T> *other) const {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0.0f;
}

template <typename T>
void Matrix<T>::scal(T value) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
float Matrix<T>::norm2() const {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0.0f;
}

template <typename T>
T Matrix<T>::min(int &arg_min, int &arg_min_raw_pos) const {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return T();
}

template <typename T>
T Matrix<T>::max(int &arg_max, int &arg_max_raw_pos) const {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return T();
}

template <typename T>
void Matrix<T>::minAndMax(T &min, T &max) const {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
Matrix<T> *Matrix<T>::maxSelDim(const int dim,
				IntGPUMirroredMemoryBlock *raw_positions,
				int shift) const {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0;
}

template <typename T>
void Matrix<T>::adjustRange(T rmin, T rmax) {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}
