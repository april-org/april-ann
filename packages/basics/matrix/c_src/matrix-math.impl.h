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
void Matrix<T>::fill(T value) {
  if (getIsDataRowOrdered())
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
  for (iterator it(begin()); it!=end(); ++it)
    *it = april_utils::clamp(*it, lower, upper);
}

template <typename T>
void Matrix<T>::zeros() {
  fill(T());
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
  if (!sameDim(other))
    ERROR_EXIT(128, "Not equal matrix dimensions or format\n");
  Matrix<T> *result = new Matrix<T>(getNumDim(), matrixSize, major_order);
  const_iterator this_it(this->begin());
  const_iterator other_it(this->begin());
  for (iterator result_it(result->begin());
       result_it != result->end(); ++result_it, ++this_it, ++other_it) {
    april_assert(this_it != this->end());
    april_assert(other_it != other->end());
    *result_it = (*this_it) + (*other_it);
  }
  return result;
}

template <typename T>
Matrix<T>* Matrix<T>::substraction(const Matrix<T> *other) {
  if (!sameDim(other))
    ERROR_EXIT(128, "Not equal matrix dimensions or format\n");
  Matrix<T> *result = new Matrix<T>(getNumDim(), matrixSize, major_order);
  const_iterator this_it(this->begin());
  const_iterator other_it(this->begin());
  for (iterator result_it(result->begin());
       result_it != result->end(); ++result_it, ++this_it, ++other_it) {
    april_assert(this_it != this->end());
    april_assert(other_it != other->end());
    *result_it = (*this_it) - (*other_it);
  }
  return result;
}

template <typename T>
Matrix<T>* Matrix<T>::multiply(const Matrix<T> *other) const {
  if (this->getNumDim() != 2 || other->getNumDim() != 2)
    ERROR_EXIT(128, "Bi-dimensional matrices expected\n");
  if (this->matrixSize[1] != other->matrixSize[0])
    ERROR_EXIT(128, "Incorrect matrix sizes\n");
  int result_dims[2] = { this->matrixSize[0], other->matrixSize[1] };
  Matrix<T> *result = new Matrix<T>(2, result_dims, major_order);
  iterator result_it(result->begin());
  const_iterator this_it(this->begin());
  for (int i=0; i<matrixSize[0]; ++i) {
    const_col_major_iterator other_it(other->begin());
    for (int j=0; j<matrixSize[1]; ++j, ++result_it) {
      const_iterator aux_this_it(this_it);
      *result_it = T();
      for (int k=0; k<result_dims[1]; ++k, ++aux_this_it, ++other_it) {
        *result_it += (*this_it) * (*other_it);
      }
    }
  }
  return result;
}

template <typename T>
T Matrix<T>::sum() const {
  T result = T();
  for (const_iterator it(begin()); it!=end(); ++it)
    result += (*it);
  return result;
}

// the argument indicates over which dimension the sum must be performed
template<typename T>
struct sum_dim_functor {
  T operator()(const Matrix<T> *slice) { return slice->sum(); }
};
template <typename T>
Matrix<T>* Matrix<T>::sum(int dim, Matrix<T> *dest) {
  return applyFunctorOverDimension<T,T>(sum_dim_functor<T>(), this, dim, dest);
}

/**** COMPONENT WISE OPERATIONS ****/

template <typename T>
void Matrix<T>::scalarAdd(T s) {
  for (iterator it(begin()); it!=end(); ++it)
    (*it) += s;
}

template <typename T>
void Matrix<T>::copy(const Matrix<T> *other) {
  UNUSED_VARIABLE(other);
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
  UNUSED_VARIABLE(other);
  UNUSED_VARIABLE(epsilon);
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
  UNUSED_VARIABLE(value);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::tan() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::tanh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::atan() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::atanh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::cos() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::cosh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::acos() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::acosh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::sin() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::sinh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::asin() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::asinh() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::abs() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::complement() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::sign() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::cmul(const Matrix<T> *other) {
  UNUSED_VARIABLE(other);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
struct axpy_functor {
  T alpha;
  axpy_functor(T alpha) : alpha(alpha) { }
  void operator()(Matrix<T> *one, const Matrix<T> *other,
		  unsigned int size,
		  unsigned int stride_one,
		  unsigned int stride_other,
		  unsigned int offset_one,
		  unsigned int offset_other) const {
    doAxpy(size, alpha,
	   other->getRawDataAccess(),
	   offset_other, stride_other,
	   one->getRawDataAccess(),
	   offset_one, stride_one,
	   one->getCudaFlag());
  }
};
template <typename T>
void Matrix<T>::axpy(T alpha, const Matrix<T> *other) {
  if (size() != other->size())
    ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d\n",
		size(), other->size());
  if (major_order != other->major_order)
    ERROR_EXIT(128, "Matrices with different major orders\n");
  axpy_functor<T> functor(alpha);
#ifdef USE_MKL
  applyBinaryFunctionWithSpanIteratorNOPARALLEL<T>(this, other, functor);
#else
  applyBinaryFunctionWithSpanIterator<T>(this, other, functor);
#endif
}

template <typename T>
void Matrix<T>::axpy(T alpha, const SparseMatrix<T> *other) {
  if (size() != other->size())
    ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d\n",
		size(), other->size());
  if (!isVector())
    ERROR_EXIT(128, "sparse AXPY only works with vectors\n");
  if ( (other->getSparseFormat() == SparseMatrix<T>::CSR_FORMAT &&
	other->getDimSize(0) != 1) ||
       (other->getSparseFormat() == SparseMatrix<T>::CSC_FORMAT &&
	other->getDimSize(1) != 1) )
    ERROR_EXIT(128, "sparse AXPY needs a CSR row-vector or a CSC col-vector\n");
  doSparseAxpy(other->nonZeroSize(), alpha,
	       other->getRawValuesAccess(),
	       other->getRawIndicesAccess(),
	       getRawDataAccess(),
	       static_cast<unsigned int>(getOffset()),
	       static_cast<unsigned int>(getVectorStride()),
	       getCudaFlag());  
}

template <typename T>
void Matrix<T>::gemm(CBLAS_TRANSPOSE trans_A,
		     CBLAS_TRANSPOSE trans_B,
		     T alpha,
		     const Matrix<T> *otherA,
		     const Matrix<T> *otherB,
		     T beta) {
  if (this == otherA || this == otherB)
    ERROR_EXIT(128, "GEMM method couldn't receive as A or B argument "
	       "the caller object\n");
  if (numDim != 2 || otherA->numDim != 2 || otherB->numDim != 2)
    ERROR_EXIT(128,"Incorrect number of dimensions, only allowed for numDim=2\n");
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
    ERROR_EXIT(128, "Matrices with different major orders\n");
  
  int M=matrixSize[0], N=matrixSize[1], K=otherA->matrixSize[col_idx_A];
  int lda, ldb, ldc;
  if (major_order == CblasRowMajor) {
    lda = (!otherA->getTransposedFlag())?(otherA->stride[0]):(otherA->stride[1]);
    ldb = (!otherB->getTransposedFlag())?(otherB->stride[0]):(otherB->stride[1]);
    ldc = (!this->getTransposedFlag()  )?(this->stride[0]  ):(this->stride[1]);
  }
  else {
    lda = (!otherA->getTransposedFlag())?(otherA->stride[1]):(otherA->stride[0]);
    ldb = (!otherB->getTransposedFlag())?(otherB->stride[1]):(otherB->stride[0]);
    ldc = (!this->getTransposedFlag()  )?(this->stride[1]  ):(this->stride[0]);
  }
  if (otherA->stride[0]+otherA->stride[1] != lda+1 ||
      otherB->stride[0]+otherB->stride[1] != ldb+1 ||
      this->stride[0]  +this->stride[1]   != ldc+1)
    ERROR_EXIT(128, "Contiguous matrices are needed\n");
  if (otherA->getTransposedFlag()) trans_A=NEGATE_CBLAS_TRANSPOSE(trans_A);
  if (otherB->getTransposedFlag()) trans_B=NEGATE_CBLAS_TRANSPOSE(trans_B);
  doGemm(major_order, trans_A, trans_B,
	 M, N, K,
	 alpha, otherA->data, lda,
	 otherB->data, ldb,
	 beta, data, ldc,
	 otherA->offset, otherB->offset, offset,
	 use_cuda);
}

template <typename T>
void Matrix<T>::gemm(CBLAS_TRANSPOSE trans_A,
		     CBLAS_TRANSPOSE trans_B,
		     T alpha,
		     const SparseMatrix<T> *otherA,
		     const Matrix<T> *otherB,
		     T beta) {
  UNUSED_VARIABLE(trans_A);
  UNUSED_VARIABLE(trans_B);
  UNUSED_VARIABLE(alpha);
  UNUSED_VARIABLE(otherA);
  UNUSED_VARIABLE(otherB);
  UNUSED_VARIABLE(beta);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::gemv(CBLAS_TRANSPOSE trans_A,
		     T alpha,
		     const Matrix<T> *otherA,
		     const Matrix<T> *otherX,
		     T beta) {
  UNUSED_VARIABLE(trans_A);
  UNUSED_VARIABLE(alpha);
  UNUSED_VARIABLE(otherA);
  UNUSED_VARIABLE(otherX);
  UNUSED_VARIABLE(beta);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::gemv(CBLAS_TRANSPOSE trans_A,
		     T alpha,
		     const SparseMatrix<T> *otherA,
		     const Matrix<T> *otherX,
		     T beta) {
  UNUSED_VARIABLE(trans_A);
  UNUSED_VARIABLE(alpha);
  UNUSED_VARIABLE(otherA);
  UNUSED_VARIABLE(otherX);
  UNUSED_VARIABLE(beta);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::ger(T alpha,
		    const Matrix<T> *otherX,
		    const Matrix<T> *otherY) {
  UNUSED_VARIABLE(alpha);
  UNUSED_VARIABLE(otherX);
  UNUSED_VARIABLE(otherY);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
T Matrix<T>::dot(const Matrix<T> *other) const {
  UNUSED_VARIABLE(other);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return T();
}

template <typename T>
T Matrix<T>::dot(const SparseMatrix<T> *other) const {
  UNUSED_VARIABLE(other);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return T();
}

template <typename T>
void Matrix<T>::scal(T value) {
  for (iterator it(begin()); it != end(); ++it)
    *it *= value;
}

template <typename T>
void Matrix<T>::div(T value) {
  for (iterator it(begin()); it != end(); ++it)
    *it /= value;
}

template <typename T>
float Matrix<T>::norm2() const {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0.0f;
}
 
template <typename T>
T Matrix<T>::min(int &arg_min, int &arg_min_raw_pos) const {
  const_iterator it = april_utils::argmin(begin(),end());
  arg_min = it.idx;
  arg_min_raw_pos = it.raw_pos;
  return *it;
}
 
template <typename T>
T Matrix<T>::max(int &arg_max, int &arg_max_raw_pos) const {
  const_iterator it = april_utils::argmax(begin(),end());
  arg_max = it.idx;
  arg_max_raw_pos = it.raw_pos;
  return *it;
}
 
template <typename T>
void Matrix<T>::minAndMax(T &min, T &max) const {
  const_iterator it(begin());
  min = max = *it;
  ++it;
  for (; it != end(); ++it) {
    if ( max < (*it) ) max = *it;
    else if ( (*it) < min ) min = *it;
  }
}

// the argument indicates over which dimension the sum must be performed
template <typename T>
struct max_dim_functor {
  T operator()(const Matrix<T> *slice) { int a,b; return slice->max(a,b); }
};
template <typename T>
struct max_and_argmax_dim_functor {
  T operator()(const Matrix<T> *slice, int32_t &argmax_pos) {
    int a,b;
    T max = slice->max(a,b);
    argmax_pos = a+1;
    return max;
  }
};
template <typename T>
Matrix<T>* Matrix<T>::max(int dim, Matrix<T> *dest, Matrix<int32_t> *argmax) {
  if (argmax == 0)
    return applyFunctorOverDimension<T,T>(max_dim_functor<T>(), this, dim, dest);
  else
    return applyFunctorOverDimension2<T,T,int32_t>(max_and_argmax_dim_functor<T>(),
						   this, dim, dest, argmax);
}

// the argument indicates over which dimension the sum must be performed
template <typename T>
struct min_dim_functor {
  T operator()(const Matrix<T> *slice) { int a,b; return slice->min(a,b); }
};
template <typename T>
struct min_and_argmin_dim_functor {
  T operator()(const Matrix<T> *slice, int32_t &argmin_pos) {
    int a,b;
    T min = slice->min(a,b);
    argmin_pos = a+1;
    return min;
  }
};
template <typename T>
Matrix<T>* Matrix<T>::min(int dim, Matrix<T> *dest, Matrix<int32_t> *argmin) {
  if (argmin == 0)
    return applyFunctorOverDimension<T,T>(min_dim_functor<T>(), this, dim, dest);
  else
    return applyFunctorOverDimension2<T,T,int32_t>(min_and_argmin_dim_functor<T>(),
						   this, dim, dest, argmin);
}

template <typename T>
Matrix<T> *Matrix<T>::maxSelDim(const int dim,
				Int32GPUMirroredMemoryBlock *raw_positions,
				int shift) const {
  UNUSED_VARIABLE(dim);
  UNUSED_VARIABLE(raw_positions);
  UNUSED_VARIABLE(shift);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
  return 0;
}

template <typename T>
void Matrix<T>::adjustRange(T rmin, T rmax) {
  UNUSED_VARIABLE(rmin);
  UNUSED_VARIABLE(rmax);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
Matrix<T> *Matrix<T>::inv() {
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::svd(Matrix<T> **U, SparseMatrix<T> **S, Matrix<T> **V) {
  UNUSED_VARIABLE(U);
  UNUSED_VARIABLE(S);
  UNUSED_VARIABLE(V);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}


// FIXME: IMPLEMENT THE BOOLEAN CONDITIONS USING CUDA WRAPPERS

/* BOOLEAN CONDITIONS: this methods transforms the given matrix in a ZERO/ONE
   matrix, depending in the truth of the given condition */
// less than
template <typename T>
void Matrix<T>::LTCondition(T value) {
  UNUSED_VARIABLE(value);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

template <typename T>
void Matrix<T>::LTCondition(Matrix<T> *value) {
  UNUSED_VARIABLE(value);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}

// greater than
template <typename T>
void Matrix<T>::GTCondition(T value) {
  UNUSED_VARIABLE(value);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}
template <typename T>
void Matrix<T>::GTCondition(Matrix<T> *value) {
  UNUSED_VARIABLE(value);
  ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
}
//
