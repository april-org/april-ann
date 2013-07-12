/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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

#include "matrix.h"
#include "matrixFloat.h"
#include "matrixFloat_math_templates.h" // functions which apply functors
#include "matrixFloat_math_functors.h"  // standard functors
#include "wrapper.h" // wrappers of mathematical function (for CPU/GPU)

// WARNING: ALL THE METHODS IMPLEMENTED HERE ARE SPECIALIZED TO FLOAT VERSION

/************* FILL FUNCTION **************/
template<>
void Matrix<float>::fill(float value) {
  applyFunctionWithSpanIterator(this, make_cwise_functor_1(value, doFill));
}

/************* CLAMP FUNCTION **************/
struct clamp_functor {
  float lower, upper;
  clamp_functor(float lower, float upper) :
    lower(lower), upper(upper) { }
  void operator()(MatrixFloat *m,
		  unsigned int size, unsigned int stride,
		  unsigned int offset) const {
    doClamp(size, m->getRawDataAccess(), stride, offset,
	    lower, upper, m->getCudaFlag());
  }
};
template<>
void Matrix<float>::clamp(float lower, float upper) {
  clamp_functor functor(lower, upper);
  applyFunctionWithSpanIterator(this, functor);
}

/************* ZEROS FUNCTION **************/
template<>
void Matrix<float>::zeros() {
  fill(0.0f);
}

/************* ONES FUNCTION **************/
template<>
void Matrix<float>::ones() {
  fill(1.0f);
}

/************* DIAG FUNCTION **************/
template<>
void Matrix<float>::diag(float value) {
  if (use_cuda) ERROR_EXIT(128, "DIAG OPERATION NOT IMPLENTED FOR CUDA\n");
  for (int i=1; i<numDim; ++i)
    if (matrixSize[i] != matrixSize[i-1])
      ERROR_EXIT(128, "Only allowed for squared matrices\n");
  float *d = data->getPPALForWrite();
  int *aux_coords = new int[numDim];
  for (int i=0; i<matrixSize[0]; ++i) {
    for (int j=0; j<numDim; ++j) aux_coords[j] = i;
    d[computeRawPos(aux_coords)] = value;
  }
  delete[] aux_coords;
}

/************* ADDITION FUNCTION **************/
template<>
Matrix<float>* Matrix<float>::addition(const Matrix<float> *other) {
  Matrix<float> *resul = this->clone();
  resul->axpy(1.0f, other);
  return resul;
}

/************* SUBSTRACTION FUNCTION **************/
template<>
Matrix<float>* Matrix<float>::substraction(const Matrix<float> *other) {
  Matrix<float> *resul = this->clone();
  resul->axpy(-1.0f, other);
  return resul;
}

/************* MULTIPLY FUNCTION **************/
template<>
Matrix<float>* Matrix<float>::multiply(const Matrix<float> *other) const {
  Matrix<float> *resul = 0;
  if (other->isVector()) {
    if (this->isColVector()) {
      // OUTER product
      int dim[2] = {getVectorSize(),other->getVectorSize()};
      resul = new Matrix<float>(2, dim, major_order);
      resul->zeros();
      resul->ger(1.0f, this, other);
    }
    else if (!this->isVector()) {
      // Matrix-Vector product
      int dim[2] = {matrixSize[0],1};
      resul = new Matrix<float>(other->numDim, dim, major_order);
      resul->zeros();
      resul->gemv(CblasNoTrans,
		  1.0f, this, other,
		  0.0f);
    }
    else {
      // DOT product
      int dim[1] = {1};
      resul = new Matrix<float>(1, dim, major_order);
      (*resul)(0) = this->dot(other);
    }
  }
  else if (numDim == 2 && other->numDim == 2 &&
	   matrixSize[1] == other->matrixSize[0]) {
    // Matrix-Matrix product
    int dim[2] = {matrixSize[0], other->matrixSize[1]};
    resul = new Matrix<float>(2,dim,major_order);
    resul->zeros();
    resul->gemm(CblasNoTrans, CblasNoTrans,
		1.0f, this, other, 0.0f);
  }
  return resul;
}

/************* SUM FUNCTION **************/
struct sum_functor {
  float operator()(const MatrixFloat *m,
		   unsigned int size, unsigned int stride,
		   unsigned int offset) const {
    return doSum(size, m->getRawDataAccess(), stride, offset, m->getCudaFlag());
  }
};
template<>
float Matrix<float>::sum() const {
  sum_functor functor;
  return applySumReductionWithSpanIterator(this, functor);
}

/**** COMPONENT WISE OPERATIONS ****/


/************* scalarAdd FUNCTION **************/
template<>
void Matrix<float>::scalarAdd(float s) {
  applyFunctionWithSpanIterator(this, make_cwise_functor_1(s, doScalarAdd));
}

/************* equals FUNCTION **************/
struct equals_functor {
  float epsilon;
  equals_functor(float epsilon) : epsilon(epsilon) { }
  bool operator()(const MatrixFloat *m1,
		  const MatrixFloat *m2,
		  unsigned int size,
		  unsigned int stride1,
		  unsigned int stride2,
		  unsigned int offset1,
		  unsigned int offset2) const {
    return doEquals(size, m1->getRawDataAccess(), m2->getRawDataAccess(),
		    stride1, stride2, offset1, offset2, epsilon,
		    m1->getCudaFlag() && m2->getCudaFlag());
  }
};
template<>
bool Matrix<float>::equals(const Matrix<float> *other, float epsilon) const {
  if (!sameDim(other)) return false;
  equals_functor functor(epsilon);
  return applyBinaryAndReductionWithSpanIterator(this, other, functor);
}

/************* LOG FUNCTION **************/
template<>
void Matrix<float>::log() {
  applyFunctionWithSpanIterator(this, make_cwise_functor_0(doLog));
}

/************* LOG1P FUNCTION **************/
template<>
void Matrix<float>::log1p() {
  applyFunctionWithSpanIterator(this, make_cwise_functor_0(doLog1p));
}

/************* EXP FUNCTION **************/
template<>
void Matrix<float>::exp() {
  applyFunctionWithSpanIterator(this, make_cwise_functor_0(doExp));
}

/************* SQRT FUNCTION **************/
template<>
void Matrix<float>::sqrt() {
  applyFunctionWithSpanIterator(this, make_cwise_functor_0(doSqrt));
}

/************* POW FUNCTION **************/
template<>
void Matrix<float>::pow(float value) {
  applyFunctionWithSpanIterator(this, make_cwise_functor_1(value, doPow));
}

/************* TANH FUNCTION **************/
template<>
void Matrix<float>::tanh() {
  applyFunctionWithSpanIterator(this, make_cwise_functor_0(doTanh));
}

template<>
Matrix<float> *Matrix<float>::cmul(const Matrix<float> *other) {
  if (size() != other->size())
    ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d\n",
		size(), other->size());
  if (major_order != other->major_order)
    ERROR_EXIT(128, "Matrices with different major orders\n");
  if (! sameDim(other) )
    ERROR_EXIT(128, "Matrices with different dimension sizes\n");
  if (!getIsContiguous() || !other->getIsContiguous())
    ERROR_EXIT(128, "Only allowed for contiguous matrices\n");
  Matrix<float> *new_mat = new Matrix(1, &total_size, major_order);
  doSsbmv(major_order, CblasLower,
	  total_size, 0,
	  1.0f, data, 1,
	  other->data, 1,
	  0.0f, new_mat->data, 1,
	  offset, other->offset, new_mat->offset,
	  use_cuda);
  return new_mat;
}

/**** BLAS OPERATIONS ****/
struct copy_functor {
  void operator()(MatrixFloat *dest, const MatrixFloat *orig,
		  unsigned int size,
		  unsigned int stride_dest,
		  unsigned int stride_orig,
		  unsigned int offset_dest,
		  unsigned int offset_orig) const {
    doScopy(size,
	    orig->getRawDataAccess(),
	    offset_orig, stride_orig,
	    dest->getRawDataAccess(),
	    offset_dest, stride_dest,
	    orig->getCudaFlag());
  }
};
template<>
void Matrix<float>::copy(const Matrix<float> *other) {
  if (size() != other->size())
    ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d\n",
		size(), other->size());
  if (major_order != other->major_order)
    ERROR_EXIT(128, "Matrices with different major orders\n");
  if (! sameDim(other) )
    ERROR_EXIT(128, "Matrices with different dimension sizes\n");
  use_cuda = other->use_cuda;
  copy_functor functor;
  applyBinaryFunctionWithSpanIterator(this, other, functor); //, 200, 200);
}

struct axpy_functor {
  float alpha;
  axpy_functor(float alpha) : alpha(alpha) { }
  void operator()(MatrixFloat *one, const MatrixFloat *other,
		  unsigned int size,
		  unsigned int stride_one,
		  unsigned int stride_other,
		  unsigned int offset_one,
		  unsigned int offset_other) const {
    doSaxpy(size, alpha,
	    other->getRawDataAccess(),
	    offset_other, stride_other,
	    one->getRawDataAccess(),
	    offset_one, stride_one,
	    one->getCudaFlag());
  }
};
template<>
void Matrix<float>::axpy(float alpha, const Matrix<float> *other) {
  if (size() != other->size())
    ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d",
		size(), other->size());
  if (major_order != other->major_order)
    ERROR_EXIT(128, "Matrices with different major orders");
  axpy_functor functor(alpha);
  applyBinaryFunctionWithSpanIterator(this, other, functor);
}

template<>
void Matrix<float>::gemm(CBLAS_TRANSPOSE trans_A,
			 CBLAS_TRANSPOSE trans_B,
			 float alpha,
			 const Matrix<float> *otherA,
			 const Matrix<float> *otherB,
			 float beta) {
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

template<>
void Matrix<float>::gemv(CBLAS_TRANSPOSE trans_A,
			 float alpha,
			 const Matrix<float> *otherA,
			 const Matrix<float> *otherX,
			 float beta) {
  if (!isVector() || !otherX->isVector() || otherA->numDim != 2)
    ERROR_EXIT(128,"Incorrect number of dimensions\n");
  int row_idx_A = 0, col_idx_A = 1;
  if (trans_A == CblasTrans) april_utils::swap(row_idx_A, col_idx_A);
  if (getVectorSize() != otherA->matrixSize[row_idx_A] ||
      otherA->matrixSize[col_idx_A] != otherX->getVectorSize())
    ERROR_EXIT4(128, "Incorrect matrixes dimensions: %dx1 + %dx%d * %dx1\n",
		getVectorSize(),
		otherA->matrixSize[row_idx_A], otherA->matrixSize[col_idx_A],
		otherX->getVectorSize());
  if (major_order != otherA->major_order ||
      otherA->major_order != otherX->major_order)
    ERROR_EXIT(128, "Matrices with different major orders\n");
  
  int M=otherA->matrixSize[0], N=otherA->matrixSize[1];
  int lda=( major_order==CblasRowMajor)?otherA->stride[0]:otherA->stride[1];
  int ldx=otherX->getVectorStride();
  int ldy=getVectorStride();
  doSgemv(major_order, trans_A,
	  M, N,
	  alpha, otherA->data, lda,
	  otherX->data, ldx,
	  beta, data, ldy,
	  otherA->offset, otherX->offset, offset,
	  use_cuda);
}

template<>
void Matrix<float>::ger(float alpha,
			const Matrix<float> *otherX,
			const Matrix<float> *otherY) {
  if (!otherX->isVector() || !otherY->isVector() || numDim!=2)
    ERROR_EXIT(128,"Incorrect number of dimensions");
  if (matrixSize[0] != otherX->getVectorSize() ||
      matrixSize[1] != otherY->getVectorSize())
    ERROR_EXIT4(128, "Incorrect matrixes dimensions: %dx%d + %dx1 * 1x%d\n",
		matrixSize[0], matrixSize[1],
		otherX->getVectorSize(), otherY->getVectorSize());
  if (major_order != otherX->major_order ||
      otherX->major_order != otherY->major_order)
    ERROR_EXIT(128, "Matrices with different major orders");
  int M=matrixSize[0], N=matrixSize[1];
  int lda=( major_order==CblasRowMajor)?stride[0]:stride[1];
  int ldx=otherX->getVectorStride();
  int ldy=otherY->getVectorStride();
  doSger(major_order,
	 M, N,
	 alpha, otherX->data, otherX->offset, ldx,
	 otherY->data, otherY->offset, ldy,
	 data, offset, lda,
	 use_cuda);
}

template<>
float Matrix<float>::dot(const Matrix<float> *other) const {
  if (!this->isVector() || !other->isVector())
    ERROR_EXIT(128,"Incorrect number of dimensions");
  if (this->getVectorSize() != other->getVectorSize())
    ERROR_EXIT2(128, "Incorrect dimensions: %d dot %d\n",
		this->getVectorSize(), other->getVectorSize());
  if (major_order != other->major_order)
    ERROR_EXIT(128, "Matrices with different major orders");
  float ret = doSdot(getVectorSize(),
		     data, offset, getVectorStride(),
		     other->data, other->offset, other->getVectorStride(),
		     use_cuda);
  return ret;
}

/********** SCAL FUNCTION ***************/
template<>
void Matrix<float>::scal(float value) {
  applyFunctionWithSpanIterator(this, make_cwise_functor_1(value, doSscal));
}

/********** NORM2 FUNCTION ***************/
struct norm2_functor {
  float operator()(const MatrixFloat *m, unsigned int size, unsigned int stride,
		   unsigned int offset) const {
    return doSnrm2(size, m->getRawDataAccess(), stride, offset,
		   m->getCudaFlag());
  }
};
struct norm2_reductor {
  float operator()(float accum, float other) const {
    return accum + other*other;
  }
};
// In this method we do ad-hoc specialization of BASIC cases because
// we avoid the SQUARE and SQRT functions
template<>
float Matrix<float>::norm2() const {
  float v;
  // Contiguous memory block
  if (getIsContiguous()) v=doSnrm2(total_size, data, 1, offset, use_cuda);
  // One dimension
  else if (numDim == 1)
    v=doSnrm2(total_size, data, stride[0], offset, use_cuda);
  // General case
  else {
    norm2_functor  functor;
    norm2_reductor reductor;
    v = applyReductionWithSpanIterator(this,
				       functor,
				       reductor,
				       0.0f);
    v = sqrtf(v);
  }
  return v;
}

// FIXME: using WRAPPER
template<>
float Matrix<float>::min(int &arg_min) const {
  const_iterator it(begin());
  const_iterator result = april_utils::argmin(it, const_iterator(end()));
  arg_min = result.getIdx();
  return *result;
}

// FIXME: using WRAPPER
template<>
float Matrix<float>::max(int &arg_max) const {
  const_iterator it(begin());
  const_iterator result = april_utils::argmax(it, const_iterator(end()));
  arg_max = result.getIdx();
  return *result;
}

// FIXME: using WRAPPER
template<>
void Matrix<float>::minAndMax(float &min, float &max) const {
  if (major_order == CblasRowMajor) {
    const_iterator it(begin());
    min = *it;
    max = *it;
    for (; it!=end(); ++it) {
      if (*it < min) min = *it;
      if (*it > max) max = *it;
    }
  }
  else {
    const_col_major_iterator it(begin());
    min = *it;
    max = *it;
    for (; it!=end(); ++it) {
      if (*it < min) min = *it;
      if (*it > max) max = *it;
    }
  }
}

// FIXME: using WRAPPER
template<>
void Matrix<float>::adjustRange(float rmin, float rmax) {
  float mmin, mmax;
  minAndMax(mmin, mmax);
  // especial case, set all values to rmin
  if (mmax - mmin == 0) fill(rmin);
  else {
    float ratio = (rmax-rmin)/(mmax-mmin);
    if (mmin > 0.0f || mmin < 0.0f) scalarAdd(-mmin);
    scal(ratio);
    if (rmin > 0.0f || rmin < 0.0f) scalarAdd(rmin);
  }
}

template class Matrix<float>;
