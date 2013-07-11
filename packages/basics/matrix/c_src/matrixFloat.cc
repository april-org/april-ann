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

#include <omp.h>
#include "omp_utils.h"
#include "matrix.h"
#include "matrixFloat.h"

// Auxiliary function template which applies a given FUNC object ( implements
// operator() ) to all the elements of a Matrix, using the best_span_iterator,
// and OMP if needed.
template<typename FUNC>
void applyFunctionWithSpanIterator(MatrixFloat *m,
				   FUNC &functor) {
  MatrixFloat::best_span_iterator span_it(m);
  const int N = span_it.numberOfIterations();
  unsigned int size   = static_cast<unsigned int>(span_it.getSize());
  unsigned int stride = static_cast<unsigned int>(span_it.getStride());
  functor(size, stride, static_cast<unsigned int>(span_it.getOffset()));
  if (N > 1) {
    if (omp_utils::get_num_threads() > 1 && N > 50 && size > 50) {
#pragma omp parallel for firstprivate(span_it) firstprivate(N)
      for (int i=1; i<N; ++i) {
	span_it.setAtIteration(i);
	functor(size, stride, static_cast<unsigned int>(span_it.getOffset()));
      }
    }
    else {
      ++span_it;
      do {
	functor(size, stride, static_cast<unsigned int>(span_it.getOffset()));
	++span_it;
      } while(span_it != m->end_span_iterator());
    }
  }
}

// Idem but for binary functions (needs two best_span_iterators)
// TODO:


// WARNING: ALL THE METHODS IMPLEMENTED HERE ARE SPECIALIZED TO FLOAT VERSION

/************* FILL FUNCTION **************/
struct fill_functor {
  FloatGPUMirroredMemoryBlock *data;
  float value;
  bool use_cuda;
  fill_functor(FloatGPUMirroredMemoryBlock *data, float value, bool use_cuda) :
    data(data), value(value), use_cuda(use_cuda) {
  }
  void operator()(unsigned int size, unsigned int stride, unsigned int offset) {
    doFill(size, data, stride, offset, value, use_cuda);
  }
};
template<>
void Matrix<float>::fill(float value) {
  fill_functor functor(data, value, use_cuda);
  applyFunctionWithSpanIterator(this, functor);
}

/************* CLAMP FUNCTION **************/
struct clamp_functor {
  FloatGPUMirroredMemoryBlock *data;
  float lower, upper;
  bool use_cuda;
  clamp_functor(FloatGPUMirroredMemoryBlock *data, float lower,
		float upper, bool use_cuda) :
    data(data), lower(lower), upper(upper), use_cuda(use_cuda) {
  }
  void operator()(unsigned int size, unsigned int stride, unsigned int offset) {
    doClamp(size, data, stride, offset, lower, upper, use_cuda);
  }
};
template<>
void Matrix<float>::clamp(float lower, float upper) {
  clamp_functor functor(data, lower, upper, use_cuda);
  applyFunctionWithSpanIterator(this, functor);
}


template<>
void Matrix<float>::zeros() {
  fill(0.0f);
}

template<>
void Matrix<float>::ones() {
  fill(1.0f);
}

// FIXME: implement using WRAPPER
template<>
void Matrix<float>::diag(float value) {
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

template<>
Matrix<float>* Matrix<float>::addition(const Matrix<float> *other) {
  Matrix<float> *resul = this->clone();
  resul->axpy(1.0f, other);
  return resul;
}

template<>
Matrix<float>* Matrix<float>::substraction(const Matrix<float> *other) {
  Matrix<float> *resul = this->clone();
  resul->axpy(-1.0f, other);
  return resul;
}

template<>
Matrix<float>* Matrix<float>::multiply(const Matrix<float> *other) const {
  Matrix<float> *resul = 0;
  if (other->isVector()) {
    if (this->isColVector()) {
      int dim[2] = {getVectorSize(),other->getVectorSize()};
      resul = new Matrix<float>(2, dim, major_order);
      resul->zeros();
      resul->ger(1.0f, this, other);
    }
    else if (!this->isVector()) {
      int dim[2] = {matrixSize[0],1};
      resul = new Matrix<float>(other->numDim, dim, major_order);
      resul->zeros();
      resul->gemv(CblasNoTrans,
		  1.0f, this, other,
		  0.0f);
    }
    else {
      int dim[1] = {1};
      resul = new Matrix<float>(1, dim, major_order);
      (*resul)(0) = this->dot(other);
    }
  }
  else if (numDim == 2 && other->numDim == 2 &&
	   matrixSize[1] == other->matrixSize[0]) {
    int dim[2] = {matrixSize[0], other->matrixSize[1]};
    resul = new Matrix<float>(2,dim,major_order);
    resul->zeros();
    resul->gemm(CblasNoTrans, CblasNoTrans,
		1.0f, this, other, 0.0f);
  }
  return resul;
}

// implement using WRAPPER
template<>
float Matrix<float>::sum() const {
  float s = 0.0;
  if (major_order == CblasRowMajor)
    for (const_iterator it(begin()); it!=end(); ++it) {
      s += *it;
    }
  else
    for (const_col_major_iterator it(begin()); it!=end(); ++it) {
      s += *it;
    }
    
  return s;
}

/**** COMPONENT WISE OPERATIONS ****/

// implement using WRAPPER
template<>
void Matrix<float>::scalarAdd(float s) {
  if (major_order == CblasRowMajor)
    for (iterator it(begin()); it!=end(); ++it) {
      *it = *it + s;
    }
  else
    for (col_major_iterator it(begin()); it!=end(); ++it) {
      *it = *it + s;
    }
}

// implement using WRAPPER
template<>
bool Matrix<float>::equals(const Matrix<float> *other, float epsilon) const {
  if (!sameDim(other)) return false;
  if (major_order == CblasRowMajor) {
    const_iterator it(begin());
    const_iterator other_it(other->begin());
    while(it != end()) {
      if (fabsf(*it - *other_it) > epsilon) return false;
      ++it;
      ++other_it;
    }
  }
  else {
    const_col_major_iterator it(begin());
    const_col_major_iterator other_it(other->begin());
    while(it != end()) {
      if (fabsf(*it - *other_it) > epsilon) return false;
      ++it;
      ++other_it;
    }
  }
  return true;
}

// implement using WRAPPER
template<>
void Matrix<float>::log() {
  if (major_order == CblasRowMajor)
    for (iterator it(begin()); it!=end(); ++it) {
      *it = logf(*it);
    }
  else
    for (col_major_iterator it(begin()); it!=end(); ++it) {
      *it = logf(*it);
    }
}

// implement using WRAPPER
template<>
void Matrix<float>::log1p() {
  if (major_order == CblasRowMajor)
    for (iterator it(begin()); it!=end(); ++it) {
      *it = log1pf(*it);
    }
  else
    for (col_major_iterator it(begin()); it!=end(); ++it) {
      *it = log1pf(*it);
    }
}

// implement using WRAPPER
template<>
void Matrix<float>::exp() {
  if (major_order == CblasRowMajor)
    for (iterator it(begin()); it!=end(); ++it) {
      *it = expf(*it);
    }
  else
    for (col_major_iterator it(begin()); it!=end(); ++it) {
      *it = expf(*it);
    }
}

// implement using WRAPPER
template<>
void Matrix<float>::sqrt() {
  for (iterator it(begin()); it!=end(); ++it) {
    *it = sqrtf(*it);
  }
}

// implement using WRAPPER
template<>
void Matrix<float>::pow(float value) {
  if (major_order == CblasRowMajor)
    for (iterator it(begin()); it!=end(); ++it) {
      *it = powf(*it, value);
    }
  else
    for (col_major_iterator it(begin()); it!=end(); ++it) {
      *it = powf(*it, value);
    }
}

// implement using WRAPPER
template<>
void Matrix<float>::tanh() {
  if (major_order == CblasRowMajor)
    for (iterator it(begin()); it!=end(); ++it) {
      *it = tanhf(*it);
    }
  else
    for (col_major_iterator it(begin()); it!=end(); ++it) {
      *it = tanhf(*it);
    }
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
  // Contiguous memory blocks
  if (getIsContiguous() && other->getIsContiguous())
    doScopy(total_size,
	    other->data, other->offset, 1,
	    data, offset, 1,
	    use_cuda);
  else if (numDim == 1)
    doScopy(total_size,
	    other->data, other->offset, other->stride[0],
	    data, offset, stride[0],
	    use_cuda);
  // General case
  else {
    best_span_iterator this_span_it(this), other_span_it(other);
    while(this_span_it != end_span_iterator()) {
      doScopy(static_cast<unsigned int>(this_span_it.getSize()),
	      other->data,
	      static_cast<unsigned int>(other_span_it.getOffset()),
	      static_cast<unsigned int>(other_span_it.getStride()),
	      data,
	      static_cast<unsigned int>(this_span_it.getOffset()),
	      static_cast<unsigned int>(this_span_it.getStride()),
	      use_cuda);      
      ++this_span_it;
      ++other_span_it;
    }
  }
}

template<>
void Matrix<float>::axpy(float alpha, const Matrix<float> *other) {
  if (size() != other->size())
    ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d",
		size(), other->size());
  if (major_order != other->major_order)
    ERROR_EXIT(128, "Matrices with different major orders");
  if (getIsContiguous() && other->getIsContiguous())
    doSaxpy(total_size,
	    alpha, other->data, other->offset, 1,
	    data, offset, 1,
	    use_cuda);
  else if (numDim == 1)
    doSaxpy(total_size,
	    alpha, other->data, other->offset, other->stride[0],
	    data, offset, stride[0],
	    use_cuda);
  // Two dimmension matrices
  else if (numDim == 2) {
    int larger_dim = 0, shorter_dim = 1;
    if (matrixSize[larger_dim] < matrixSize[shorter_dim])
      april_utils::swap(larger_dim, shorter_dim);
    int this_pos  = offset;
    int other_pos = other->offset;
    for (int i=0; i<matrixSize[shorter_dim]; ++i) {
      doSaxpy(matrixSize[larger_dim],
	      alpha, other->data, other_pos, other->stride[larger_dim],
	      data, this_pos, stride[larger_dim],
	      use_cuda);
      this_pos  += stride[shorter_dim];
      other_pos += other->stride[shorter_dim];
    }
  }
  // General case
  else {
    best_span_iterator this_span_it(this), other_span_it(other);
    while(this_span_it != end_span_iterator()) {
	doSaxpy(static_cast<unsigned int>(this_span_it.getSize()),
		alpha,
		other->data,
		static_cast<unsigned int>(other_span_it.getOffset()),
		static_cast<unsigned int>(other_span_it.getStride()),
		data,
		static_cast<unsigned int>(this_span_it.getOffset()),
		static_cast<unsigned int>(this_span_it.getStride()),
		use_cuda);
      ++this_span_it;
      ++other_span_it;
    }
  }
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

template<>
void Matrix<float>::scal(float value) {
  // Contiguous memory block
  if (getIsContiguous()) doSscal(total_size, value, data, offset, 1, use_cuda);
  else if (numDim == 1)
    doSscal(total_size, value, data, offset, stride[0], use_cuda);
  // Two dimmension matrix
  else if (numDim == 2) {
    int larger_dim = 0, shorter_dim = 1;
    if (matrixSize[larger_dim] < matrixSize[shorter_dim])
      april_utils::swap(larger_dim, shorter_dim);
    int pos  = offset;
    for (int i=0; i<matrixSize[shorter_dim]; ++i) {
      doSscal(matrixSize[larger_dim], value,
	      data, pos, stride[larger_dim],
	      use_cuda);
      pos += stride[shorter_dim];
    }
  }
  // General case
  else {
    best_span_iterator this_span_it(this);
    while(this_span_it != end_span_iterator()) {
      doSscal(static_cast<unsigned int>(this_span_it.getSize()),
	      value,
	      data,
	      static_cast<unsigned int>(this_span_it.getOffset()),
	      static_cast<unsigned int>(this_span_it.getStride()),
	      use_cuda);
      ++this_span_it;
    }
  }
}

template<>
float Matrix<float>::norm2() const {
  float v;
  // Contiguous memory block
  if (getIsContiguous()) v=doSnrm2(total_size, data, offset, 1, use_cuda);
  else if (numDim == 1)
    v=doSnrm2(total_size, data, offset, stride[0], use_cuda);
  // Two dimmension matrix
  else if (numDim == 2) {
    v = 0.0f;
    int larger_dim = 0, shorter_dim = 1;
    if (matrixSize[larger_dim] < matrixSize[shorter_dim])
      april_utils::swap(larger_dim, shorter_dim);
    int pos  = offset;
    switch(matrixSize[shorter_dim]) {
    case 1:
      v = doSnrm2(matrixSize[larger_dim],
		  data, pos, stride[larger_dim],
		  use_cuda);
      break;
    default:
      for (int i=0; i<matrixSize[shorter_dim]; ++i) {
	float aux = doSnrm2(matrixSize[larger_dim],
			    data, pos, stride[larger_dim],
			    use_cuda);
	v   += aux*aux;
	pos += stride[shorter_dim];
      }
      v = sqrtf(v);
    }
  }
  // General case
  else {
    v = 0.0f;
    best_span_iterator this_span_it(this);
    while(this_span_it != end_span_iterator()) {
      float aux = doSnrm2(static_cast<unsigned int>(this_span_it.getSize()),
			  data,
			  static_cast<unsigned int>(this_span_it.getOffset()),
			  static_cast<unsigned int>(this_span_it.getStride()),	  
			  use_cuda);
      v += aux*aux;
      ++this_span_it;
    }
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
