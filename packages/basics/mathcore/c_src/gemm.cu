/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
 *
 * The APRIL-MLP toolkit is free software; you can redistribute it and/or modify it
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
#include "wrapper.h"
#include "cuda_utils.h"
#include "unused_variable.h"

#ifdef USE_CUDA
/***************************************
 ************** CUDA SECTION ***********
 ***************************************/

cublasStatus_t wrapperCublasGemm(cublasHandle_t &handle,
				 cublasOperation_t &cublas_a_transpose,
				 cublasOperation_t &cublas_b_transpose,
				 int m, int n, int k,
				 float *alpha,
				 const float *a_mem,
				 unsigned int a_inc,
				 const float *b_mem,
				 unsigned int b_inc,
				 float *beta,
				 float *c_mem,
				 unsigned int c_inc) {
  return cublasSgemm(handle, cublas_a_transpose, cublas_b_transpose,
		     m, n, k,
		     alpha, a_mem, a_inc,
		     b_mem, b_inc,
		     beta, c_mem, c_inc);
}

cublasStatus_t wrapperCublasGemm(cublasHandle_t &handle,
				 cublasOperation_t &cublas_a_transpose,
				 cublasOperation_t &cublas_b_transpose,
				 int m, int n, int k,
				 double *alpha,
				 const double *a_mem,
				 unsigned int a_inc,
				 const double *b_mem,
				 unsigned int b_inc,
				 double *beta,
				 double *c_mem,
				 unsigned int c_inc) {
  return cublasDgemm(handle, cublas_a_transpose, cublas_b_transpose,
		     m, n, k,
		     alpha, a_mem, a_inc,
		     b_mem, b_inc,
		     beta, c_mem, c_inc);
}

cublasStatus_t wrapperCublasGemm(cublasHandle_t &handle,
				 cublasOperation_t &cublas_a_transpose,
				 cublasOperation_t &cublas_b_transpose,
				 int m, int n, int k,
				 ComplexF *alpha,
				 const ComplexF *a_mem,
				 unsigned int a_inc,
				 const ComplexF *b_mem,
				 unsigned int b_inc,
				 ComplexF *beta,
				 ComplexF *c_mem,
				 unsigned int c_inc) {
  return cublasCgemm(handle, cublas_a_transpose, cublas_b_transpose,
		     m, n, k,
		     reinterpret_cast<const cuComplex*>(alpha),
		     reinterpret_cast<const cuComplex*>(a_mem), a_inc,
		     reinterpret_cast<const cuComplex*>(b_mem), b_inc,
		     reinterpret_cast<const cuComplex*>(beta),
                     reinterpret_cast<cuComplex*>(c_mem), c_inc);
}

cublasStatus_t wrapperCusparseCSRMM(cusparseHandle_t &handle,
				    cusparseOperation_t &cusparse_a_transpose,
				    int m, int n, int k,
				    int NNZ,
				    float *alpha,
				    const cusparseMatDescr_t descrA,
				    const float *a_values_mem,
				    const int *a_first_index_mem,
				    const int *a_indices_mem,
				    const float *b_mem,
				    unsigned int b_inc,
				    float *beta,
				    float *c_mem,
				    unsigned int c_inc) {
  return cusparseScsrmm(handle, cusparse_a_transpose,
			m, n, k,
			NNZ,
			alpha,
			descrA,
			a_values_mem, a_first_index_mem, a_indices_mem,
			b_mem, b_inc,
			beta, c_mem, c_inc);
}

cublasStatus_t wrapperCusparseCSRMM(cusparseHandle_t &handle,
				    cusparseOperation_t &cusparse_a_transpose,
				    int m, int n, int k,
				    int NNZ,
				    double *alpha,
				    const cusparseMatDescr_t descrA,
				    const double *a_values_mem,
				    const int *a_first_index_mem,
				    const int *a_indices_mem,
				    const double *b_mem,
				    unsigned int b_inc,
				    double *beta,
				    double *c_mem,
				    unsigned int c_inc) {
  return cusparseDcsrmm(handle, cusparse_a_transpose,
			m, n, k,
			NNZ,
			alpha,
			descrA,
			a_values_mem, a_first_index_mem, a_indices_mem,
			b_mem, b_inc,
			beta, c_mem, c_inc);
}

cublasStatus_t wrapperCusparseCSRMM(cusparseHandle_t &handle,
				    cusparseOperation_t &cusparse_a_transpose,
				    int m, int n, int k,
				    int NNZ,
				    ComplexF *alpha,
				    const cusparseMatDescr_t descrA,
				    const ComplexF *a_values_mem,
				    const int *a_first_index_mem,
				    const int *a_indices_mem,
				    const ComplexF *b_mem,
				    unsigned int b_inc,
				    ComplexF *beta,
				    ComplexF *c_mem,
				    unsigned int c_inc) {
  return cusparseCcsrmm(handle, cusparse_a_transpose,
			m, n, k,
			NNZ,
			reinterpret_cast<const cuComplex*>(alpha),
			descrA,
			reinterpret_cast<const cuComplex*>(a_values_mem),
			a_first_index_mem, a_indices_mem,
			reinterpret_cast<const cuComplex*>(b_mem), b_inc,
			reinterpret_cast<const cuComplex*>(beta),
			reinterpret_cast<const cuComplex*>(c_mem), c_inc);
}

#endif

/***************************************
 ************* CBLAS SECTION ***********
 ***************************************/

void wrapperCblasGemm(CBLAS_ORDER &major_type,
		      CBLAS_TRANSPOSE a_transpose,
		      CBLAS_TRANSPOSE b_transpose,
		      int m, int n, int k,
		      float alpha,
		      const float *a_mem, unsigned int a_inc,
		      const float *b_mem, unsigned int b_inc,
		      float beta, float *c_mem, unsigned int c_inc) {
  cblas_sgemm(major_type, a_transpose, b_transpose,
	      m, n, k,
	      alpha, a_mem, a_inc,
	      b_mem, b_inc,
	      beta, c_mem, c_inc);
}

void wrapperCblasGemm(CBLAS_ORDER &major_type,
		      CBLAS_TRANSPOSE a_transpose,
		      CBLAS_TRANSPOSE b_transpose,
		      int m, int n, int k,
		      double alpha,
		      const double *a_mem, unsigned int a_inc,
		      const double *b_mem, unsigned int b_inc,
		      double beta, double *c_mem, unsigned int c_inc) {
  cblas_dgemm(major_type, a_transpose, b_transpose,
	      m, n, k,
	      alpha, a_mem, a_inc,
	      b_mem, b_inc,
	      beta, c_mem, c_inc);
}

void wrapperCblasGemm(CBLAS_ORDER &major_type,
		      CBLAS_TRANSPOSE a_transpose,
		      CBLAS_TRANSPOSE b_transpose,
		      int m, int n, int k,
		      ComplexF alpha,
		      const ComplexF *a_mem, unsigned int a_inc,
		      const ComplexF *b_mem, unsigned int b_inc,
		      ComplexF beta, ComplexF *c_mem, unsigned int c_inc) {
  cblas_cgemm(major_type, a_transpose, b_transpose,
	      m, n, k,
	      &alpha, a_mem, a_inc,
	      b_mem, b_inc,
	      &beta, c_mem, c_inc);
}

template <typename T>
void wrapperCblasSparseMM(SPARSE_FORMAT sparse_format,
			  CBLAS_TRANSPOSE a_transpose,
			  int m, int n, int k,
			  T alpha,
			  const T *a_values_mem,
			  const int *a_indices_mem,
			  const int *a_first_index_mem,
			  const T *b_mem, unsigned int b_inc,
			  T beta, T *c_mem, unsigned int c_inc) {
  cblas_sparse_mm(sparse_format, a_transpose,
                  m, n, k,
                  alpha, a_values_mem, a_indices_mem, a_first_index_mem,
                  b_mem, static_cast<int>(b_inc),
                  beta, c_mem, static_cast<int>(c_inc));
}

/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/

template <typename T>
void doGemm(CBLAS_ORDER major_type,
	    CBLAS_TRANSPOSE a_transpose,
	    CBLAS_TRANSPOSE b_transpose,
	    int m,
	    int n,
	    int k,
	    T alpha,
	    const GPUMirroredMemoryBlock<T>* a,
	    unsigned int a_inc,
	    const GPUMirroredMemoryBlock<T>* b,
	    unsigned int b_inc,
	    T beta,
	    GPUMirroredMemoryBlock<T>* c,
	    unsigned int c_inc,
	    unsigned int a_shift,
	    unsigned int b_shift,
	    unsigned int c_shift,
	    bool use_gpu)
{
  const T *a_mem, *b_mem;
  T *c_mem;
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    assert(major_type == CblasColMajor);
    //printf("Doing a sgemm with comp=1 & cuda=1\n");
    a_mem = a->getGPUForRead() + a_shift;
    b_mem = b->getGPUForRead() + b_shift;
    c_mem = c->getGPUForReadAndWrite() + c_shift;
    cublasOperation_t cublas_a_transpose = getCublasOperation(a_transpose);
    cublasOperation_t cublas_b_transpose = getCublasOperation(b_transpose);

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = wrapperCublasGemm(handle, cublas_a_transpose, cublas_b_transpose,
			       m, n, k,
			       &alpha, a_mem, a_inc,
			       b_mem, b_inc,
			       &beta, c_mem, c_inc);

    checkCublasError(status);
  }
  else {
    //printf("Doing a sgemm with comp=1 & cuda=0\n");
#endif
    //printf("Doing a sgemm with comp=0 & cuda=0\n");
    a_mem = a->getPPALForRead() + a_shift;
    b_mem = b->getPPALForRead() + b_shift;
    c_mem = c->getPPALForReadAndWrite() + c_shift;

    // matrix matrix product: C = \alpha op(A) op(B) + \beta C
    wrapperCblasGemm(major_type,   // Row or Col Major
		     a_transpose,  // Transpose or not A
		     b_transpose,  // Transpose or not B
		     m,            // num rows of A (before transpose)
		     n,            // num rows at B (before transpose)
		     k,            // Common dimension between A and B
		     alpha,        // Alpha value
		     a_mem,        // A matrix
		     a_inc,        // A matrix stride
		     b_mem,        // B matrix
		     b_inc,        // B matrix stride
		     beta,         // Beta value
		     c_mem,        // C matrix
		     c_inc);       // C matrix stride
#ifdef USE_CUDA
  }
#endif
}

template <typename T>
void doSparseMM(CBLAS_ORDER major_order,
		SPARSE_FORMAT sparse_format,
		CBLAS_TRANSPOSE a_transpose,
		int m,
		int n,
		int k,
		T alpha,
		const GPUMirroredMemoryBlock<T>* a_values,
		const Int32GPUMirroredMemoryBlock* a_indices,
		const Int32GPUMirroredMemoryBlock* a_first_index,
		const GPUMirroredMemoryBlock<T>* b,
		int b_inc,
		T beta,
		GPUMirroredMemoryBlock<T>* c,
		int c_inc,
		int b_shift,
		int c_shift,
		bool use_gpu)
{
  const T *a_values_mem, *b_mem;
  const int *a_indices_mem, *a_first_index_mem;
  T *c_mem;
  const int NNZ = static_cast<int>(a_values->getSize());
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    cusparseStatus_t status;
    cusparseHandle_t handle = GPUHelper::getSparseHandler();
    if (major_order != CblasColMajor)
      ERROR_EXIT(128, "Column major matrices are expected\n");
    if (sparse_format != SparseMatrix<T>::CSR_FORMAT)
      a_transpose = NEGATE_CBLAS_TRANSPOSE(a_transpose);
    //printf("Doing a sgemm with comp=1 & cuda=1\n");
    a_values_mem = a_values->getGPUForRead();
    a_indices_mem = a_indices->getGPUForRead();
    a_first_index_mem = a_first_index->getGPUForRead();
    b_mem = b->getGPUForRead() + b_shift;
    c_mem = c->getGPUForReadAndWrite() + c_shift;
    cusparseOperation_t cusparse_a_transpose = getCusparseOperation(a_transpose);
    
    status = cusparseSetStream(handle, GPUHelper::getCurrentStream());
    checkCusparseError(status);
    cusparseMatDescr_t descrA = {
      CUSPARSE_MATRIX_TYPE_GENERAL,
      0, // fill mode
      0, // diag type
      CUSPARSE_INDEX_BASE_ZERO
    };
    status = wrapperCusparseCSRMM(handle,
				  cusparse_a_transpose,
				  m, n, k,
				  NNZ,
				  &alpha,
				  descrA,
				  a_values_mem,
				  a_first_index_mem,
				  a_indices_mem,
				  b_mem, b_inc,
				  &beta, c_mem, c_inc);
    checkCusparseError(status);
  }
  else {
    //printf("Doing a sgemm with comp=1 & cuda=0\n");
#endif
    if (major_order != CblasRowMajor)
      ERROR_EXIT(128, "Row major matrices are expected\n");
    //printf("Doing a sgemm with comp=0 & cuda=0\n");
    a_values_mem = a_values->getPPALForRead();
    a_indices_mem = a_indices->getPPALForRead();
    a_first_index_mem = a_first_index->getPPALForRead();
    b_mem = b->getPPALForRead() + b_shift;
    c_mem = c->getPPALForReadAndWrite() + c_shift;
    // matrix matrix product: C = \alpha op(A) op(B) + \beta C
    wrapperCblasSparseMM(sparse_format,
			 a_transpose,
			 m,            // num rows of A (before transpose)
			 n,            // num rows at B (before transpose)
			 k,            // Common dimension between A and B
			 alpha,        // Alpha value
			 a_values_mem,
			 a_indices_mem,
			 a_first_index_mem,
			 b_mem,        // B matrix
			 b_inc,        // B matrix stride
			 beta,         // Beta value
			 c_mem,        // C matrix
			 c_inc);       // C matrix stride
#ifdef USE_CUDA
  }
#endif
}

template void doGemm<float>(CBLAS_ORDER major_type,
			    CBLAS_TRANSPOSE a_transpose,
			    CBLAS_TRANSPOSE b_transpose,
			    int m,
			    int n,
			    int k,
			    float alpha,
			    const GPUMirroredMemoryBlock<float>* a,
			    unsigned int a_inc,
			    const GPUMirroredMemoryBlock<float>* b,
			    unsigned int b_inc,
			    float beta,
			    GPUMirroredMemoryBlock<float>* c,
			    unsigned int c_inc,
			    unsigned int a_shift,
			    unsigned int b_shift,
			    unsigned int c_shift,
			    bool use_gpu);

template void doGemm<double>(CBLAS_ORDER major_type,
			     CBLAS_TRANSPOSE a_transpose,
			     CBLAS_TRANSPOSE b_transpose,
			     int m,
			     int n,
			     int k,
			     double alpha,
			     const GPUMirroredMemoryBlock<double>* a,
			     unsigned int a_inc,
			     const GPUMirroredMemoryBlock<double>* b,
			     unsigned int b_inc,
			     double beta,
			     GPUMirroredMemoryBlock<double>* c,
			     unsigned int c_inc,
			     unsigned int a_shift,
			     unsigned int b_shift,
			     unsigned int c_shift,
			     bool use_gpu);

template void doGemm<ComplexF>(CBLAS_ORDER major_type,
			       CBLAS_TRANSPOSE a_transpose,
			       CBLAS_TRANSPOSE b_transpose,
			       int m,
			       int n,
			       int k,
			       ComplexF alpha,
			       const GPUMirroredMemoryBlock<ComplexF>* a,
			       unsigned int a_inc,
			       const GPUMirroredMemoryBlock<ComplexF>* b,
			       unsigned int b_inc,
			       ComplexF beta,
			       GPUMirroredMemoryBlock<ComplexF>* c,
			       unsigned int c_inc,
			       unsigned int a_shift,
			       unsigned int b_shift,
			       unsigned int c_shift,
			       bool use_gpu);

template void doSparseMM<float>(CBLAS_ORDER major_order,
                                SPARSE_FORMAT sparse_format,
                                CBLAS_TRANSPOSE a_transpose,
                                int m,
                                int n,
                                int k,
                                float alpha,
                                const GPUMirroredMemoryBlock<float>* a_values,
                                const Int32GPUMirroredMemoryBlock* a_indices,
                                const Int32GPUMirroredMemoryBlock* a_first_index,
                                const GPUMirroredMemoryBlock<float>* b,
                                int b_inc,
                                float beta,
                                GPUMirroredMemoryBlock<float>* c,
                                int c_inc,
                                int b_shift,
                                int c_shift,
                                bool use_gpu);

template void doSparseMM<double>(CBLAS_ORDER major_order,
                                 SPARSE_FORMAT sparse_format,
                                 CBLAS_TRANSPOSE a_transpose,
                                 int m,
                                 int n,
                                 int k,
                                 double alpha,
                                 const GPUMirroredMemoryBlock<double>* a_values,
                                 const Int32GPUMirroredMemoryBlock* a_indices,
                                 const Int32GPUMirroredMemoryBlock* a_first_index,
                                 const GPUMirroredMemoryBlock<double>* b,
                                 int b_inc,
                                 double beta,
                                 GPUMirroredMemoryBlock<double>* c,
                                 int c_inc,
                                 int b_shift,
                                 int c_shift,
                                 bool use_gpu);

template void doSparseMM<ComplexF>(CBLAS_ORDER major_order,
                                   SPARSE_FORMAT sparse_format,
                                   CBLAS_TRANSPOSE a_transpose,
                                   int m,
                                   int n,
                                   int k,
                                   ComplexF alpha,
                                   const GPUMirroredMemoryBlock<ComplexF>* a_values,
                                   const Int32GPUMirroredMemoryBlock* a_indices,
                                   const Int32GPUMirroredMemoryBlock* a_first_index,
                                   const GPUMirroredMemoryBlock<ComplexF>* b,
                                   int b_inc,
                                   ComplexF beta,
                                   GPUMirroredMemoryBlock<ComplexF>* c,
                                   int c_inc,
                                   int b_shift,
                                   int c_shift,
                                   bool use_gpu);
