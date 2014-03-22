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
	    GPUMirroredMemoryBlock<T>* a,
	    unsigned int a_inc,
	    GPUMirroredMemoryBlock<T>* b,
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

template void doGemm<float>(CBLAS_ORDER major_type,
			    CBLAS_TRANSPOSE a_transpose,
			    CBLAS_TRANSPOSE b_transpose,
			    int m,
			    int n,
			    int k,
			    float alpha,
			    GPUMirroredMemoryBlock<float>* a,
			    unsigned int a_inc,
			    GPUMirroredMemoryBlock<float>* b,
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
			     GPUMirroredMemoryBlock<double>* a,
			     unsigned int a_inc,
			     GPUMirroredMemoryBlock<double>* b,
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
			       GPUMirroredMemoryBlock<ComplexF>* a,
			       unsigned int a_inc,
			       GPUMirroredMemoryBlock<ComplexF>* b,
			       unsigned int b_inc,
			       ComplexF beta,
			       GPUMirroredMemoryBlock<ComplexF>* c,
			       unsigned int c_inc,
			       unsigned int a_shift,
			       unsigned int b_shift,
			       unsigned int c_shift,
			       bool use_gpu);
