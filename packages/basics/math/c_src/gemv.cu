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

#ifdef USE_CUDA
/***************************************
 ************** CUDA SECTION ***********
 ***************************************/

cublasStatus_t wrapperCublasGemv(cublasHandle_t &handle,
				 cublasOperation_t &cublas_a_transpose,
				 int m, int n,
				 const float *alpha,
				 const float *a_mem,
				 unsigned int a_inc,
				 const float *x_mem,
				 unsigned int x_inc,
				 const float *beta,
				 float *y_mem,
				 unsigned int y_inc) {
  return cublasSgemv(handle, cublas_a_transpose,
		     m, n,
		     alpha, a_mem, a_inc,
		     x_mem, x_inc,
		     beta, y_mem, y_inc);
}

cublasStatus_t wrapperCublasGemv(cublasHandle_t &handle,
				 cublasOperation_t &cublas_a_transpose,
				 int m, int n,
				 const ComplexF *alpha,
				 const ComplexF *a_mem,
				 unsigned int a_inc,
				 const ComplexF *x_mem,
				 unsigned int x_inc,
				 const ComplexF *beta,
				 ComplexF *y_mem,
				 unsigned int y_inc) {
  return cublasCgemv(handle, cublas_a_transpose,
		     m, n,
		     reinterpret_cast<const cuComplex*>(alpha),
                     reinterpret_cast<const cuComplex*>(a_mem), a_inc,
		     reinterpret_cast<const cuComplex*>(x_mem), x_inc,
		     reinterpret_cast<const cuComplex*>(beta),
                     reinterpret_cast<cuComplex*>(y_mem), y_inc);
}
#endif

/***************************************
 ************* CBLAS SECTION ***********
 ***************************************/

void wrapperCblasGemv(CBLAS_ORDER &major_type,
		      CBLAS_TRANSPOSE a_transpose,
		      int m, int n,
		      float alpha,
		      const float *a_mem, unsigned int a_inc,
		      const float *x_mem, unsigned int x_inc,
		      float beta,
		      float *y_mem, unsigned int y_inc) {
  cblas_sgemv(major_type, a_transpose, m, n, alpha, a_mem, a_inc,
	      x_mem, x_inc, beta, y_mem, y_inc);
}

void wrapperCblasGemv(CBLAS_ORDER &major_type,
		      CBLAS_TRANSPOSE a_transpose,
		      int m, int n,
		      ComplexF alpha,
		      const ComplexF *a_mem, unsigned int a_inc,
		      const ComplexF *x_mem, unsigned int x_inc,
		      ComplexF beta,
		      ComplexF *y_mem, unsigned int y_inc) {
  cblas_cgemv(major_type, a_transpose, m, n, &alpha, a_mem, a_inc,
	      x_mem, x_inc, &beta, y_mem, y_inc);
}


/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/

template<typename T>
void doGemv(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
	    int m, int n,
	    T alpha, GPUMirroredMemoryBlock<T> *a, unsigned int a_inc,
	    GPUMirroredMemoryBlock<T> *x, unsigned int x_inc,
	    T beta, GPUMirroredMemoryBlock<T> *y, unsigned int y_inc,
	    unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
	    bool use_gpu) {
  const T *a_mem, *x_mem;
  T *y_mem;
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    assert(major_type == CblasColMajor);
    cublasOperation_t cublas_a_transpose = getCublasOperation(a_transpose);
    a_mem = a->getGPUForRead() + a_shift;
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForReadAndWrite() + y_shift;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = wrapperCublasGemv(handle, cublas_a_transpose,
			       m, n,
			       &alpha, a_mem, a_inc,
			       x_mem, x_inc,
			       &beta, y_mem, y_inc);
    
    checkCublasError(status);
  }
  else {
#endif
    a_mem = a->getPPALForRead() + a_shift;
    x_mem = x->getPPALForRead() + x_shift;
    y_mem = y->getPPALForReadAndWrite() + y_shift;
    wrapperCblasGemv(major_type, a_transpose,
                m, n,
                alpha, a_mem, a_inc,
                x_mem, x_inc,
                beta, y_mem, y_inc);
#ifdef USE_CUDA
  }
#endif
}

template void doGemv<float>(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
			    int m, int n,
			    float alpha, GPUMirroredMemoryBlock<float> *a, unsigned int a_inc,
			    GPUMirroredMemoryBlock<float> *x, unsigned int x_inc,
			    float beta, GPUMirroredMemoryBlock<float> *y, unsigned int y_inc,
			    unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
			    bool use_gpu);

template void doGemv<ComplexF>(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
			       int m, int n,
			       ComplexF alpha, GPUMirroredMemoryBlock<ComplexF> *a, unsigned int a_inc,
			       GPUMirroredMemoryBlock<ComplexF> *x, unsigned int x_inc,
			       ComplexF beta, GPUMirroredMemoryBlock<ComplexF> *y, unsigned int y_inc,
			       unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
			       bool use_gpu);
