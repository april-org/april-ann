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

cublasStatus_t wrapperCublasSbmv(cublasHandle_t &handle,
				 cublasFillMode_t uplo_cublas,
				 int n, int k,
				 float *alpha,
				 const float *a_mem, unsigned int a_lda,
				 const float *x_mem, unsigned int x_inc,
				 float *beta,
				 float *y_mem, unsigned int y_inc) {
  return cublasSsbmv(handle, uplo_cublas,
		     n, k,
		     alpha, a_mem, a_lda,
		     x_mem, x_inc,
		     beta, y_mem, y_inc);
}
#endif

/***************************************
 ************* CBLAS SECTION ***********
 ***************************************/

void wrapperCblasSbmv(CBLAS_ORDER major_type,
		      CBLAS_UPLO uplo,
		      int n, int k, float alpha,
		      const float *a_mem, unsigned int a_lda,
		      const float *x_mem, unsigned int x_inc,
		      float beta,
		      float *y_mem, unsigned int y_inc) {
  cblas_ssbmv(major_type, uplo,
	      n, k,
	      alpha, a_mem, a_lda,
	      x_mem, x_inc,
	      beta, y_mem, y_inc);
}

/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/

template<typename T>
void doSbmv(CBLAS_ORDER major_type,
	    CBLAS_UPLO uplo,
	    int n, int k,
	    T alpha, GPUMirroredMemoryBlock<T> *a, unsigned int a_lda,
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
    a_mem = a->getGPUForRead() + a_shift;
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForReadAndWrite() + y_shift;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);
    cublasFillMode_t uplo_cublas = CUBLAS_FILL_MODE_UPPER;
    if (uplo == CblasLower) uplo_cublas = CUBLAS_FILL_MODE_LOWER;
    status = wrapperCublasSbmv(handle, uplo_cublas,
			       n, k,
			       &alpha, a_mem, a_lda,
			       x_mem, x_inc,
			       &beta, y_mem, y_inc);
    checkCublasError(status);
  }
  else {
#endif
    a_mem = a->getPPALForRead() + a_shift;
    x_mem = x->getPPALForRead() + x_shift;
    y_mem = y->getPPALForReadAndWrite() + y_shift;

    wrapperCblasSbmv(major_type, uplo,
		     n, k,
		     alpha, a_mem, a_lda,
		     x_mem, x_inc,
		     beta, y_mem, y_inc);
#ifdef USE_CUDA
  }
#endif
}

template void doSbmv<float>(CBLAS_ORDER major_type,
			    CBLAS_UPLO uplo,
			    int n, int k,
			    float alpha, GPUMirroredMemoryBlock<float> *a, unsigned int a_lda,
			    GPUMirroredMemoryBlock<float> *x, unsigned int x_inc,
			    float beta, GPUMirroredMemoryBlock<float> *y, unsigned int y_inc,
			    unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
			    bool use_gpu);
