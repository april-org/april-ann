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

cublasStatus_t wrapperCublasGer(cublasHandle_t &handle,
				unsigned int m, unsigned int n,
				float *alpha,
				const float *x_mem,
				unsigned int x_inc,
				const float *y_mem,
				unsigned int y_inc,
				float *a_mem,
				unsigned int a_inc) {
  return cublasSger(handle, m, n, alpha, x_mem, x_inc,
		    y_mem, y_inc,
		    a_mem, a_inc);
}

cublasStatus_t wrapperCublasGer(cublasHandle_t &handle,
				unsigned int m, unsigned int n,
				ComplexF *alpha,
				const ComplexF *x_mem,
				unsigned int x_inc,
				const ComplexF *y_mem,
				unsigned int y_inc,
				ComplexF *a_mem,
				unsigned int a_inc) {
  ERROR_EXIT(256, "Ger operation not implemented in CUDA\n");
  return CUBLAS_STATUS_INTERNAL_ERROR;
}
#endif

/***************************************
 ************* CBLAS SECTION ***********
 ***************************************/

void wrapperCblasGer(CBLAS_ORDER major_type,
		     int m, int n, float alpha,
		     const float *x_mem, unsigned int x_inc,
		     const float *y_mem, unsigned int y_inc,
		     float *a_mem, unsigned int a_inc) {
  cblas_sger(major_type,
	     m, n,
	     alpha,
	     x_mem, x_inc,
	     y_mem, y_inc,
	     a_mem, a_inc);
}

void wrapperCblasGer(CBLAS_ORDER major_type,
		     int m, int n, ComplexF alpha,
		     const ComplexF *x_mem, unsigned int x_inc,
		     const ComplexF *y_mem, unsigned int y_inc,
		     ComplexF *a_mem, unsigned int a_inc) {
  cblas_cgeru(major_type,
	      m, n,
	      &alpha,
	      x_mem, x_inc,
	      y_mem, y_inc,
	      a_mem, a_inc);
}

/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/

template<typename T>
void doGer(CBLAS_ORDER major_type,
	   unsigned int m,
	   unsigned int n,
	   T alpha,
	   GPUMirroredMemoryBlock<T> *x,
	   unsigned int x_shift,
	   unsigned int x_inc,
	   GPUMirroredMemoryBlock<T> *y,
	   unsigned int y_shift,
	   unsigned int y_inc,
	   GPUMirroredMemoryBlock<T> *a,
	   unsigned int a_shift,
	   unsigned int a_inc,
	   bool use_gpu) {
  const T *x_mem;
  const T *y_mem;
  T *a_mem;
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    assert(major_type == CblasColMajor);
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForRead() + y_shift;
    a_mem = a->getGPUForReadAndWrite() + a_shift;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = wrapperCublasGer(handle,
			      m, n,
			      &alpha,
			      x_mem, x_inc,
			      y_mem, y_inc,
			      a_mem, a_inc);
    
    checkCublasError(status);
  }
  else {
#endif
    x_mem = x->getPPALForRead() + x_shift;
    y_mem = y->getPPALForRead() + y_shift;
    a_mem = a->getPPALForReadAndWrite() + a_shift;

    wrapperCblasGer(major_type,
		    m, n,
		    alpha,
		    x_mem, x_inc,
		    y_mem, y_inc,
		    a_mem, a_inc);
#ifdef USE_CUDA
  }
#endif
}

template void doGer<float>(CBLAS_ORDER major_type,
			   unsigned int m,
			   unsigned int n,
			   float alpha,
			   GPUMirroredMemoryBlock<float> *x,
			   unsigned int x_shift,
			   unsigned int x_inc,
			   GPUMirroredMemoryBlock<float> *y,
			   unsigned int y_shift,
			   unsigned int y_inc,
			   GPUMirroredMemoryBlock<float> *a,
			   unsigned int a_shift,
			   unsigned int a_inc,
			   bool use_gpu);

template void doGer<ComplexF>(CBLAS_ORDER major_type,
			      unsigned int m,
			      unsigned int n,
			      ComplexF alpha,
			      GPUMirroredMemoryBlock<ComplexF> *x,
			      unsigned int x_shift,
			      unsigned int x_inc,
			      GPUMirroredMemoryBlock<ComplexF> *y,
			      unsigned int y_shift,
			      unsigned int y_inc,
			      GPUMirroredMemoryBlock<ComplexF> *a,
			      unsigned int a_shift,
			      unsigned int a_inc,
			      bool use_gpu);
