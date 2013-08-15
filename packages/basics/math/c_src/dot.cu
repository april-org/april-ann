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

cublasStatus_t wrapperCublasDot(cublasHandle_t &handle,
				unsigned int size,
				const float *x_mem,
				unsigned int x_inc,
				const float *y_mem,
				unsigned int y_inc,
				float *ret) {
  return cublasSdot(handle,
		    size,
		    x_mem, x_inc,
		    y_mem, y_inc,
		    ret);
}

cublasStatus_t wrapperCublasDot(cublasHandle_t &handle,
				unsigned int size,
				const ComplexF *x_mem,
				unsigned int x_inc,
				const ComplexF *y_mem,
				unsigned int y_inc,
				ComplexF *ret) {
  return cublasScdotu(handle,
		      size,
		      x_mem, x_inc,
		      y_mem, y_inc,
		      ret);
}

#endif

/***************************************
 ************* CBLAS SECTION ***********
 ***************************************/

float wrapperCblasDot(CBLAS_ORDER &major_type,
		      unsigned int size,
		      const float *x_mem, unsigned int x_inc,
		      const float *y_mem, unsigned int y_inc) {
  return cblas_sdot(size,
		    x_mem, x_inc,
		    y_mem, y_inc);
}

ComplexF wrapperCblasDot(CBLAS_ORDER &major_type,
			 unsigned int size,
			 const ComplexF *x_mem, unsigned int x_inc,
			 const ComplexF *y_mem, unsigned int y_inc) {
  ComplexF ret;
  cblas_zdotu_sub(size,
		  x_mem, x_inc,
		  y_mem, y_inc,
		  &ret);
  return ret;
}

/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/

template <typename T>
T doDot(unsigned int size,
	const GPUMirroredMemoryBlock<T> *x,
	unsigned int x_shift,
	unsigned int x_inc,
	const GPUMirroredMemoryBlock<T> *y,
	unsigned int y_shift,
	unsigned int y_inc,
	bool use_gpu) {
  const T *x_mem;
  const T *y_mem;
  T ret;
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForRead() + y_shift;
    
    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);
    status = wrapperCublasDot(handle, size, x_mem, x_inc, y_mem, y_inc, &ret);
    checkCublasError(status);
  }
  else {
#endif
    x_mem = x->getPPALForRead() + x_shift;
    y_mem = y->getPPALForRead() + y_shift;
    
    ret = wrapperCblasDot(size,
			  x_mem, x_inc,
			  y_mem, y_inc);
#ifdef USE_CUDA
  }
#endif
  return ret;
}
