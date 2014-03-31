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

cublasStatus_t wrapperCublasNrm2(cublasHandle_t &handle,
				 unsigned int n,
				 const float *x_mem,
				 unsigned int x_inc,
				 float *result) {
  return cublasSnrm2(handle, n, x_mem, x_inc, result);
}

cublasStatus_t wrapperCublasNrm2(cublasHandle_t &handle,
				 unsigned int size,
				 const ComplexF *x_mem,
				 unsigned int x_inc,
				 float *result) {
  UNUSED_VARIABLE(handle);
  UNUSED_VARIABLE(size);
  UNUSED_VARIABLE(x_mem);
  UNUSED_VARIABLE(x_inc);
  UNUSED_VARIABLE(result);
  ERROR_EXIT(256, "Nrm2 for complex numbers not implemented in CUDA\n");
  return CUBLAS_STATUS_INTERNAL_ERROR;
}

#endif

/***************************************
 ************* CBLAS SECTION ***********
 ***************************************/

float wrapperCblasNrm2(unsigned int size,
		       const float *x_mem, unsigned int x_inc) {
  return cblas_snrm2(size, x_mem, x_inc);
}

float wrapperCblasNrm2(unsigned int size,
		       const ComplexF *x_mem, unsigned int x_inc) {
  return cblas_scnrm2(size, x_mem, x_inc);
}

/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/

template <typename T>
float doNrm2(unsigned int n,
	     const GPUMirroredMemoryBlock<T> *x,
	     unsigned int inc,
	     unsigned int shift,
	     bool use_gpu) {
  float result;
  const T *x_mem;
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    x_mem  = x->getGPUForRead() + shift;
    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);
    status = wrapperCublasNrm2(handle, n, x_mem, inc, &result);
    checkCublasError(status);
  }
  else {
#endif
    x_mem = x->getPPALForRead() + shift;
    result = wrapperCblasNrm2(n, x_mem, inc);
#ifdef USE_CUDA
  }
#endif
  return result;
}

template float doNrm2<float>(unsigned int n,
			     const GPUMirroredMemoryBlock<float> *x,
			     unsigned int inc,
			     unsigned int shift,
			     bool use_gpu);

template float doNrm2<ComplexF>(unsigned int n,
				const GPUMirroredMemoryBlock<ComplexF> *x,
				unsigned int inc,
				unsigned int shift,
				bool use_gpu);
