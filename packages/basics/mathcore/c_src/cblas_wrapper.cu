/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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
#include <cmath>
#include "unused_variable.h"
#include "wrapper.h"
#include "cuda_utils.h"

#define fsignf(x) ((x)<0.0f) ? (-1.0f) : ( ((x)>0.0f) ? (1.0f) : (0.0f) )

///////////////////////////////////////////////////////////
/////////////////// Kernels ///////////////////////////////
///////////////////////////////////////////////////////////

#ifdef USE_CUDA

#define CWISE_FUNC_KERNEL(func) template<typename T>		\
  __global__ void						\
  func##FuncKernel(T *v, unsigned int N, unsigned int stride) {	\
    unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;	\
    if (x_idx < N) {						\
      T *aux = v + x_idx*stride;				\
      *aux = func(*aux);					\
    }								\
  }

CWISE_FUNC_KERNEL(logf);
CWISE_FUNC_KERNEL(log1pf);
CWISE_FUNC_KERNEL(expf);
CWISE_FUNC_KERNEL(sqrtf);
CWISE_FUNC_KERNEL(tanf);
CWISE_FUNC_KERNEL(tanhf);
CWISE_FUNC_KERNEL(atanf);
CWISE_FUNC_KERNEL(atanhf);
CWISE_FUNC_KERNEL(sinf);
CWISE_FUNC_KERNEL(sinhf);
CWISE_FUNC_KERNEL(asinf);
CWISE_FUNC_KERNEL(asinhf);
CWISE_FUNC_KERNEL(cosf);
CWISE_FUNC_KERNEL(coshf);
CWISE_FUNC_KERNEL(acosf);
CWISE_FUNC_KERNEL(acoshf);
CWISE_FUNC_KERNEL(fabsf);
CWISE_FUNC_KERNEL(fsignf);

#undef CWISE_FUNC_KERNEL

template<typename T>
__global__ void complementFuncKernel(T *v, unsigned int N, unsigned int stride) {
  unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (x_idx < N) {
    T *aux = v + x_idx*stride;
    *aux = (1.0f - *aux);
  }
}

#define CWISE_FUNC_KERNEL(func) template<typename T>		\
  __global__ void						\
  func##FuncKernel(T *v, unsigned int N, unsigned int stride,	\
		   T value) {					\
    unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;	\
    if (x_idx < N) {						\
      T *aux = v + x_idx*stride;				\
      *aux = func(*aux, value);					\
    }								\
  }

CWISE_FUNC_KERNEL(powf);

#undef CWISE_FUNC_KERNEL

#endif

///////////////////////////////////////////////////////////
//////////////////// BLAS wrappers ////////////////////////
///////////////////////////////////////////////////////////

void doPLogP(unsigned int N,
	     FloatGPUMirroredMemoryBlock *v,
	     unsigned int stride,
	     unsigned int shift,
	     bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    ERROR_PRINT("CUDA VERSION NOT IMPLEMENTED\n");
  }
  //  else {
#endif
  float *v_mem = v->getPPALForReadAndWrite() + shift;
  for (unsigned int i=0; i<N; ++i, v_mem += stride)
    if (*v_mem > 0.0f || *v_mem < 0.0f) *v_mem = (*v_mem) * logf(*v_mem);
#ifdef USE_CUDA
  //  }
#endif  
}

void doLog(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    logfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float log_NZ = logf(NEAR_ZERO);
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) {
      // TODO: implement this sanity check at CUBLAS
      if (*v_mem > NEAR_ZERO || *v_mem < -NEAR_ZERO) *v_mem = logf(*v_mem);
      else *v_mem = log_NZ;
    }
#ifdef USE_CUDA
  }
#endif  
}

void doLog1p(unsigned int N,
	     FloatGPUMirroredMemoryBlock *v,
	     unsigned int stride,
	     unsigned int shift,
	     bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    log1pfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = log1pf(*v_mem);
#ifdef USE_CUDA
  }
#endif  
}

void doExp(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    expfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = expf(*v_mem);
#ifdef USE_CUDA
  }
#endif  
}

void doSqrt(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    sqrtfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = sqrtf(*v_mem);
#ifdef USE_CUDA
  }
#endif  
}

void doTan(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    tanfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = tanf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doTanh(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    tanhfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = tanhf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doAtan(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    atanfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = atanf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doAtanh(unsigned int N,
	     FloatGPUMirroredMemoryBlock *v,
	     unsigned int stride,
	     unsigned int shift,
	     bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    atanhfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = atanhf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doSin(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    sinfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = sinf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doSinh(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    sinhfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = sinhf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doAsin(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    asinfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = asinf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doAsinh(unsigned int N,
	     FloatGPUMirroredMemoryBlock *v,
	     unsigned int stride,
	     unsigned int shift,
	     bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    asinhfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = asinhf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doCos(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    cosfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = cosf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doCosh(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    coshfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = coshf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doAcos(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    acosfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = acosf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doAcosh(unsigned int N,
	     FloatGPUMirroredMemoryBlock *v,
	     unsigned int stride,
	     unsigned int shift,
	     bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    acoshfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = acoshf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doAbs(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    fabsfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = fabsf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doComplement(unsigned int N,
		  FloatGPUMirroredMemoryBlock *v,
		  unsigned int stride,
		  unsigned int shift,
		  bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    complementFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = (1.0f - *v_mem);
#ifdef USE_CUDA
  }
#endif
}

void doSign(unsigned int N,
	    FloatGPUMirroredMemoryBlock *v,
	    unsigned int stride,
	    unsigned int shift,
	    bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    fsignfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride) *v_mem = fsignf(*v_mem);
#ifdef USE_CUDA
  }
#endif
}


void doPow(unsigned int N,
	   FloatGPUMirroredMemoryBlock *v,
	   unsigned int stride,
	   unsigned int shift,
	   float value,
	   bool use_gpu) {
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    float *v_ptr = v->getGPUForReadAndWrite() + shift;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    powfFuncKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride, value);
  }
  else {
#endif
    float *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem += stride)
      *v_mem = powf(*v_mem, value);
#ifdef USE_CUDA
  }
#endif
}
