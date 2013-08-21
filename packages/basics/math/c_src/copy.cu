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

cublasStatus_t wrapperCublasCopy(cublasHandle_t &handle,
				 int N,
				 const float *x_mem,
				 unsigned int x_inc,
				 float *y_mem,
				 unsigned int y_inc) {
  return cublasScopy(handle, N, x_mem, x_inc, y_mem, y_inc);
}

cublasStatus_t wrapperCublasCopy(cublasHandle_t &handle,
				 int N,
				 const ComplexF *x_mem,
				 unsigned int x_inc,
				 ComplexF *y_mem,
				 unsigned int y_inc) {
  return cublasCcopy(handle, N, x_mem, x_inc, y_mem, y_inc);
}

template<typename T>
__global__ void copyLoopKernel(unsigned int N,
			       const T *x_mem,
			       unsigned int x_inc,
			       T *y_mem,
			       unsigned int y_inc,
			       unsigned int times,
			       unsigned int y_ld) {
  unsigned int matrix_x_pos, matrix_y_pos;
  matrix_x_pos = blockIdx.x*blockDim.x + threadIdx.x;
  matrix_y_pos = blockIdx.y*blockDim.y + threadIdx.y;
  if (matrix_x_pos < times && matrix_y_pos < N) {
    unsigned int index_x = matrix_y_pos*x_inc;
    unsigned int index_y = matrix_x_pos*y_ld + matrix_y_pos*y_inc;
    y_mem[index_y] = x_mem[index_x];
  }
}

#endif

/***************************************
 ************* CBLAS SECTION ***********
 ***************************************/

void wrapperCblasCopy(int N, const float *x_mem, unsigned int x_inc,
		      float *y_mem, unsigned int y_inc) {
  cblas_scopy(N, x_mem, x_inc, y_mem, y_inc);
}

void wrapperCblasCopy(int N, const ComplexF *x_mem, unsigned int x_inc,
		      ComplexF *y_mem, unsigned int y_inc) {
  cblas_ccopy(N, x_mem, x_inc, y_mem, y_inc);
}

/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/

template<typename T>
void doCopy(int N, const GPUMirroredMemoryBlock<T>* x,
	    unsigned int x_shift,
	    unsigned int x_inc,
	    GPUMirroredMemoryBlock<T>* y,
	    unsigned int y_shift,
	    unsigned int y_inc,
	    bool use_gpu)
{
  const T *x_mem;
  T *y_mem;
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    //printf("Doing a scopy with comp=1 & cuda=1\n");
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForWrite() + y_shift;
    
    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);
    
    status = wrapperCublasCopy(handle, N, x_mem, x_inc, y_mem, y_inc);
    
    checkCublasError(status);
  }
  else {
    //printf("Doing a scopy with comp=1 & cuda=0\n");
#endif
#ifndef USE_CUDA
    //printf("Doing a scopy with comp=0 & cuda=0\n");
#endif
    x_mem = x->getPPALForRead() + x_shift;
    y_mem = y->getPPALForWrite() + y_shift;

    wrapperCblasCopy(N, x_mem, x_inc, y_mem, y_inc);
#ifdef USE_CUDA
  }
#endif
}

template<typename T>
void doCopyLoop(int N,
		GPUMirroredMemoryBlock<T>* x,
		unsigned int x_inc,
		GPUMirroredMemoryBlock<T>* y,
		unsigned int y_inc,
		unsigned int times,
		const unsigned int stride,
		bool use_gpu)
{
  const T *x_mem;
  T *y_mem;
#ifdef USE_CUDA
  if (use_gpu) {
    //printf("Doing a scopy with comp=1 & cuda=1\n");
    x_mem = x->getGPUForRead();
    y_mem = y->getGPUForWrite();

    const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
    dim3 block, grid;
    // Number of threads on each block dimension
    block.x = min(MAX_THREADS, times);
    block.y = min(MAX_THREADS/block.x, N);
    block.z = 1;

    grid.x = (times/block.x +
	      (times % block.x ? 1 : 0));
    grid.y = (N/block.y + (N % block.y ? 1 : 0));
    grid.z = 1;

    copyLoopKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (N, x_mem, x_inc, y_mem, y_inc, times, stride);
  }
  else {
    //printf("Doing a scopy with comp=1 & cuda=0\n");
#endif
#ifndef USE_CUDA
    //printf("Doing a scopy with comp=0 & cuda=0\n");
#endif
    x_mem = x->getPPALForRead();
    y_mem = y->getPPALForWrite();

    for (unsigned int i = 0; i < times; i++)
      wrapperCblasCopy(N, 
		       x_mem, x_inc,
		       y_mem + i * stride , y_inc);
#ifdef USE_CUDA
  }
#endif
}

template void doCopy<float>(int N, const GPUMirroredMemoryBlock<float>* x,
			    unsigned int x_shift,
			    unsigned int x_inc,
			    GPUMirroredMemoryBlock<float>* y,
			    unsigned int y_shift,
			    unsigned int y_inc,
			    bool use_gpu);
template void doCopy<ComplexF>(int N, const GPUMirroredMemoryBlock<ComplexF>* x,
			       unsigned int x_shift,
			       unsigned int x_inc,
			       GPUMirroredMemoryBlock<ComplexF>* y,
			       unsigned int y_shift,
			       unsigned int y_inc,
			       bool use_gpu);

template void doCopyLoop<float>(int N,
				GPUMirroredMemoryBlock<float>* x,
				unsigned int x_inc,
				GPUMirroredMemoryBlock<float>* y,
				unsigned int y_inc,
				unsigned int times,
				const unsigned int stride,
				bool use_gpu);
template void doCopyLoop<ComplexF>(int N,
				   GPUMirroredMemoryBlock<ComplexF>* x,
				   unsigned int x_inc,
				   GPUMirroredMemoryBlock<ComplexF>* y,
				   unsigned int y_inc,
				   unsigned int times,
				   const unsigned int stride,
				   bool use_gpu);
