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
#include "unused_variable.h"
#include "wrapper.h"
#include "cuda_utils.h"

#ifdef USE_CUDA
/***************************************
 ************** CUDA SECTION ***********
 ***************************************/

cublasStatus_t wrapperCublasAxpy(cublasHandle_t &handle,
				 int N,
				 float *alpha,
				 const float *x_mem,
				 unsigned int x_inc,
				 float *y_mem,
				 unsigned int y_inc) {
  return cublasSaxpy(handle, N, alpha, x_mem, x_inc, y_mem, y_inc);
}

cublasStatus_t wrapperCublasAxpy(cublasHandle_t &handle,
				 int N,
				 ComplexF *alpha,
				 const ComplexF *x_mem,
				 unsigned int x_inc,
				 ComplexF *y_mem,
				 unsigned int y_inc) {
  return cublasCaxpy(handle, N,
                     reinterpret_cast<const cuComplex*>(alpha),
                     reinterpret_cast<const cuComplex*>(x_mem), x_inc,
                     reinterpret_cast<cuComplex*>(y_mem), y_inc);
}

template<typename T>
__global__ void axpyLoopKernel(unsigned int N,
			       T alpha,
			       const T *x_mem,
			       unsigned int x_inc,
			       T *y_mem,
			       unsigned int y_inc,
			       unsigned int times,
			       unsigned int x_ld,
			       unsigned int y_ld) {
  unsigned int matrix_x_pos, matrix_y_pos;
  matrix_x_pos = blockIdx.x*blockDim.x + threadIdx.x;
  matrix_y_pos = blockIdx.y*blockDim.y + threadIdx.y;
  if (matrix_x_pos < times && matrix_y_pos < N) {
    unsigned int index_x = matrix_x_pos*x_ld + matrix_y_pos*x_inc;
    unsigned int index_y = matrix_x_pos*y_ld + matrix_y_pos*y_inc;
    T val = alpha * x_mem[index_x];
    // This loop is used to synchronize the threads for accessing
    // the global memory where they write the results. The loop
    // gets all the values from the threads at the index X in 
    // the current block, synchronizing the access to Y.
    for (unsigned int i=0; i<blockDim.x; ++i) {
      if (i==threadIdx.x) y_mem[index_y] += val;
      __syncthreads();
    }
  }
}

#endif

/***************************************
 ************* CBLAS SECTION ***********
 ***************************************/

void wrapperCblasAxpy(int N, float alpha,
		      const float *x_mem, unsigned int x_inc,
		      float *y_mem, unsigned int y_inc) {
  cblas_saxpy(N, alpha, x_mem, x_inc, y_mem, y_inc);
}

void wrapperCblasAxpy(int N, ComplexF alpha,
		      const ComplexF *x_mem, unsigned int x_inc,
		      ComplexF *y_mem, unsigned int y_inc) {
  cblas_caxpy(N, &alpha, x_mem, x_inc, y_mem, y_inc);
}

/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/
template <typename T>
void doAxpy(int N,
	    T alpha,
	    const GPUMirroredMemoryBlock<T>* x,
	    unsigned int x_shift,
	    unsigned int x_inc,
	    GPUMirroredMemoryBlock<T>* y,
	    unsigned int y_shift,
	    unsigned int y_inc,
	    bool use_gpu)
{
  const T *x_mem;
  T *y_mem;
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    //printf("Doing a saxpy with comp=1 & cuda=1\n");
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForReadAndWrite() + y_shift;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = wrapperCublasAxpy(handle, N, &alpha, x_mem, x_inc, y_mem, y_inc);

    checkCublasError(status);
  }
  else {
    //printf("Doing a saxpy with comp=1 & cuda=0\n");
#endif
#ifndef USE_CUDA
    //printf("Doing a saxpy with comp=0 & cuda=0\n");
#endif
    x_mem = x->getPPALForRead() + x_shift;
    y_mem = y->getPPALForReadAndWrite() + y_shift;
    
    wrapperCblasAxpy(N, alpha, x_mem, x_inc, y_mem, y_inc);
#ifdef USE_CUDA
  }
#endif
}

template <typename T>
void doAxpyLoop(int N,
		T alpha,
		GPUMirroredMemoryBlock<T>* x,
		unsigned int x_inc,
		unsigned int x_shift,
		GPUMirroredMemoryBlock<T>* y,
		unsigned int y_inc,
		unsigned int y_shift,
		unsigned int times,
		const unsigned int x_stride,
		const unsigned int y_stride,
		bool use_gpu)
{
  const T *x_mem;
  T *y_mem;
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForReadAndWrite() + y_shift;

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

    axpyLoopKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (N, alpha, x_mem, x_inc, y_mem, y_inc, times, x_stride, y_stride);
  }
  else {
    //printf("Doing a saxpy loop with comp=1 & cuda=0\n");
#endif
#ifndef USE_CUDA
    //printf("Doing a saxpy loop with comp=0 & cuda=0\n");
#endif
    x_mem = x->getPPALForRead() + x_shift;
    y_mem = y->getPPALForReadAndWrite() + y_shift;

    for (unsigned int i = 0; i < times; i++)
      wrapperCblasAxpy(N, alpha,
		       x_mem + i * x_stride, x_inc, 
		       y_mem + i * y_stride, y_inc);
#ifdef USE_CUDA
  }
#endif
}

template void doAxpy<float>(int N,
			    float alpha,
                    	    const GPUMirroredMemoryBlock<float>* x,
			    unsigned int x_shift,
			    unsigned int x_inc,
			    GPUMirroredMemoryBlock<float>* y,
			    unsigned int y_shift,
			    unsigned int y_inc,
			    bool use_gpu);

template void doAxpy<ComplexF>(int N,
			       ComplexF alpha,
			       const GPUMirroredMemoryBlock<ComplexF>* x,
			       unsigned int x_shift,
			       unsigned int x_inc,
			       GPUMirroredMemoryBlock<ComplexF>* y,
			       unsigned int y_shift,
			       unsigned int y_inc,
			       bool use_gpu);

template void doAxpyLoop<float>(int N,
				float alpha,
				GPUMirroredMemoryBlock<float>* x,
				unsigned int x_inc,
				unsigned int x_shift,
				GPUMirroredMemoryBlock<float>* y,
				unsigned int y_inc,
				unsigned int y_shift,
				unsigned int times,
				const unsigned int x_stride,
				const unsigned int y_stride,
				bool use_gpu);

template void doAxpyLoop<ComplexF>(int N,
				   ComplexF alpha,
				   GPUMirroredMemoryBlock<ComplexF>* x,
				   unsigned int x_inc,
				   unsigned int x_shift,
				   GPUMirroredMemoryBlock<ComplexF>* y,
				   unsigned int y_inc,
				   unsigned int y_shift,
				   unsigned int times,
				   const unsigned int x_stride,
				   const unsigned int y_stride,
				   bool use_gpu);
