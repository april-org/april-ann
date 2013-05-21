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
#include <cmath>
#include "wrapper.h"

#ifdef USE_CUDA
cublasOperation_t getCublasOperation(CBLAS_TRANSPOSE operation) {
  if (operation == CblasNoTrans)
    return CUBLAS_OP_N;
  else if (operation == CblasTrans)
    return CUBLAS_OP_T;
  else // operation == CblasConjTrans
    return CUBLAS_OP_C;
}
#endif

///////////////////////////////////////////////////////////
/////////////////// Kernels ///////////////////////////////
///////////////////////////////////////////////////////////

#ifdef USE_CUDA
__global__ void scopyLoopKernel(unsigned int N,
				const float *x_mem,
				unsigned int x_inc,
				float *y_mem,
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

__global__ void saxpyLoopKernel(unsigned int N,
				float alpha,
				const float *x_mem,
				unsigned int x_inc,
				float *y_mem,
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
    float val = alpha * x_mem[index_x];
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

///////////////////////////////////////////////////////////
//////////////////// BLAS wrappers ////////////////////////
///////////////////////////////////////////////////////////

void doSgemv(CBLAS_ORDER major_type, CBLAS_TRANSPOSE a_transpose,
	     int m, int n,
	     float alpha, FloatGPUMirroredMemoryBlock *a, unsigned int a_inc,
	     FloatGPUMirroredMemoryBlock *x, unsigned int x_inc,
	     float beta, FloatGPUMirroredMemoryBlock *y, unsigned int y_inc,
	     unsigned int a_shift, unsigned int x_shift, unsigned int y_shift,
	     bool use_gpu) {
  const float *a_mem, *x_mem;
  float *y_mem;
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

    status = cublasSgemv(handle, cublas_a_transpose,
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
    cblas_sgemv(major_type, a_transpose,
                m, n,
                alpha, a_mem, a_inc,
                x_mem, x_inc,
                beta, y_mem, y_inc);
#ifdef USE_CUDA
  }
#endif
}


void doScopy(int N, FloatGPUMirroredMemoryBlock* x,
	     unsigned int x_shift,
	     unsigned int x_inc,
	     FloatGPUMirroredMemoryBlock* y,
	     unsigned int y_shift,
	     unsigned int y_inc,
	     bool use_gpu)
{
  const float *x_mem;
  float *y_mem;
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    //printf("Doing a scopy with comp=1 & cuda=1\n");
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForWrite() + y_shift;
    
    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);
    
    status = cublasScopy(handle, N, x_mem, x_inc, y_mem, y_inc);
    
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

    cblas_scopy(N, x_mem, x_inc, y_mem, y_inc);
#ifdef USE_CUDA
  }
#endif
}

void doScopyLoop(int N,
		 FloatGPUMirroredMemoryBlock* x,
		 unsigned int x_inc,
		 FloatGPUMirroredMemoryBlock* y,
		 unsigned int y_inc,
		 unsigned int times,
		 const unsigned int stride,
		 bool use_gpu)
{
  const float *x_mem;
  float *y_mem;
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

    scopyLoopKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
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
      cblas_scopy(N, 
		  x_mem, x_inc,
		  y_mem + i * stride , y_inc);
#ifdef USE_CUDA
  }
#endif
}

void doSaxpy(int N,
	     float alpha,
	     FloatGPUMirroredMemoryBlock* x,
	     unsigned int x_shift,
	     unsigned int x_inc,
	     FloatGPUMirroredMemoryBlock* y,
	     unsigned int y_shift,
	     unsigned int y_inc,
	     bool use_gpu)
{
  const float *x_mem;
  float *y_mem;
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    //printf("Doing a saxpy with comp=1 & cuda=1\n");
    x_mem = x->getGPUForRead() + x_shift;
    y_mem = y->getGPUForReadAndWrite() + y_shift;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = cublasSaxpy(handle, N, &alpha, x_mem, x_inc, y_mem, y_inc);

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

    cblas_saxpy(N, alpha, x_mem, x_inc, y_mem, y_inc);
#ifdef USE_CUDA
  }
#endif
}

void doSaxpyLoop(int N,
		 float alpha,
		 FloatGPUMirroredMemoryBlock* x,
		 unsigned int x_inc,
		 FloatGPUMirroredMemoryBlock* y,
		 unsigned int y_inc,
		 unsigned int times,
		 const unsigned int x_stride,
		 const unsigned int y_stride,
		 bool use_gpu)
{
  const float *x_mem;
  float *y_mem;
#ifdef USE_CUDA
  if (use_gpu) {
    x_mem = x->getGPUForRead();
    y_mem = y->getGPUForReadAndWrite();

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

    saxpyLoopKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (N, alpha, x_mem, x_inc, y_mem, y_inc, times, x_stride, y_stride);
  }
  else {
    //printf("Doing a saxpy loop with comp=1 & cuda=0\n");
#endif
#ifndef USE_CUDA
    //printf("Doing a saxpy loop with comp=0 & cuda=0\n");
#endif
    x_mem = x->getPPALForRead();
    y_mem = y->getPPALForReadAndWrite();

    for (unsigned int i = 0; i < times; i++)
      cblas_saxpy(N, alpha,
		  x_mem + i * x_stride, x_inc, 
		  y_mem + i * y_stride, y_inc);
#ifdef USE_CUDA
  }
#endif
}

void doSgemm(CBLAS_ORDER major_type,
	     CBLAS_TRANSPOSE a_transpose,
	     CBLAS_TRANSPOSE b_transpose,
	     int m,
	     int n,
	     int k,
	     float alpha,
	     FloatGPUMirroredMemoryBlock* a,
	     unsigned int a_inc,
	     FloatGPUMirroredMemoryBlock* b,
	     unsigned int b_inc,
	     float beta,
	     FloatGPUMirroredMemoryBlock* c,
	     unsigned int c_inc,
	     unsigned int a_shift,
	     unsigned int b_shift,
	     unsigned int c_shift,
	     bool use_gpu)
{
  const float *a_mem, *b_mem;
  float *c_mem;
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

    status = cublasSgemm(handle, cublas_a_transpose, cublas_b_transpose,
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
    cblas_sgemm(major_type,   // Row or Col Major
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

void doVectorSetToZero(FloatGPUMirroredMemoryBlock *v,
		       unsigned int v_size,
		       unsigned int inc,
		       unsigned int shift,
		       bool use_gpu) {
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    float *ptr   = v->getGPUForWrite() + shift;
    float  value = 0.0f;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = cublasSscal(handle, v_size, &value, ptr, inc);
    // FIXME: To use cuMemsetD32 instead of cublasSscal
    checkCublasError(status);
  }
  else {
#endif
    float *ptr = v->getPPALForWrite() + shift;
    VECTOR_SSET(v_size, 0.0f, ptr, inc);
#ifdef USE_CUDA
  }
#endif
}

void doVectorSet(FloatGPUMirroredMemoryBlock *v,
		 float value,
		 unsigned int v_size,
		 unsigned int inc,
		 unsigned int shift,
		 bool use_gpu) {
  if (use_gpu) ERROR_EXIT(128, "CUDA version not implemented yet\n");
  float *ptr = v->getPPALForWrite() + shift;
  VECTOR_SSET(v_size, value, ptr, inc);
}

void doSscal(unsigned int size,
	     float alpha,
	     FloatGPUMirroredMemoryBlock *x,
	     unsigned int shift,
	     unsigned int inc,
	     bool use_gpu) {
  float *x_mem;
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    x_mem = x->getGPUForReadAndWrite() + shift;

    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);

    status = cublasSscal(handle, size, &alpha, x_mem, inc);

    checkCublasError(status);
  }
  else {
#endif
    x_mem = x->getPPALForReadAndWrite() + shift;
    cblas_sscal(size, alpha, x_mem, inc);
#ifdef USE_CUDA
  }
#endif
}

void doSger(CBLAS_ORDER major_type,
	    unsigned int m,
	    unsigned int n,
	    float alpha,
	    FloatGPUMirroredMemoryBlock *x,
	    unsigned int x_shift,
	    unsigned int x_inc,
	    FloatGPUMirroredMemoryBlock *y,
	    unsigned int y_shift,
	    unsigned int y_inc,
	    FloatGPUMirroredMemoryBlock *a,
	    unsigned int a_shift,
	    unsigned int a_inc,
	    bool use_gpu) {
  const float *x_mem;
  const float *y_mem;
  float *a_mem;
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

    status = cublasSger(handle,
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

    cblas_sger(major_type,
	       m, n,
	       alpha,
	       x_mem, x_inc,
	       y_mem, y_inc,
	       a_mem, a_inc);
#ifdef USE_CUDA
  }
#endif
}

float doSnrm2(unsigned int n,
	      FloatGPUMirroredMemoryBlock *x,
	      unsigned int shift,
	      unsigned int inc,
	      bool use_gpu) {
  float result;
  const float *x_mem;
#ifdef USE_CUDA
  if (use_gpu) {
    cublasStatus_t status;
    cublasHandle_t handle = GPUHelper::getHandler();
    x_mem  = x->getGPUForRead() + shift;
    status = cublasSetStream(handle, GPUHelper::getCurrentStream());
    checkCublasError(status);
    status = cublasSnrm(handle, n, x_mem, inc, &result);
    checkCublasError(status);
  }
  else {
#endif
    x_mem = x->getPPALForRead() + shift;
    result = cblas_snrm2(n, x_mem, inc);
#ifdef USE_CUDA
  }
#endif
  return result;
}
