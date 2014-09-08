/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "error_print.h"

// AUXILIAR INLINE FUNCTIONS //
#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include "cblas_headers.h"
#include "ceiling_power_of_two.h"
#include "complex_number.h"
#include "maxmin.h"
#include "gpu_helper.h"

#define MAX_CUDA_REDUCE_THREAD_SIZE 1024
#define MIN_CUDA_REDUCE_THREAD_SIZE    8
#define MAX_CUDA_REDUCE_NUM_THREADS 1024 // must be multiple of 2 (wrap size=32)

#define APRIL_CUDA_EXPORT __host__ __device__
#define APRIL_CUDA_ERROR_EXIT(code,msg) aprilCudaErrorExit((code),(msg))

// FIXME: implement properly this feature to control errors in CUDA kernels and
// CPU equivalent code.
static __host__ __device__ void aprilCudaErrorExit(int code, const char *msg) {
  UNUSED_VARIABLE(code);
  UNUSED_VARIABLE(msg);
  // do nothing
}

// static void aprilCudaErrorExit(int code, const char *msg) {
//   ERROR_EXIT(code,msg);
// }

namespace AprilMath {
  namespace CUDA {

    //
    // From Reduction SDK sample:
    //
    // Utility class used to avoid linker errors with extern
    // unsized shared memory arrays with templated type
    //
    template<class T>
    struct SharedMemory
    {
      __device__ inline operator       T*()
      {
        extern __shared__ int __smem[];
        return (T*) (void *) __smem;
      }
      
      __device__ inline operator const T*() const
      {
        extern __shared__ int __smem[];
        return (T*) (void *) __smem;
      }
    };
    //
    // from Reduction SDK sample:
    // specialize to avoid unaligned memory 
    // access compile errors
    //
#define SPECIALIZE_CUDA_SHARED_MEMORY(FULLTYPE,TYPE)            \
    namespace AprilMath {                                       \
      namespace CUDA {                                          \
        template<>                                              \
        struct SharedMemory<FULLTYPE>                           \
        {                                                       \
          __device__ inline operator FULLTYPE*()                \
          {                                                     \
            extern __shared__ FULLTYPE                          \
              __smem_##TYPE[];                                  \
              return (FULLTYPE*)__smem_##TYPE;                  \
          }                                                     \
          __device__ inline operator const FULLTYPE*() const    \
          {                                                     \
            extern __shared__ FULLTYPE                          \
              __smem_##TYPE[];                                  \
            return (FULLTYPE*)__smem_##TYPE;                    \
          }                                                     \
        };                                                      \
      }                                                         \
    }
 
    static __device__ unsigned int getArrayIndex(const dim3 &blockIdx,
                                                 const dim3 &blockDim,
                                                 const dim3 &threadIdx) {
      return blockIdx.x * blockDim.x + threadIdx.x;
    }
    
    static __device__ void getMatrixIndices(const dim3 &blockIdx,
                                            const dim3 &blockDim,
                                            const dim3 &threadIdx,
                                            unsigned int &matrix_row_pos,
                                            unsigned int &matrix_col_pos) {
      matrix_row_pos = blockIdx.x*blockDim.x + threadIdx.x;
      matrix_col_pos = blockIdx.y*blockDim.y + threadIdx.y;
    }

    static void computeReductionSize(int N, int &numThreads,
                                     int &threadSize, int &numBlocks) {
      int N2 = AprilUtils::ceilingPowerOfTwo(N);
      numThreads = AprilUtils::max(1, AprilUtils::min(N2/MIN_CUDA_REDUCE_THREAD_SIZE,
                                                      AprilUtils::min(MAX_CUDA_REDUCE_NUM_THREADS,
                                                                      static_cast<int>(GPUHelper::getMaxThreadsPerBlock()))));
      threadSize = AprilUtils::max(1, AprilUtils::min(MAX_CUDA_REDUCE_THREAD_SIZE,
                                                      N/numThreads));
      numBlocks  = AprilUtils::max(1, N/(numThreads*threadSize));
    }
    
    static void computeBlockAndGridSizesFor2DMatrix(unsigned int N,
                                                    unsigned int M,
                                                    dim3 &block, dim3 &grid) {
      const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
  
      // Number of threads on each block dimension
      block.x = AprilUtils::min(MAX_THREADS, N);
      block.y = AprilUtils::min(MAX_THREADS/block.x, M);
      block.z = 1;
  
      grid.x = (N/block.x + (N % block.x ? 1 : 0));
      grid.y = (M/block.y + (M % block.y ? 1 : 0));
      grid.z = 1;
      // TODO: FIXME: Check that the grid size does not exceed the limits of the GPU
    }

    static void computeBlockAndGridSizesForArray(unsigned int bunch_size,
                                                 dim3 &block, dim3 &grid) {
      const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
  
      // Number of threads on each block dimension
      block.x = AprilUtils::min(MAX_THREADS, bunch_size);
      block.y = 1;
      block.z = 1;
  
      grid.x = (bunch_size/block.x +
                (bunch_size % block.x ? 1 : 0));
      grid.y = 1;
      grid.z = 1;
      // TODO: FIXME: Check that the grid size does not exceed the limits of the GPU
    }

    static cublasOperation_t getCublasOperation(CBLAS_TRANSPOSE operation) {
      if (operation == CblasNoTrans) {
        return CUBLAS_OP_N;
      }
      else if (operation == CblasTrans) {
        return CUBLAS_OP_T;
      }
      else { // operation == CblasConjTrans
        return CUBLAS_OP_C;
      }
    }

    static cusparseOperation_t getCusparseOperation(CBLAS_TRANSPOSE operation) {
      if (operation == CblasNoTrans) {
        return CUSPARSE_OPERATION_NON_TRANSPOSE;
      }
      else if (operation == CblasTrans) {
        return CUSPARSE_OPERATION_TRANSPOSE;
      }
      else { // operation == CblasConjTrans
        return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
      }
    }
    
  } // namespace CUDA
} // namespace AprilMath

#else

#define APRIL_CUDA_EXPORT
#define APRIL_CUDA_ERROR_EXIT(code,msg) ERROR_EXIT(code,msg)
#define SPECIALIZE_CUDA_SHARED_MEMORY(FULLTYPE,TYPE)

#endif

SPECIALIZE_CUDA_SHARED_MEMORY(int32_t,int32_t);
SPECIALIZE_CUDA_SHARED_MEMORY(float,float);
SPECIALIZE_CUDA_SHARED_MEMORY(double,double);
SPECIALIZE_CUDA_SHARED_MEMORY(AprilMath::ComplexF,ComplexF);

#endif // CUDA_UTILS_H
