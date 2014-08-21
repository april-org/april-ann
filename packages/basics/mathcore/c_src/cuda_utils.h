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

// AUXILIAR INLINE FUNCTIONS //
#ifdef USE_CUDA

#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "gpu_helper.h"

namespace april_math {

  static __device__ void getColumnMajorBunchMatrixPositions(const dim3 &blockIdx,
                                                            const dim3 &blockDim,
                                                            const dim3 &threadIdx,
                                                            unsigned int &matrix_x_pos,
                                                            unsigned int &matrix_y_pos) {
    matrix_x_pos = blockIdx.x*blockDim.x + threadIdx.x;
    matrix_y_pos = (blockIdx.y*blockDim.y + threadIdx.y);
  }

  static void computeBlockAndGridSizesForAColumnMajorBunch(unsigned int bunch_size,
                                                           unsigned int size,
                                                           dim3 &block, dim3 &grid) {
    const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
  
    // Number of threads on each block dimension
    block.x = min(MAX_THREADS, bunch_size);
    block.y = min(MAX_THREADS/block.x, size);
    block.z = 1;
  
    grid.x = (bunch_size/block.x +
              (bunch_size % block.x ? 1 : 0));
    grid.y = (size/block.y + (size % block.y ? 1 : 0));
    grid.z = 1;
    // TODO: FIXME: Check that the grid size does not exceed the limits of the GPU
  }

  static void computeBlockAndGridSizesForARowMajorBunch(unsigned int bunch_size,
                                                        unsigned int size,
                                                        dim3 &block, dim3 &grid) {
    const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
  
    // Number of threads on each block dimension
    block.x = min(MAX_THREADS, size);
    block.y = min(MAX_THREADS/block.x, bunch_size);
    block.z = 1;
  
    grid.x = (size/block.x +
              (size % block.x ? 1 : 0));
    grid.y = (bunch_size/block.y + (bunch_size % block.y ? 1 : 0));
    grid.z = 1;
    // TODO: FIXME: Check that the grid size does not exceed the limits of the GPU
  }

  static void computeBlockAndGridSizesForAnArray(unsigned int bunch_size,
                                                 dim3 &block, dim3 &grid) {
    const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
  
    // Number of threads on each block dimension
    block.x = min(MAX_THREADS, bunch_size);
    block.y = 1;
    block.z = 1;
  
    grid.x = (bunch_size/block.x +
              (bunch_size % block.x ? 1 : 0));
    grid.y = 1;
    grid.z = 1;
    // TODO: FIXME: Check that the grid size does not exceed the limits of the GPU
  }

  static cublasOperation_t getCublasOperation(CBLAS_TRANSPOSE operation) {
    if (operation == CblasNoTrans)
      return CUBLAS_OP_N;
    else if (operation == CblasTrans)
      return CUBLAS_OP_T;
    else // operation == CblasConjTrans
      return CUBLAS_OP_C;
  }

  static cusparseOperation_t getCusparseOperation(CBLAS_TRANSPOSE operation) {
    if (operation == CblasNoTrans)
      return CUSPARSE_OPERATION_NON_TRANSPOSE;
    else if (operation == CblasTrans)
      return CUSPARSE_OPERATION_TRANSPOSE;
    else // operation == CblasConjTrans
      return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  }

} // namespace april_math

#endif

#endif // CUDA_UTILS_H
