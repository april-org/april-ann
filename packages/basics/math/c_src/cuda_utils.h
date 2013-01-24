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

// AUXILIAR INLINE FUNCTIONS //
#ifdef USE_CUDA
static __device__ float avoid_number_in_cuda(float val, float number,
					     float near_zero) {
  if (((number - epsilon) < val) && (val < (number + epsilon))) {
    if (val < number) return number - epsilon ;
    else return number + epsilon;
  }
  return val;
}

static __device__ void getColumnMajorBunchMatrixPositions(const dim3 &blockIdx,
							  const dim3 &blockDim,
							  const dim3 &threadIdx,
							  unsigned int &matrix_x_pos,
							  unsigned int &matrix_y_pos) {
  matrix_x_pos = blockIdx.x*blockDim.x + threadIdx.x;
  matrix_y_pos = (blockIdx.y*blockDim.y + threadIdx.y);
}

static void computeBlockAndGridSizesForAColumnMajorBunch(const ANNConfiguration &conf,
							 unsigned int size,
							 dim3 &block, dim3 &grid) {
  const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
  
  // Number of threads on each block dimension
  block.x = min(MAX_THREADS, conf.cur_bunch_size);
  block.y = min(MAX_THREADS/block.x, size);
  block.z = 1;
  
  grid.x = (conf.cur_bunch_size/block.x +
	    (conf.cur_bunch_size % block.x ? 1 : 0));
  grid.y = (size/block.y + (size % block.y ? 1 : 0));
  grid.z = 1;
  // TODO: FIXME: Check that the grid size does not exceed the limits of the GPU
}

static void computeBlockAndGridSizesForARowMajorBunch(const ANNConfiguration &conf,
						      unsigned int size,
						      dim3 &block, dim3 &grid) {
  const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
  
  // Number of threads on each block dimension
  block.x = min(MAX_THREADS, size);
  block.y = min(MAX_THREADS/block.x, conf.cur_bunch_size);
  block.z = 1;
  
  grid.x = (size/block.x +
	    (size % block.x ? 1 : 0));
  grid.y = (conf.cur_bunch_size/block.y + (conf.cur_bunch_size % block.y ? 1 : 0));
  grid.z = 1;
  // TODO: FIXME: Check that the grid size does not exceed the limits of the GPU
}

static void computeBlockAndGridSizesForAnArray(const ANNConfiguration &conf,
					       dim3 &block, dim3 &grid) {
  const unsigned int MAX_THREADS = GPUHelper::getMaxThreadsPerBlock();
  
  // Number of threads on each block dimension
  block.x = min(MAX_THREADS, conf.cur_bunch_size);
  block.y = 1;
  block.z = 1;
  
  grid.x = (conf.cur_bunch_size/block.x +
	    (conf.cur_bunch_size % block.x ? 1 : 0));
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
#endif
