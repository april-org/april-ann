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
#include "ceiling_power_of_two.h"
using april_utils::ceilingPowerOfTwo;

#ifdef USE_CUDA
/***************************************
 ************** CUDA SECTION ***********
 ***************************************/

template<typename T>
__global__ void sumVectorFirstReduction(const T *v,
					T *sums,
					unsigned int reduction_top,
					unsigned int size,
					unsigned int stride) {
  unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int active_reduction = reduction_top >> 1;
  if (x_idx < size && x_idx < active_reduction) {
    unsigned int x_pos = x_idx * stride;
    unsigned int passive_index = (x_idx + active_reduction) * stride;
    if (x_idx + active_reduction < size)
      sums[x_pos] = v[x_pos] + v[passive_index];
    else
      sums[x_pos] = v[x_pos];
  }
}

template<typename T>
__global__ void sumVectorNextReduction(T *sums,
				       unsigned int reduction_top,
				       unsigned int size,
				       unsigned int stride) {
  unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int active_reduction = reduction_top >> 1;
  
  if (x_idx < size && x_idx < active_reduction) {
    unsigned int index = x_idx*stride;
    unsigned int passive_index = (x_idx+active_reduction)*stride;
    sums[index] = sums[index] + sums[passive_index];
  }
}

#endif

/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/

template<typename T>
T doSum(unsigned int N,
	const GPUMirroredMemoryBlock<T> *v,
	unsigned int stride,
	unsigned int shift,
	bool use_gpu,
	T zero) {
  T sum(zero);
#ifdef USE_CUDA
  if (use_gpu) {
    GPUMirroredMemoryBlock<T> sums(N);
    const T *v_ptr             = v->getGPUForRead() + shift;
    T *sums_ptr                = sums.getGPUForWrite();
    unsigned int units_top     = ceilingPowerOfTwo(N);
    unsigned int top_reduction = units_top;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    sumVectorFirstReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr,
       sums_ptr,
       top_reduction,
       N, stride);
    for (top_reduction >>= 1; top_reduction != 1; top_reduction >>= 1) {
      computeBlockAndGridSizesForAnArray(top_reduction, block, grid);
      sumVectorNextReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (sums_ptr,
         top_reduction,
	 N, stride);
    }
    sum = sums_ptr[0];
  }
  else {
#endif
    const T *v_mem = v->getPPALForRead() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem+=stride) sum += *v_mem;
#ifdef USE_CUDA
  }
#endif
  return sum;
}

template float doSum<float>(unsigned int N,
			    const GPUMirroredMemoryBlock<float> *v,
			    unsigned int stride,
			    unsigned int shift,
			    bool use_gpu,
			    float zero);

template ComplexF doSum<ComplexF>(unsigned int N,
				  const GPUMirroredMemoryBlock<ComplexF> *v,
				  unsigned int stride,
				  unsigned int shift,
				  bool use_gpu,
				  ComplexF zero);
