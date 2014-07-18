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
#include "ceiling_power_of_two.h"
#include "cuda_utils.h"
#include "unused_variable.h"
#include "wrapper.h"
using april_utils::ceilingPowerOfTwo;

#ifndef USE_CUDA
#define __host__
#define __device__
#endif

__host__ __device__ float absolute_value(const float &v) { return fabsf(v); }
__host__ __device__ float absolute_value(const ComplexF &v) { return v.abs(); }

#ifndef USE_CUDA
#undef __host__
#undef __device__
#endif

#ifdef USE_CUDA
/***************************************
 ************** CUDA SECTION ***********
 ***************************************/

template<typename T>
__global__ void equalsVectorFirstReduction(const T *v1_mem, const T*v2_mem,
                                           bool *equals_mem,
                                           unsigned int reduction_top,
                                           unsigned int size,
                                           unsigned int stride1,
                                           unsigned int stride2,
                                           float epsilon) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int active_reduction = reduction_top >> 1;
  if (idx < size && idx < active_reduction) {
    // index and passive_index assumed as stride=1
    unsigned int index = idx;
    unsigned int passive_index = (idx + active_reduction);
    if (idx + active_reduction < size) {
      unsigned int index1 = idx * stride1;
      unsigned int index2 = idx * stride2;
      unsigned int passive_index1 = passive_index * stride1;
      unsigned int passive_index2 = passive_index * stride2;
      T aux_index(v1_mem[index1] - v2_mem[index2]);
      T aux_passive_index(v1_mem[passive_index1] - v2_mem[passive_index2]);
      equals_mem[index] = ( (absolute_value(aux_index) < epsilon) &&
                            (absolute_value(aux_passive_index) < epsilon) );
    }
    else {
      equals_mem[index] = true;
    }
  }
}

__global__ void equalsVectorNextReduction(bool *equals_mem,
                                          unsigned int reduction_top,
                                          unsigned int size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int active_reduction = reduction_top >> 1;
  
  if (idx < size && idx < active_reduction) {
    // index and passive_index assumed as stride=1
    unsigned int index = idx;
    unsigned int passive_index = (idx+active_reduction);
    equals_mem[index] = ( equals_mem[index] && equals_mem[passive_index] );
  }
}
#endif

/***************************************
 ************* CBLAS SECTION ***********
 ***************************************/

template<typename T>
bool doEquals(unsigned int N,
	      const GPUMirroredMemoryBlock<T> *v1,
	      const GPUMirroredMemoryBlock<T> *v2,
	      unsigned int stride1,
	      unsigned int stride2,
	      unsigned int shift1,
	      unsigned int shift2,
	      float epsilon,
	      bool use_gpu) {
  bool eq = true;
#ifndef USE_CUDA
  UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
  if (use_gpu) {
    GPUMirroredMemoryBlock<bool> equals(N);
    bool *equals_ptr   = equals.getGPUForWrite();
    const T *v1_mem = v1->getGPUForRead() + shift1;
    const T *v2_mem = v2->getGPUForRead() + shift2;
    unsigned int units_top     = ceilingPowerOfTwo(N);
    unsigned int top_reduction = units_top;
    dim3 block, grid;
    computeBlockAndGridSizesForAnArray(N, block, grid);
    equalsVectorFirstReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v1_mem, v2_mem, equals_ptr, top_reduction,
       N, stride1, stride2, epsilon);
    for (top_reduction >>= 1; top_reduction != 1; top_reduction >>= 1) {
      computeBlockAndGridSizesForAnArray(top_reduction, block, grid);
      equalsVectorNextReduction<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (equals_ptr, top_reduction, N);
    }
    // TODO: Improve the efficiency of this assignment, now it needs
    // to copy the whole sums array even when only the first one
    // component is needed
    bool result = equals.getPPALForRead()[0];
    return result;
  }
  else {
#endif
    const T *v1_mem = v1->getPPALForRead() + shift1;
    const T *v2_mem = v2->getPPALForRead() + shift2;
    for (unsigned int i=0; i<N && eq; ++i, v1_mem+=stride1, v2_mem+=stride2) {
      T aux(*v1_mem - *v2_mem);
      eq = eq && (absolute_value(aux) < epsilon);
    }
#ifdef USE_CUDA
  }
#endif
  return eq;
}

template bool doEquals<float>(unsigned int N,
			      const GPUMirroredMemoryBlock<float> *v1,
			      const GPUMirroredMemoryBlock<float> *v2,
			      unsigned int stride1,
			      unsigned int stride2,
			      unsigned int shift1,
			      unsigned int shift2,
			      float epsilon,
			      bool use_gpu);

template bool doEquals<ComplexF>(unsigned int N,
                                 const GPUMirroredMemoryBlock<ComplexF> *v1,
                                 const GPUMirroredMemoryBlock<ComplexF> *v2,
                                 unsigned int stride1,
                                 unsigned int stride2,
                                 unsigned int shift1,
                                 unsigned int shift2,
                                 float epsilon,
                                 bool use_gpu);
