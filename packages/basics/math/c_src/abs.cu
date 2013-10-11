/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#include "unused_variable.h"
using april_utils::ceilingPowerOfTwo;

#ifdef USE_CUDA
/***************************************
 ************** CUDA SECTION ***********
 ***************************************/

template<typename T>
__global__ void absKernel(T *v, unsigned int N, unsigned int stride) {
  unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (x_idx < N) {
    unsigned int x_pos = x_idx*stride;
    v[x_pos] = fabs(v[x_pos]);
  }
}
#endif

/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/

template<typename T>
void doAbs(unsigned int N,
	   GPUMirroredMemoryBlock<T> *v,
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
    absKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
      (v_ptr, N, stride);
  }
  else {
#endif
    T *v_mem = v->getPPALForReadAndWrite() + shift;
    for (unsigned int i=0; i<N; ++i, v_mem+=stride)
      *v_mem = fabs(*v_mem);
#ifdef USE_CUDA
  }
#endif
}

template void doAbs<float>(unsigned int N,
			   GPUMirroredMemoryBlock<float> *v,
			   unsigned int stride,
			   unsigned int shift,
			   bool use_gpu);
