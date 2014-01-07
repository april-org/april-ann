
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

/***************************************
 *********** TEMPLATE SECTION **********
 ***************************************/
template <typename T>
void doCmul(int N,
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
    ERROR_PRINT("CUDA VERSION NOT IMPLEMENTED\n");
  }
  // else {
#endif
  x_mem = x->getPPALForRead() + x_shift;
  y_mem = y->getPPALForReadAndWrite() + y_shift;
  for (int i=0; i<N; ++i, x_mem+=x_inc, y_mem+=y_inc)
    (*y_mem) = (*y_mem) * (*x_mem);
#ifdef USE_CUDA
  // }
#endif
}

template void doCmul<float>(int N,
                    	    const GPUMirroredMemoryBlock<float>* x,
			    unsigned int x_shift,
			    unsigned int x_inc,
			    GPUMirroredMemoryBlock<float>* y,
			    unsigned int y_shift,
			    unsigned int y_inc,
			    bool use_gpu);

template void doCmul<ComplexF>(int N,
			       const GPUMirroredMemoryBlock<ComplexF>* x,
			       unsigned int x_shift,
			       unsigned int x_inc,
			       GPUMirroredMemoryBlock<ComplexF>* y,
			       unsigned int y_shift,
			       unsigned int y_inc,
			       bool use_gpu);
