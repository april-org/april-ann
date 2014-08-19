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
#include "unused_variable.h"

namespace april_math {

  /***************************************
   ************* CBLAS SECTION ***********
   ***************************************/

  float absolute_value(const float &v) { return fabsf(v); }
  float absolute_value(const ComplexF &v) { return v.abs(); }

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
      ERROR_PRINT("CUDA VERSION NOT IMPLEMENTED\n");
    }
    // else {
#endif
    const T *v1_mem = v1->getPPALForRead() + shift1;
    const T *v2_mem = v2->getPPALForRead() + shift2;
    for (unsigned int i=0; i<N && eq; ++i, v1_mem+=stride1, v2_mem+=stride2) {
      T aux(*v1_mem - *v2_mem);
      eq = eq && (absolute_value(aux) < epsilon);
    }
#ifdef USE_CUDA
    //  }
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

} // namespace april_math
