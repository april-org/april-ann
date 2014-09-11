/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2014, Salvador España-Boquera, Adrian Palacios Corella, Francisco
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
#ifndef MAP_TEMPLATE_H
#define MAP_TEMPLATE_H

#include "cuda_kernel_templates.h"
#include "cuda_utils.h"
#include "gpu_mirrored_memory_block.h"

namespace AprilMath {
    
  template<typename T, typename O, typename F>
  void genericMap1Call(unsigned int N,
                       const GPUMirroredMemoryBlock<T> *input,
                       unsigned int input_stride,
                       unsigned int input_shift,
                       GPUMirroredMemoryBlock<O> *output,
                       unsigned int output_stride,
                       unsigned int output_shift,
                       bool use_gpu,
                       F map_op) {
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      CUDA::genericCudaMap1Call(N,
                                input, input_stride, input_shift,
                                output, output_stride, output_shift,
                                map_op);
    }
    else {
#endif
      const T *input_mem = input->getPPALForRead() + input_shift;
      O *output_mem = output->getPPALForWrite() + output_shift;
      for (unsigned int i=0; i<N; ++i,
             input_mem+=input_stride, output_mem+=output_stride) {
        *output_mem = map_op(*input_mem);
      }
#ifdef USE_CUDA
    }
#endif
  }

  template<typename T1, typename T2, typename O, typename F>
  void genericMap2Call(unsigned int N,
                       const GPUMirroredMemoryBlock<T1> *input1,
                       unsigned int input1_stride,
                       unsigned int input1_shift,
                       const GPUMirroredMemoryBlock<T2> *input2,
                       unsigned int input2_stride,
                       unsigned int input2_shift,
                       GPUMirroredMemoryBlock<O> *output,
                       unsigned int output_stride,
                       unsigned int output_shift,
                       bool use_gpu,
                       F map_op) {
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      CUDA::genericCudaMap2Call(N,
                                input1, input1_stride, input1_shift,
                                input2, input2_stride, input2_shift,
                                output, output_stride, output_shift,
                                map_op);
    }
    else {
#endif
      const T1 *input1_mem = input1->getPPALForRead() + input1_shift;
      const T2 *input2_mem = input2->getPPALForRead() + input2_shift;
      O *output_mem = output->getPPALForWrite() + output_shift;
      for (unsigned int i=0; i<N; ++i,
             output_mem+=output_stride,
             input1_mem+=input1_stride,
             input2_mem+=input2_stride) {
        *output_mem = map_op(*input1_mem, *input2_mem);
      }
#ifdef USE_CUDA
    }
#endif
  }

  template<typename T, typename O, typename OP>
  struct ScalarToSpanMap1 {
    const OP functor;
    ScalarToSpanMap1(const OP &functor) : functor(functor) { }
    void operator()(unsigned int N,
                    const GPUMirroredMemoryBlock<T> *input,
                    unsigned int input_stride,
                    unsigned int input_shift,
                    GPUMirroredMemoryBlock<O> *output,
                    unsigned int output_stride,
                    unsigned int output_shift,
                    bool use_cuda) const {
      genericMap1Call(N, input, input_stride, input_shift,
                      output, output_stride, output_shift,
                      use_cuda, functor);
    }
  };

  template<typename T1, typename T2, typename O, typename OP>
  struct ScalarToSpanMap2 {
    const OP functor;
    ScalarToSpanMap2(const OP &functor) : functor(functor) { }
    void operator()(unsigned int N,
                    const GPUMirroredMemoryBlock<T1> *input1,
                    unsigned int input1_stride,
                    unsigned int input1_shift,
                    const GPUMirroredMemoryBlock<T2> *input2,
                    unsigned int input2_stride,
                    unsigned int input2_shift,
                    GPUMirroredMemoryBlock<O> *output,
                    unsigned int output_stride,
                    unsigned int output_shift,
                    bool use_cuda) const {
      genericMap2Call(N, input1, input1_stride, input1_shift,
                      input2, input2_stride, input2_shift,
                      output, output_stride, output_shift,
                      use_cuda, functor);
    }
  };
  
} // namespace AprilMath

#endif // MAP_TEMPLATE_H
