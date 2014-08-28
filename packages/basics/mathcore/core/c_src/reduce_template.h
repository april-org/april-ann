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
#ifndef REDUCE_TEMPLATE_H
#define REDUCE_TEMPLATE_H

#include "cuda_kernel_templates.h"
#include "cuda_utils.h"
#include "gpu_mirrored_memory_block.h"

namespace april_math {
  
  /**
   * @brief Performs a reduce over a vector and stores its result at
   * another vector.
   *
   * @tparam T - The type for input and output vectors.
   *
   * @tparam F - An operator implemented as a functor.
   *
   * @param input - The input vector.
   *
   * @param input_stride - The stride between consecutive values at input.
   *
   * @param input_shift - The first valid position at input vector.
   *
   * @param zero - The value of reduce over zero elements.
   *
   * @param reduce_op - The functor operator with the reduce over two
   *
   * @param dest - The output vector.
   *
   * @param dest_shift - The first valid position at dest vector.
   *
   * @note Only the position located at dest_shift will be written in dest
   * vector.
   *
   * @note The reduce_op functor must be associative, commutative and
   * idempotent.
   *
   * @note For CUDA compilation, it is needed that reduce_op has been exported
   * using APRIL_CUDA_EXPORT macro.
   */
  template<typename T, typename O, typename F>
  O genericReduce1Call(unsigned int N,
                       const GPUMirroredMemoryBlock<T> *input,
                       unsigned int input_stride,
                       unsigned int input_shift,
                       bool use_gpu,
                       const O &zero,
                       F reduce_op,
                       GPUMirroredMemoryBlock<O> *dest=0,
                       unsigned int dest_shift=0) {
    O result(zero);
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      april_utils::SharedPtr< GPUMirroredMemoryBlock<O> > cuda_dest(dest);
      if (cuda_dest.empty()) cuda_dest = new GPUMirroredMemoryBlock<O>(1);
      CUDA::genericCudaReduceCall(N, input, input_stride, input_shift, zero,
                                  cuda_dest.get(), dest_shift,
                                  reduce_op);
      cuda_dest->getValue(dest_shift, result);
    }
    else {
#endif
      const T *v_mem = input->getPPALForRead() + input_shift;
      for (unsigned int i=0; i<N; ++i, v_mem+=input_stride) {
        result = reduce_op(result, *v_mem);
      }
      if (dest != 0) {
        O *dest_ptr = dest->getPPALForWrite() + dest_shift;
        *dest_ptr = result;
      }
#ifdef USE_CUDA
    }
#endif
    return result;
  }

  template<typename T, typename O, typename OP>
  struct ScalarToSpanReduce1 {
    const OP functor;
    ScalarToSpanReduce1(const OP &functor) : functor(functor) { }
    O operator()(unsigned int N,
                 const GPUMirroredMemoryBlock<T> *input,
                 unsigned int input_stride,
                 unsigned int input_shift,
                 bool use_cuda,
                 const O &zero,
                 GPUMirroredMemoryBlock<O> *dest=0,
                 unsigned int dest_raw_pos=0) {
      return genericReduce1Call(N, input, input_stride, input_shift,
                                use_cuda, zero, functor,
                                dest, dest_raw_pos);
    }
  };
  
} // namespace april_math

#endif // REDUCE_TEMPLATE_H
