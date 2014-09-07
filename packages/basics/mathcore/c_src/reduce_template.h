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

#include "cmath_overloads.h"
#include "cuda_kernel_templates.h"
#include "cuda_utils.h"
#include "gpu_mirrored_memory_block.h"

namespace AprilMath {

  /**
   * @brief Performs a reduce over a vector and stores its result at
   * another vector.
   *
   * @tparam T - The type for input vectors.
   *
   * @tparam O - The type for output vector (reduction type).
   *
   * @tparam F - An operator implemented as a functor.
   *
   * @tparam P - An operator implemented as a functor.
   *
   * @param input - The input vector.
   *
   * @param input_stride - The stride between consecutive values at input.
   *
   * @param input_shift - The first valid position at input vector.
   *
   * @param zero - The value of reduce over zero elements.
   *
   * @param reduce_op - The functor operator with the reduce over two inputs.
   *
   * @param partials_reduce_op - The functor operator with the reduce over two
   * partial results.
   *
   * @param dest - The output vector.
   *
   * @param dest_shift - The first valid position at dest vector.
   *
   * @note Only the position located at dest_shift will be written in dest
   * vector.
   *
   * @note For CUDA compilation, it is needed that all functros have been
   * exported using APRIL_CUDA_EXPORT macro and to be inline operations which
   * can call functions defined for host and device.
   *
   * @note The zero parameter is needed by CUDA implementation to initialize
   * partials.
   */
  template<typename T, typename O, typename F, typename P>
  void genericReduceCall(unsigned int N,
                         const GPUMirroredMemoryBlock<T> *input,
                         unsigned int input_stride,
                         unsigned int input_shift,
                         bool use_gpu,
                         const O &zero,
                         F reduce_op,
                         P partials_reduce_op,
                         GPUMirroredMemoryBlock<O> *dest,
                         unsigned int dest_shift) {
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
    UNUSED_VARIABLE(partials_reduce_op);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      CUDA::genericCudaReduceCall(N, input, input_stride, input_shift,
                                  dest, dest_shift, zero,
                                  reduce_op, partials_reduce_op);
    }
    else {
#endif
      O result(zero);
      const T *v_mem = input->getPPALForRead() + input_shift;
      for (unsigned int i=0; i<N; ++i, v_mem+=input_stride) {
        reduce_op(result, *v_mem);
      }
      O *dest_ptr = dest->getPPALForReadAndWrite() + dest_shift;
      partials_reduce_op(*dest_ptr, result);
#ifdef USE_CUDA
    }
#endif
  }
  
  template<typename T, typename F>
  void genericReduceMinMaxCall(unsigned int N,
                               const GPUMirroredMemoryBlock<T> *input,
                               unsigned int input_stride,
                               unsigned int input_shift,
                               bool use_gpu,
                               const T &zero,
                               F reduce_op,
                               GPUMirroredMemoryBlock<int32_t> *which,
                               unsigned int which_shift,
                               GPUMirroredMemoryBlock<T> *dest,
                               unsigned int dest_shift) {
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
    UNUSED_VARIABLE(zero);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      CUDA::genericCudaReduceMinMaxCall(N, input, input_stride, input_shift, zero,
                                        which, which_shift,
                                        dest, dest_shift,
                                        reduce_op);
    }
    else {
#endif
      T *dest_ptr = dest->getPPALForReadAndWrite() + dest_shift;
      int32_t *which_ptr = which->getPPALForReadAndWrite() + which_shift;
      const T *v_mem = input->getPPALForRead() + input_shift;
      for (unsigned int i=0; i<N; ++i, v_mem+=input_stride) {
        reduce_op(*dest_ptr, *v_mem, *which_ptr, static_cast<int>(i));
      }
#ifdef USE_CUDA
    }
#endif
  }
  
  template<typename T, typename O, typename OP1>
  struct ScalarToSpanReduce {
    const OP1 functor;
    ScalarToSpanReduce(const OP1 &functor) : functor(functor) { }
    template<typename OP2>
    void operator()(unsigned int N,
                    const GPUMirroredMemoryBlock<T> *input,
                    unsigned int input_stride,
                    unsigned int input_shift,
                    bool use_cuda,
                    const O &zero,
                    const OP2 &functor2,
                    GPUMirroredMemoryBlock<O> *dest,
                    unsigned int dest_raw_pos) const {
      genericReduceCall(N, input, input_stride, input_shift,
                        use_cuda, zero, functor, functor2,
                        dest, dest_raw_pos);
    }
  };

  template<typename T, typename OP>
  struct ScalarToSpanReduceMinMax {
    const OP functor;
    ScalarToSpanReduceMinMax(const OP &functor) : functor(functor) { }
    template<typename OP2>
    void operator()(unsigned int N,
                    const GPUMirroredMemoryBlock<T> *input,
                    unsigned int input_stride,
                    unsigned int input_shift,
                    bool use_cuda,
                    const T &zero,
                    const OP2 &functor2, // ignored here
                    GPUMirroredMemoryBlock<int32_t> *which,
                    unsigned int which_raw_pos,
                    GPUMirroredMemoryBlock<T> *dest,
                    unsigned int dest_raw_pos) const {
      UNUSED_VARIABLE(functor2);
      genericReduceMinMaxCall(N, input, input_stride, input_shift,
                              use_cuda, zero, functor,
                              which, which_raw_pos,
                              dest, dest_raw_pos);
    }
  };
  
} // namespace AprilMath

#endif // REDUCE_TEMPLATE_H
