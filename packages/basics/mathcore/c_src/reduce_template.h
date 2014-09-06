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
  template<typename T, typename F>
  T genericReduceCall(unsigned int N,
                      const GPUMirroredMemoryBlock<T> *input,
                      unsigned int input_stride,
                      unsigned int input_shift,
                      bool use_gpu,
                      const T &zero,
                      F reduce_op,
                      GPUMirroredMemoryBlock<T> *dest=0,
                      unsigned int dest_shift=0) {
    T result(zero);
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      AprilUtils::SharedPtr< GPUMirroredMemoryBlock<T> > cuda_dest(dest);
      if (cuda_dest.empty()) cuda_dest = new GPUMirroredMemoryBlock<T>(1);
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
        T *dest_ptr = dest->getPPALForWrite() + dest_shift;
        *dest_ptr = result;
      }
#ifdef USE_CUDA
    }
#endif
    return result;
  }

  /**
   * @brief Performs a sum reduction over a vector and stores its result at
   * another vector.
   *
   * It has been specialized because works betwen 2 and 3 times faster using
   * operator += than using the AprilMath::r_add function template.
   *
   * @tparam T - The type for input and output vectors.
   *
   * @param input - The input vector.
   *
   * @param input_stride - The stride between consecutive values at input.
   *
   * @param input_shift - The first valid position at input vector.
   *
   * @param dest - The output vector.
   *
   * @param dest_shift - The first valid position at dest vector.
   *
   * @note Only the position located at dest_shift will be written in dest
   * vector.
   */
  template<typename T>
  T sumReduceCall(unsigned int N,
                  const GPUMirroredMemoryBlock<T> *input,
                  unsigned int input_stride,
                  unsigned int input_shift,
                  bool use_gpu,
                  GPUMirroredMemoryBlock<T> *dest=0,
                  unsigned int dest_shift=0) {
    const T zero(0.0f);
    T result(zero);
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      AprilUtils::SharedPtr< GPUMirroredMemoryBlock<T> > cuda_dest(dest);
      if (cuda_dest.empty()) cuda_dest = new GPUMirroredMemoryBlock<T>(1);
      CUDA::genericCudaReduceCall(N, input, input_stride, input_shift, zero,
                                  cuda_dest.get(), dest_shift,
                                  AprilMath::Functors::r_add<T>());
      cuda_dest->getValue(dest_shift, result);
    }
    else {
#endif
      const T *v_mem = input->getPPALForRead() + input_shift;
      for (unsigned int i=0; i<N; ++i, v_mem+=input_stride) {
        result += *v_mem;
      }
      if (dest != 0) {
        T *dest_ptr = dest->getPPALForWrite() + dest_shift;
        *dest_ptr = result;
      }
#ifdef USE_CUDA
    }
#endif
    return result;
  }

  template<typename T, typename F>
  T genericReduceMinMaxCall(unsigned int N,
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
    T result(zero);
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      AprilUtils::SharedPtr< GPUMirroredMemoryBlock<int32_t> > cuda_which(which);
      AprilUtils::SharedPtr< GPUMirroredMemoryBlock<T> > cuda_dest(dest);
      if (cuda_which.empty()) cuda_which = new GPUMirroredMemoryBlock<int32_t>(1);
      if (cuda_dest.empty()) cuda_dest = new GPUMirroredMemoryBlock<T>(1);
      CUDA::genericCudaReduceMinMaxCall(N, input, input_stride, input_shift, zero,
                                        cuda_which.get(), which_shift,
                                        cuda_dest.get(), dest_shift,
                                        reduce_op);
      cuda_dest->getValue(dest_shift, result);
    }
    else {
#endif
      unsigned int w=0, best=0;
      const T *v_mem = input->getPPALForRead() + input_shift;
      for (unsigned int i=0; i<N; ++i, v_mem+=input_stride) {
        result = reduce_op(result, *v_mem, w);
        if (w == 1) best = i;
      }
      if (dest != 0) {
        T *dest_ptr = dest->getPPALForWrite() + dest_shift;
        *dest_ptr = result;
      }
      if (which != 0) {
        int32_t *which_ptr = which->getPPALForWrite() + which_shift;
        // best+1 because in Lua it starts at 1
        *which_ptr = static_cast<int32_t>(best+1);
      }
#ifdef USE_CUDA
    }
#endif
    return result;
  }

  template<typename T, typename OP>
  struct ScalarToSpanReduce {
    const OP functor;
    ScalarToSpanReduce(const OP &functor) : functor(functor) { }
    T operator()(unsigned int N,
                 const GPUMirroredMemoryBlock<T> *input,
                 unsigned int input_stride,
                 unsigned int input_shift,
                 bool use_cuda,
                 const T &zero,
                 GPUMirroredMemoryBlock<T> *dest=0,
                 unsigned int dest_raw_pos=0) const {
      return genericReduceCall(N, input, input_stride, input_shift,
                               use_cuda, zero, functor,
                               dest, dest_raw_pos);
    }
  };

  template<typename T, typename OP>
  struct ScalarToSpanReduceMinMax {
    const OP functor;
    ScalarToSpanReduceMinMax(const OP &functor) : functor(functor) { }
    T operator()(unsigned int N,
                 const GPUMirroredMemoryBlock<T> *input,
                 unsigned int input_stride,
                 unsigned int input_shift,
                 bool use_cuda,
                 const T &zero,
                 GPUMirroredMemoryBlock<int32_t> *which=0,
                 unsigned int which_raw_pos=0,
                 GPUMirroredMemoryBlock<T> *dest=0,
                 unsigned int dest_raw_pos=0) const {
      return genericReduceMinMaxCall(N, input, input_stride, input_shift,
                                     use_cuda, zero, functor,
                                     which, which_raw_pos,
                                     dest, dest_raw_pos);
    }
  };
  
} // namespace AprilMath

#endif // REDUCE_TEMPLATE_H
