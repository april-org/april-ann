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
   * @tparam O - The type for output vector (reduction type).
   * @tparam F - A functor typename.
   * @tparam P - A functor typename.
   *
   * @param input - The input vector.
   * @param input_stride - The stride between consecutive values at input.
   * @param input_shift - The first valid position at input vector.
   * @param zero - The value of reduce over zero elements.
   * @param reduce_op - The functor operator with the reduce over two inputs.
   * @param partials_reduce_op - The functor operator with the reduce over two
   * partial results.
   * @param[in,out] dest - The output vector, it content will be reduced with
   * the result of the current reduction.
   * @param dest_shift - The first valid position at dest vector.
   * @param set_dest_to_zero - A boolean indicating if dest will be set to zero
   * or its content will be reused in the reduction.
   *
   * @note Only the position located at dest_shift will be written in dest
   * vector.
   *
   * @note For CUDA compilation, it is needed that all functors have been
   * exported using APRIL_CUDA_EXPORT macro and to be inline operations which
   * can call functions defined for host and device.
   *
   * @note reduce_op implements <tt>void operator()(O &acc, const T &other) const</tt>
   *
   * @note partials_reduce_op implements <tt>void operator()(O &acc, const O &other) const</tt>
   */
  template<typename T, typename O, typename F, typename P>
  void genericReduce1Call(unsigned int N,
                          const GPUMirroredMemoryBlock<T> *input,
                          unsigned int input_stride,
                          unsigned int input_shift,
                          bool use_gpu,
                          const O &zero,
                          F reduce_op,
                          P partials_reduce_op,
                          GPUMirroredMemoryBlock<O> *dest,
                          unsigned int dest_shift,
                          bool set_dest_to_zero) {
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
    UNUSED_VARIABLE(partials_reduce_op);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      CUDA::genericCudaReduce1Call(N, input, input_stride, input_shift,
                                   zero,
                                   dest, dest_shift,
                                   set_dest_to_zero,
                                   reduce_op, partials_reduce_op);
    }
    else {
#endif
      const T *v_mem = input->getPPALForRead() + input_shift;
      O *dest_ptr;
      if (set_dest_to_zero) {
        dest_ptr = dest->getPPALForWrite() + dest_shift;
        *dest_ptr = zero;
      }
      else {
        dest_ptr = dest->getPPALForReadAndWrite() + dest_shift;
      }
      for (unsigned int i=0; i<N; ++i, v_mem+=input_stride) {
        reduce_op(*dest_ptr, *v_mem);
      }
#ifdef USE_CUDA
    }
#endif
  }

  /**
   * @brief Performs a reduce over two vectors and stores its result at
   * another vector.
   *
   * @tparam T1 - The type for input1 vectors.
   * @tparam T2 - The type for input1 vectors.
   * @tparam O - The type for output vector (reduction type).
   * @tparam F - A functor typename.
   * @tparam P - A functor typename.
   *
   * @param input1 - The input1 vector.
   * @param input1_stride - The stride between consecutive values at input1.
   * @param input1_shift - The first valid position at input1 vector.
   * @param input2 - The input2 vector.
   * @param input2_stride - The stride between consecutive values at input2.
   * @param input2_shift - The first valid position at input2 vector.
   * @param zero - The value of reduce over zero elements.
   * @param reduce_op - The functor operator with the reduce over two inputs.
   * @param partials_reduce_op - The functor operator with the reduce over two
   * partial results.
   * @param[in,out] dest - The output vector, it content will be reduced with
   * the result of the current reduction.
   * @param dest_shift - The first valid position at dest vector.
   * @param set_dest_to_zero - A boolean indicating if dest will be set to zero
   * or its content will be reused in the reduction.
   *
   * @note Only the position located at dest_shift will be written in dest
   * vector.
   *
   * @note For CUDA compilation, it is needed that all functros have been
   * exported using APRIL_CUDA_EXPORT macro and to be inline operations which
   * can call functions defined for host and device.
   *
   * @note reduce_op implements <tt>void operator()(O &acc, const T1 &other1, const T2 &other2) const</tt>
   *
   * @note partials_reduce_op implements <tt>void operator()(O &acc, const O &other) const</tt>
   */
  template<typename T1, typename T2, typename O, typename F, typename P>
  void genericReduce2Call(unsigned int N,
                          const GPUMirroredMemoryBlock<T1> *input1,
                          unsigned int input1_stride,
                          unsigned int input1_shift,
                          const GPUMirroredMemoryBlock<T2> *input2,
                          unsigned int input2_stride,
                          unsigned int input2_shift,
                          bool use_gpu,
                          const O &zero,
                          F reduce_op,
                          P partials_reduce_op,
                          GPUMirroredMemoryBlock<O> *dest,
                          unsigned int dest_shift,
                          bool set_dest_to_zero) {
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
    UNUSED_VARIABLE(partials_reduce_op);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      CUDA::genericCudaReduce2Call(N, input1, input1_stride, input1_shift,
                                   input2, input2_stride, input2_shift,
                                   zero,
                                   dest, dest_shift,
                                   set_dest_to_zero,
                                   reduce_op, partials_reduce_op);
    }
    else {
#endif
      const T1 *v1_mem = input1->getPPALForRead() + input1_shift;
      const T2 *v2_mem = input2->getPPALForRead() + input2_shift;
      O *dest_ptr;
      if (set_dest_to_zero) {
        dest_ptr = dest->getPPALForWrite() + dest_shift;
        *dest_ptr = zero;
      }
      else {
        dest_ptr = dest->getPPALForReadAndWrite() + dest_shift;
      }
      for (unsigned int i=0; i<N; ++i,
             v1_mem+=input1_stride, v2_mem+=input2_stride) {
        reduce_op(*dest_ptr, *v1_mem, *v2_mem);
      }
#ifdef USE_CUDA
    }
#endif
  }

  /**
   * @brief Performs a reduce over a vector and stores its result at
   * another vector.
   *
   * @tparam T - The type for input vectors.
   * @tparam O - The type for output vector (reduction type).
   * @tparam F - A functor typename.
   *
   * @param input - The input vector.
   * @param input_stride - The stride between consecutive values at input.
   * @param input_shift - The first valid position at input vector.
   * @param zero - The value of reduce over zero elements.
   * @param reduce_op - The functor operator with the reduce over two inputs.
   * @param partials_reduce_op - The functor operator with the reduce over two
   * partial results.
   * @param[in,out] dest - The output vector, it content will be reduced with
   * the result of the current reduction.
   * @param dest_shift - The first valid position at dest vector.
   * @param set_dest_to_zero - A boolean indicating if dest will be set to zero
   * or its content will be reused in the reduction.
   *
   * @note Only the position located at dest_shift will be written in dest
   * vector.
   *
   * @note For CUDA compilation, it is needed that all functors have been
   * exported using APRIL_CUDA_EXPORT macro and to be inline operations which
   * can call functions defined for host and device.
   *
   * @note reduce_op implements <tt>void operator()(T &, const T &, int32_t &, const int32_t &) const</tt>
   */  
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
                               unsigned int dest_shift,
                               bool set_dest_to_zero) {
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
    UNUSED_VARIABLE(zero);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      CUDA::genericCudaReduceMinMaxCall(N, input, input_stride, input_shift,
                                        zero,
                                        which, which_shift,
                                        dest, dest_shift,
                                        set_dest_to_zero,
                                        reduce_op);
    }
    else {
#endif
      const T *v_mem = input->getPPALForRead() + input_shift;
      T *dest_ptr;
      int32_t *which_ptr;
      if (set_dest_to_zero) {
        dest_ptr = dest->getPPALForWrite() + dest_shift;
        which_ptr = which->getPPALForWrite() + which_shift;
        *dest_ptr = zero;
        *which_ptr = 0;
      }
      else {
        dest_ptr = dest->getPPALForReadAndWrite() + dest_shift;
        which_ptr = which->getPPALForReadAndWrite() + which_shift;
      }
      for (unsigned int i=0; i<N; ++i, v_mem+=input_stride) {
        // i+1 because Lua startas at 1.
        reduce_op(*dest_ptr, *v_mem, *which_ptr, static_cast<int>(i+1));
      }
#ifdef USE_CUDA
    }
#endif
  }
  
  /**
   * @brief A functor which transforms a unary reduction functor into a
   * unary span reduction functor.
   *
   * @tparam T - The input type.
   * @tparam O - The output type.
   * @tparam OP1 - The type of the unary functor.
   *
   * @note OP1 implements <tt>void operator()(O &acc, const T &other) const</tt>
   *
   * @note The unary span is the concept of unary reduction applied over a vector
   * span.
   *
   * @note This struct is used by AprilMath::MatrixExt reduce functions to
   * force user-defined reductions to follow CBLAS wrappers API.
   *
   * @see AprilMath::MatrixExt for more information about API of CBLAS and
   * user-defined reductions.
   */
  template<typename T, typename O, typename OP1>
  struct ScalarToSpanReduce1 {
    /// An instance of the unary functor.
    const OP1 functor;
    /// The constructor stores the functor.
    ScalarToSpanReduce1(const OP1 &functor) : functor(functor) { }
    template<typename OP2>
    /**
     * @brief The unary span functor implements an operator() which receives the
     * vector (pointer, size, stride and shift/offset) where @c functor will be
     * applied.
     *
     * Additionally, it needs another @c functor2 for reduction of partial
     * results.
     *
     * @param N - The size of the vector.
     * @param input - A pointer to the vector.
     * @param input_stride - The stride between consecutive elements.
     * @param input_shift - The position of the first valid element.
     * @param use_cuda - A boolean indicating if use or not CUDA.
     * @param zero - The initial value of the reduction.
     * @param functor2 - The functor for partial results reduction.
     * @param dest - A pointer to a destination memory block.
     * @param dest_raw_pos - The position in dest where result will be stored.
     * @param set_dest_to_zero - Indicates if dest will be set to zero or its
     * content will be reused in the reduction.
     *
     * @note functor2 implements <tt>void operator(O &acc, const O &other) const</tt>
     */
    void operator()(unsigned int N,
                    const GPUMirroredMemoryBlock<T> *input,
                    unsigned int input_stride,
                    unsigned int input_shift,
                    bool use_cuda,
                    const O &zero,
                    const OP2 &functor2,
                    GPUMirroredMemoryBlock<O> *dest,
                    unsigned int dest_raw_pos,
                    bool set_dest_to_zero) const {
      genericReduce1Call(N, input, input_stride, input_shift,
                         use_cuda, zero, functor, functor2,
                         dest, dest_raw_pos, set_dest_to_zero);
    }
  };

  /**
   * @brief A functor which transforms a binary reduction functor into a
   * binary span reduction functor.
   *
   * @tparam T1 - The input1 type.
   * @tparam T2 - The input2 type.
   * @tparam O - The output type.
   * @tparam OP1 - The type of the unary functor.
   *
   * @note OP1 implements <tt>void operator()(O &acc, const T1 &other, const T2 &other) const</tt>
   *
   * @note The binary span is the concept of binary reduction applied over a vector
   * span.
   *
   * @note This struct is used by AprilMath::MatrixExt reduce functions to
   * force user-defined reductions to follow CBLAS wrappers API.
   *
   * @see AprilMath::MatrixExt for more information about API of CBLAS and
   * user-defined reductions.
   */
  template<typename T1, typename T2, typename O, typename OP1>
  struct ScalarToSpanReduce2 {
    /// An instance of the given functor typename.
    const OP1 functor;
    /// The constructor stores the given functor.
    ScalarToSpanReduce2(const OP1 &functor) : functor(functor) { }
    /**
     * @brief The binary span functor implements an operator() which receives
     * the input vectors (pointer, size, stride and shift/offset) where @c
     * functor will be applied.
     *
     * Additionally, it needs another @c functor2 for reduction of partial
     * results.
     *
     * @param N - The size of the vector.
     * @param input1 - A pointer to the vector.
     * @param input1_stride - The stride between consecutive elements.
     * @param input1_shift - The position of the first valid element.
     * @param input2 - A pointer to the vector.
     * @param input2_stride - The stride between consecutive elements.
     * @param input2_shift - The position of the first valid element.
     * @param use_cuda - A boolean indicating if use or not CUDA.
     * @param zero - The initial value of the reduction.
     * @param functor2 - The functor for partial results reduction.
     * @param dest - A pointer to a destination memory block.
     * @param dest_raw_pos - The position in dest where result will be stored.
     * @param set_dest_to_zero - Indicates if dest will be set to zero or its
     * content will be reused in the reduction.
     *
     * @note functor2 implements <tt>void operator(O &acc, const O &other) const</tt>
     */
    template<typename OP2>
    void operator()(unsigned int N,
                    const GPUMirroredMemoryBlock<T1> *input1,
                    unsigned int input1_stride,
                    unsigned int input1_shift,
                    const GPUMirroredMemoryBlock<T2> *input2,
                    unsigned int input2_stride,
                    unsigned int input2_shift,
                    bool use_cuda,
                    const O &zero,
                    const OP2 &functor2,
                    GPUMirroredMemoryBlock<O> *dest,
                    unsigned int dest_raw_pos,
                    bool set_dest_to_zero) const {
      genericReduce2Call(N, input1, input1_stride, input1_shift,
                         input2, input2_stride, input2_shift,
                         use_cuda, zero, functor, functor2,
                         dest, dest_raw_pos, set_dest_to_zero);
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
                    unsigned int dest_raw_pos,
                    bool set_dest_to_zero) const {
      UNUSED_VARIABLE(functor2);
      genericReduceMinMaxCall(N, input, input_stride, input_shift,
                              use_cuda, zero, functor,
                              which, which_raw_pos,
                              dest, dest_raw_pos, set_dest_to_zero);
    }
  };
  
} // namespace AprilMath

#endif // REDUCE_TEMPLATE_H
