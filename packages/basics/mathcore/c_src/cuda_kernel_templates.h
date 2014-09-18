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
#ifndef CUDA_KERNEL_TEMPLATES_H
#define CUDA_KERNEL_TEMPLATES_H

#ifdef USE_CUDA

#include "ceiling_power_of_two.h"
#include "complex_number.h"
#include "cublas_error.h"
#include "cuda_utils.h"
#include "cusparse_error.h"
#include "gpu_mirrored_memory_block.h"
#include "gpu_helper.h"
#include "smart_ptr.h"
#include "unused_variable.h"

namespace AprilMath {
  
  namespace CUDA {
    
    /////////////////////////////// REDUCE ////////////////////////////////////    

    /**
     * @brief Two pass CUDA reduce kernel.
     *
     * Performs a reduction with @c T input type and @c O output type, by using
     * @c F functor to reduce input data into output data, and @c P functor to
     * reduce intermediate output data.
     *
     * @tparam T - The type for input vectors.
     * @tparam O - The type for output vectors.
     * @tparam F - An operator implemented as a functor.
     * @tparam P - An operator implemented as a functor.
     * @tparam overwrite_output - A boolean indicating if output data has to be
     * ignored (overwritten) or reduced with the result of this computation.
     *
     * @param input - The input vector.
     * @param input_stride - The stride between consecutive values at input.
     * @param output - The output vector.
     * @param N - Size of the input vector.
     * @param reduce_op - The functor operator with the reduction over two
     * values.
     * @param partials_reduce_op - The functor operator with the reduction over
     * two partial values.
     * @param zero - The initial value for the reduction.
     *
     * @note Adapted from CUDA Handbook:
     *   - http://www.cudahandbook.com/
     *   - https://github.com/ArchaeaSoftware/cudahandbook
     *
     * @code
     * // First pass call.
     * genericCudaReduce1Kernel<false>
     *   <<<num_blocks, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
     *   (input_ptr, input_stride, blocks_ptr, 1u, N, reduce_op,
     *   partials_reduce_op, zero);
     * // Second pass call.
     * genericCudaReduce1Kernel<true>
     *   <<<1, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
     *   (blocks_ptr, 1u, dest_ptr, 1u,
     *   static_cast<unsigned int>(num_blocks), partials_reduce_op,
     *   partials_reduce_op, zero);
     * @endcode
     */
    template<bool overwrite_output,
             typename T, typename O, typename F, typename P>
    __global__ void genericCudaReduce1Kernel(const T *input,
                                             unsigned int input_stride,
                                             O *output,
                                             unsigned int output_stride,
                                             unsigned int N,
                                             F reduce_op,
                                             P partials_reduce_op,
                                             O zero) {
      SharedMemory<O> partials;
      O result = zero;
      const int tid = threadIdx.x;
      for ( size_t i = blockIdx.x*blockDim.x + tid;
            i < N;
            i += blockDim.x*gridDim.x ) {
        reduce_op(result, input[i * input_stride]);
      }
      partials[tid] = result;
      __syncthreads();
      for ( int activeThreads = blockDim.x>>1; 
            activeThreads; 
            activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
          partials_reduce_op(partials[tid], partials[tid+activeThreads]);
        }
        __syncthreads();
      }
      if ( tid == 0 ) {
        if (overwrite_output) {
          output[blockIdx.x * output_stride] = zero;
        }
        partials_reduce_op(output[blockIdx.x * output_stride], partials[0]);
      }
    }

    /**
     * @brief Two pass CUDA reduce kernel for two input vectors.
     *
     * Performs a reduction with @c T1 and @c T2 input types and @c O output
     * type, by using @c F functor to reduce input data into output data, and @c
     * P functor to reduce intermediate output data.
     *
     * @tparam T1 - The type for input1 vectors.
     * @tparam T2 - The type for input2 vectors.
     * @tparam O - The type for output vectors.
     * @tparam F - An operator implemented as a functor.
     * @tparam P - An operator implemented as a functor.
     * @tparam overwrite_output - A boolean indicating if output data has to be
     * ignored (overwritten) or reduced with the result of this computation.
     *
     * @param input1 - The input1 vector.
     * @param input1_stride - The stride between consecutive values at input1.
     * @param input2 - The input2 vector.
     * @param input2_stride - The stride between consecutive values at input2.
     * @param output - The output vector.
     * @param N - Size of the input vector.
     * @param reduce_op - The functor operator with the reduction over two
     * values.
     * @param partials_reduce_op - The functor operator with the reduction over
     * two partial values.
     * @param zero - The initial value for the reduction.
     *
     * @note Adapted from CUDA Handbook:
     *   - http://www.cudahandbook.com/
     *   - https://github.com/ArchaeaSoftware/cudahandbook
     *
     * @code
     * // First pass call.
     * genericCudaReduce2Kernel<false>
     *   <<<num_blocks, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
     *   (input1_ptr, input1_stride, input2_ptr, input2_stride, blocks_ptr, 1u, N, reduce_op,
     *   partials_reduce_op, zero);
     * // Second pass call.
     * genericCudaReduce1Kernel<true>
     *   <<<1, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
     *   (blocks_ptr, 1u, dest_ptr, 1u,
     *   static_cast<unsigned int>(num_blocks), partials_reduce_op,
     *   partials_reduce_op, zero);
     * @endcode
     */
    template<bool overwrite_output,
             typename T1, typename T2, typename O, typename F, typename P>
    __global__ void genericCudaReduce2Kernel(const T1 *input1,
                                             unsigned int input1_stride,
                                             const T2 *input2,
                                             unsigned int input2_stride,
                                             O *output,
                                             unsigned int output_stride,
                                             unsigned int N,
                                             F reduce_op,
                                             P partials_reduce_op,
                                             O zero) {
      SharedMemory<O> partials;
      O result = zero;
      const int tid = threadIdx.x;
      for ( size_t i = blockIdx.x*blockDim.x + tid;
            i < N;
            i += blockDim.x*gridDim.x ) {
        reduce_op(result, input1[i * input1_stride], input2[i * input2_stride]);
      }
      partials[tid] = result;
      __syncthreads();
      for ( int activeThreads = blockDim.x>>1; 
            activeThreads; 
            activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
          partials_reduce_op(partials[tid], partials[tid+activeThreads]);
        }
        __syncthreads();
      }
      if ( tid == 0 ) {
        if (overwrite_output) {
          output[blockIdx.x * output_stride] = zero;
        }
        partials_reduce_op(output[blockIdx.x * output_stride], partials[0]);
      }
    }

    /**
     * @brief First pass of CUDA reduce kernel for Min/Max reduction.
     *
     * The Min/Max reduction allows to retrieve ArgMin/ArgMax value. Performs a
     * reduction with @c T input/output type, by using @c F functor to reduce
     * input data into intermediate output data vector.
     *
     * @tparam T - The type for input/output vectors.
     * @tparam F - An operator implemented as a functor for reduce_op param.
     *
     * @param input - The input vector.
     * @param input_stride - The stride between consecutive values at input.
     * @param which - The output vector for ArgMax/ArgMin result.
     * @param which_stride - The stride between consecutive values at which.
     * @param output - The output vector.
     * @param output_stride - The stride between consecutive values at output.
     * @param N - Size of the input vector.
     * @param reduce_op - The functor operator with the reduction over two
     * values.
     * @param zero - The initial value for the reduction.
     *
     * @note Adapted from CUDA Handbook:
     *   - http://www.cudahandbook.com/
     *   - https://github.com/ArchaeaSoftware/cudahandbook
     *
     * @note This kernel overwrites the content of @c which and @c output
     * vectors.
     */
    template<typename T, typename F>
    __global__ void genericCudaReduceMinMaxKernel(const T *input,
                                                  unsigned int input_stride,
                                                  int32_t *which,
                                                  unsigned int which_stride,
                                                  T *output,
                                                  unsigned int output_stride,
                                                  unsigned int N,
                                                  F reduce_op,
                                                  T zero) {
      SharedMemory<T> partials;
      int32_t *which_partials = (int32_t*)(&partials[blockDim.x]);
      int32_t which_result=0;
      T result = zero;
      const int tid = threadIdx.x;
      for ( size_t i = blockIdx.x*blockDim.x + tid; i < N;
            i += blockDim.x*gridDim.x ) {
        reduce_op(result, input[i * input_stride],
                  // i+1 because Lua startas at 1.
                  which_result, static_cast<int32_t>(i+1));
      }
      partials[tid] = result;
      which_partials[tid] = which_result;
      __syncthreads();
      for ( int activeThreads = blockDim.x>>1; activeThreads; 
            activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
          reduce_op(partials[tid], partials[tid+activeThreads],
                    which_partials[tid], which_partials[tid+activeThreads]);
          
        }
        __syncthreads();
      }
      if ( tid == 0 ) {
        // Overwrites the output
        output[blockIdx.x * output_stride] = partials[0];
        which[blockIdx.x * which_stride] = which_partials[0];
      }
    }

    /**
     * @brief First pass of CUDA reduce kernel for Min/Max reduction.
     *
     * The Min/Max reduction allows to retrieve ArgMin/ArgMax value. Performs a
     * reduction with @c T input/output type, by using @c F functor to reduce
     * input data into intermediate output data vector.
     *
     * @tparam T - The type for input/output vectors.
     * @tparam F - An operator implemented as a functor for reduce_op param.
     * @tparam overwrite_output - A boolean indicating if output data has to be
     * ignored (overwritten) or reduced with the result of this computation.
     *
     * @param which_input - The input vector with intermediate ArgMin/ArgMax values.
     * @param which_input_stride - The stride between consecutive values at input.
     * @param input - The input vector.
     * @param input_stride - The stride between consecutive values at input.
     * @param which - The output for ArgMax/ArgMin result.
     * @param which_stride - The stride between consecutive values at which.
     * @param output - The output vector.
     * @param output_stride - The stride between consecutive values at output.
     * @param N - Size of the input vector.
     * @param reduce_op - The functor operator with the reduction over two
     * values.
     * @param zero - The initial value for the reduction.
     *
     * @note Adapted from CUDA Handbook:
     *   - http://www.cudahandbook.com/
     *   - https://github.com/ArchaeaSoftware/cudahandbook
     */
    template<bool overwrite_output, typename T, typename F>
    __global__ void genericCudaReduceMinMaxKernel2(const int32_t *which_input,
                                                   unsigned int which_input_stride,
                                                   const T *input,
                                                   unsigned int input_stride,
                                                   int32_t *which,
                                                   unsigned int which_stride,
                                                   T *output,
                                                   unsigned int output_stride,
                                                   unsigned int N,
                                                   F reduce_op,
                                                   T zero) {
      SharedMemory<T> partials;
      int32_t *which_partials = (int32_t*)(&partials[blockDim.x]);
      int32_t which_result=0;
      T result = zero;
      const int tid = threadIdx.x;
      for ( size_t i = blockIdx.x*blockDim.x + tid; i < N;
            i += blockDim.x*gridDim.x ) {
        reduce_op(result, input[i * input_stride],
                  which_result, which_input[i * which_input_stride]);
      }
      partials[tid] = result;
      which_partials[tid] = which_result;
      __syncthreads();
      for ( int activeThreads = blockDim.x>>1; activeThreads; 
            activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
          reduce_op(partials[tid], partials[tid+activeThreads],
                    which_partials[tid], which_partials[tid+activeThreads]);
          
        }
        __syncthreads();
      }
      if ( tid == 0 ) {
        if (overwrite_output)  {
          output[blockIdx.x * output_stride] = zero;
          which[blockIdx.x * which_stride] = 0;
        }
        reduce_op(output[blockIdx.x * output_stride], partials[0],
                  which[blockIdx.x * which_stride], which_partials[0]);
      }
    }
    
    /**
     * @brief Performs a CUDA reduce over a vector and stores its result at
     * a given position into another vector.
     *
     * The reduce operations are performed in a two-pass fashion. First
     * reducing the input vector into an intermediate memory as large as
     * the number executed blocks. Second pass reduces the intermediate memory
     * into the given position of the output vector.
     *
     * @tparam T - The type for input vector.
     * @tparam O - The type for output vector.
     * @tparam F - An operator implemented as a functor for reduce_op param.
     * @tparam P - An operator implemented as a functor for partials_reduce_op param.
     *
     * @param input - The input vector.
     * @param input_stride - The stride between consecutive values at input.
     * @param input_shift - The first valid position at input vector.
     * @param zero - The value of reduce over zero elements.
     * @param dest - The output vector.
     * @param dest_shift - The first valid position at dest vector.
     * @param set_dest_to_zero - A boolean indicating if dest memory has to be
     * initialized to zero, otherwise its content will be reduced allowing to
     * perform chains of multiple reductions over the same dest pointer.
     * @param reduce_op - The functor operator with the reduce one input values
     * into an output accumulator.
     * @param partials_reduce_op - The functor operator with the reduce over two
     * output values.
     */
    template<typename T, typename O, typename F, typename P>
    void genericCudaReduce1Call(unsigned int N,
                                const GPUMirroredMemoryBlock<T> *input,
                                unsigned int input_stride,
                                unsigned int input_shift,
                                O zero,
                                GPUMirroredMemoryBlock<O> *dest,
                                unsigned int dest_shift,
                                bool set_dest_to_zero,
                                F reduce_op, P partials_reduce_op) {
      if (N == 0u) {
        if (set_dest_to_zero) dest->putValue(dest_shift, zero);
        return;
      }
      int num_threads, thread_size, num_blocks;
      computeReductionSize(static_cast<int>(N), num_threads,
                           thread_size, num_blocks);
      const T *input_ptr = input->getGPUForRead() + input_shift;
      if (num_blocks > 1) {
        AprilUtils::SharedPtr< GPUMirroredMemoryBlock<O> >
          blocks(new GPUMirroredMemoryBlock<O>(static_cast<unsigned int>(num_blocks)));
        O *blocks_ptr = blocks->getGPUForWrite();
        // first pass
        genericCudaReduce1Kernel<false>
          <<<num_blocks, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
          (input_ptr, input_stride, blocks_ptr, 1u, N, reduce_op,
           partials_reduce_op, zero);
        computeSecondReductionSize(num_blocks, num_threads);
        // second pass
        if (set_dest_to_zero) {
          O *dest_ptr = dest->getGPUForWrite() + dest_shift;
          genericCudaReduce1Kernel<true>
            <<<1, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (blocks_ptr, 1u, dest_ptr, 1u,
             static_cast<unsigned int>(num_blocks), partials_reduce_op,
             partials_reduce_op, zero);
        }
        else {
          O *dest_ptr = dest->getGPUForReadAndWrite() + dest_shift;
          genericCudaReduce1Kernel<false>
            <<<1, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (blocks_ptr, 1u, dest_ptr, 1u,
             static_cast<unsigned int>(num_blocks), partials_reduce_op,
             partials_reduce_op, zero);
        }
      } // num_blocks > 1
      else { // num_blocks == 1
        // only one pass
        if (set_dest_to_zero) {
          O *dest_ptr = dest->getGPUForWrite() + dest_shift;
          genericCudaReduce1Kernel<true>
            <<<1, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (input_ptr, input_stride, dest_ptr, 1u, N, reduce_op,
             partials_reduce_op, zero);
        }
        else {
          O *dest_ptr = dest->getGPUForReadAndWrite() + dest_shift;
          genericCudaReduce1Kernel<false>
            <<<1, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (input_ptr, input_stride, dest_ptr, 1u, N, reduce_op,
             partials_reduce_op, zero);
        }
      }
      // TODO: check return value (cudaError_t)
      // cudaDeviceSynchronize();
    } // function genericCudaReduce1Call


    /**
     * @brief Performs a CUDA reduce over a two vectors and stores its result at
     * a given position into another vector.
     *
     * The reduce operations are performed in a two-pass fashion. First
     * reducing the input vectors into an intermediate memory as large as
     * the number executed blocks. Second pass reduces the intermediate memory
     * into the given position of the output vector.
     *
     * @tparam T1 - The type for input1 vector.
     * @tparam T2 - The type for input2 vector.
     * @tparam O - The type for output vector.
     * @tparam F - An operator implemented as a functor for reduce_op param.
     * @tparam P - An operator implemented as a functor for partials_reduce_op param.
     *
     * @param input1 - The input1 vector.
     * @param input1_stride - The stride between consecutive values at input1.
     * @param input1_shift - The first valid position at input1 vector.
     * @param input2 - The input2 vector.
     * @param input2_stride - The stride between consecutive values at input2.
     * @param input2_shift - The first valid position at input2 vector.
     * @param zero - The value of reduce over zero elements.
     * @param dest - The output vector.
     * @param dest_shift - The first valid position at dest vector.
     * @param set_dest_to_zero - A boolean indicating if dest memory has to be
     * initialized to zero, otherwise its content will be reduced allowing to
     * perform chains of multiple reductions over the same dest pointer.
     * @param reduce_op - The functor operator with the reduce one input values
     * into an output accumulator.
     * @param partials_reduce_op - The functor operator with the reduce over two
     * output values.
     */
    template<typename T1, typename T2, typename O, typename F, typename P>
    void genericCudaReduce2Call(unsigned int N,
                                const GPUMirroredMemoryBlock<T1> *input1,
                                unsigned int input1_stride,
                                unsigned int input1_shift,
                                const GPUMirroredMemoryBlock<T2> *input2,
                                unsigned int input2_stride,
                                unsigned int input2_shift,
                                O zero,
                                GPUMirroredMemoryBlock<O> *dest,
                                unsigned int dest_shift,
                                bool set_dest_to_zero,
                                F reduce_op, P partials_reduce_op) {
      if (N == 0u) {
        if (set_dest_to_zero) dest->putValue(dest_shift, zero);
        return;
      }
      int num_threads, thread_size, num_blocks;
      computeReductionSize(static_cast<int>(N), num_threads,
                           thread_size, num_blocks);
      const T1 *input1_ptr = input1->getGPUForRead() + input1_shift;
      const T2 *input2_ptr = input2->getGPUForRead() + input2_shift;
      if (num_blocks > 1) {
        AprilUtils::SharedPtr< GPUMirroredMemoryBlock<O> >
          blocks(new GPUMirroredMemoryBlock<O>(static_cast<unsigned int>(num_blocks)));
        O *blocks_ptr = blocks->getGPUForWrite();
        // first pass
        genericCudaReduce2Kernel<false>
          <<<num_blocks, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
          (input1_ptr, input1_stride, input2_ptr, input2_stride,
           blocks_ptr, 1u, N, reduce_op,
           partials_reduce_op, zero);
        computeSecondReductionSize(num_blocks, num_threads);
        // second pass
        if (set_dest_to_zero) {
          O *dest_ptr = dest->getGPUForWrite() + dest_shift;
          genericCudaReduce1Kernel<true>
            <<<1, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (blocks_ptr, 1u, dest_ptr, 1u,
             static_cast<unsigned int>(num_blocks), partials_reduce_op,
             partials_reduce_op, zero);
        }
        else {
          O *dest_ptr = dest->getGPUForReadAndWrite() + dest_shift;
          genericCudaReduce1Kernel<false>
            <<<1, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (blocks_ptr, 1u, dest_ptr, 1u,
             static_cast<unsigned int>(num_blocks), partials_reduce_op,
             partials_reduce_op, zero);
        }
      } // num_blocks > 1
      else { // num_blocks == 1
        // only one pass
        if (set_dest_to_zero) {
          O *dest_ptr = dest->getGPUForWrite() + dest_shift;
          genericCudaReduce2Kernel<true>
            <<<1, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (input1_ptr, input1_stride, input2_ptr, input2_stride,
             dest_ptr, 1u, N, reduce_op,
             partials_reduce_op, zero);
        }
        else {
          O *dest_ptr = dest->getGPUForReadAndWrite() + dest_shift;
          genericCudaReduce2Kernel<false>
            <<<1, num_threads, num_threads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (input1_ptr, input1_stride, input2_ptr, input2_stride,
             dest_ptr, 1u, N, reduce_op,
             partials_reduce_op, zero);
        }
      }
      // TODO: check return value (cudaError_t)
      // cudaDeviceSynchronize();
    } // function genericCudaReduce2Call
    

    /**
     * @brief Performs a CUDA Min/Max reduce over a vector and stores its result
     * at a given position into another vector.
     *
     * The reduce operations are performed in a two-pass fashion. First
     * reducing the input vector into an intermediate memory as large as
     * the number executed blocks. Second pass reduces the intermediate memory
     * into the given position of the output vector.
     *
     * @tparam T - The type for input/output vectors.
     * @tparam F - An operator implemented as a functor for reduce_op param.
     *
     * @param input - The input vector.
     * @param input_stride - The stride between consecutive values at input.
     * @param input_shift - The first valid position at input vector.
     * @param zero - The value of reduce over zero elements.
     * @param which - The output vector where to store ArgMin/ArgMax result.
     * @param which_shift - The first valid position at which vector.
     * @param dest - The output vector.
     * @param dest_shift - The first valid position at dest vector.
     * @param set_dest_to_zero - A boolean indicating if dest memory has to be
     * initialized to zero, otherwise its content will be reduced allowing to
     * perform chains of multiple reductions over the same dest pointer.
     * @param reduce_op - The functor operator with the reduce one input values
     * into an output accumulator.
     */
    template<typename T, typename F>
    void genericCudaReduceMinMaxCall(unsigned int N,
                                     const GPUMirroredMemoryBlock<T> *input,
                                     unsigned int input_stride,
                                     unsigned int input_shift,
                                     T zero,
                                     GPUMirroredMemoryBlock<int32_t> *which,
                                     unsigned int which_shift,
                                     GPUMirroredMemoryBlock<T> *dest,
                                     unsigned int dest_shift,
                                     bool set_dest_to_zero,
                                     F reduce_op) {
      if (N == 0u) {
        if (set_dest_to_zero) {
          dest->putValue(dest_shift, zero);
          which->putValue(which_shift, 0);
        }
        return;
      }
      int num_threads, thread_size, num_blocks;
      computeReductionSize(static_cast<int>(N), num_threads,
                           thread_size, num_blocks);        
      const T *input_ptr = input->getGPUForRead() + input_shift;

      AprilUtils::SharedPtr< GPUMirroredMemoryBlock<T> >
        blocks(new GPUMirroredMemoryBlock<T>(static_cast<unsigned int>(num_blocks)));
      AprilUtils::SharedPtr< GPUMirroredMemoryBlock<int32_t> >
        which_blocks(new GPUMirroredMemoryBlock<int32_t>(static_cast<unsigned int>(num_blocks)));
      T *blocks_ptr = blocks->getGPUForWrite();
      int32_t *which_blocks_ptr = which_blocks->getGPUForWrite();
      // First pass
      genericCudaReduceMinMaxKernel<<<num_blocks, num_threads, num_threads*(sizeof(T)+sizeof(int32_t)),
        GPUHelper::getCurrentStream()>>>
        (input_ptr, input_stride, which_blocks_ptr, 1u, blocks_ptr, 1u,
         N, reduce_op, zero);
      computeSecondReductionSize(num_blocks, num_threads);
      if (set_dest_to_zero) {
        int32_t *which_ptr = which->getGPUForWrite() + which_shift;
        T *dest_ptr = dest->getGPUForWrite() + dest_shift;
        // Second pass
        genericCudaReduceMinMaxKernel2<true><<<num_blocks, num_threads, num_threads*(sizeof(T)+sizeof(int32_t)),
          GPUHelper::getCurrentStream()>>>
          (which_blocks_ptr, 1u, blocks_ptr, 1u,
           which_ptr, 1u,  dest_ptr, 1u,
           static_cast<unsigned int>(num_blocks), reduce_op, zero);
      }
      else  {
        int32_t *which_ptr = which->getGPUForReadAndWrite() + which_shift;
        T *dest_ptr = dest->getGPUForReadAndWrite() + dest_shift;
        // Second pass
        genericCudaReduceMinMaxKernel2<false><<<num_blocks, num_threads, num_threads*(sizeof(T)+sizeof(int32_t)),
          GPUHelper::getCurrentStream()>>>
          (which_blocks_ptr, 1u, blocks_ptr, 1u,
           which_ptr, 1u,  dest_ptr, 1u,
           static_cast<unsigned int>(num_blocks), reduce_op, zero);
      }
      // TODO: check return value (cudaError_t)
      // cudaDeviceSynchronize();
    } // function genericCudaReduceMinMaxCall

    /////////////////////////////// MAP ////////////////////////////////////
    
    /**
     * @brief Executes the CUDA map kernel over a vector position.
     *
     * @tparam T - The type for input vector.
     * @tparam O - The type for output vector.
     * @tparam F - An operator implemented as a functor.
     *
     * @param input - The input vector.
     * @param input_stride - The stride between consecutive values at input.
     * @param output - The output vector.
     * @param output_stride - The stride between consecutive values at output.
     * @param size - Size of the input and output vectors.
     * @param map_op - The functor operator with the unary map function.
     */
    template<typename T, typename O, typename F>
    __global__ void genericCudaMap1Kernel(const T *input,
                                          unsigned int input_stride,
                                          O *output,
                                          unsigned int output_stride,
                                          unsigned int size,
                                          F map_op) {
      for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
            i < size;
            i += blockDim.x*gridDim.x ) {
        output[i*output_stride] = map_op(input[i*input_stride]);
      }
    }

    /**
     * @brief Executes the CUDA map kernel over two vectors.
     *
     * @tparam T1 - The type for input1 vector.
     * @tparam T2 - The type for input2 vector.
     * @tparam O - The type for output vector.
     * @tparam F - An operator implemented as a functor.
     *
     * @param input1 - The first input vector.
     * @param input1_stride - The stride between consecutive values at input1.
     * @param input2 - The second input vector.
     * @param input2_stride - The stride between consecutive values at input2.
     * @param output - The output vector.
     * @param output_stride - The stride between consecutive values at output.
     * @param size - Size of the input and output vectors.
     * @param map_op - The functor operator with the binary map function.
     */
    template<typename T1, typename T2, typename O, typename F>
    __global__ void genericCudaMap2Kernel(const T1 *input1,
                                          unsigned int input1_stride,
                                          const T2 *input2,
                                          unsigned int input2_stride,
                                          O *output,
                                          unsigned int output_stride,
                                          unsigned int size,
                                          F map_op) {
      for ( size_t i = blockIdx.x*blockDim.x + threadIdx.x;
            i < size;
            i += blockDim.x*gridDim.x ) {
        output[i*output_stride] = map_op(input1[i*input1_stride],
                                         input2[i*input2_stride]);
      }
    }

    /**
     * @brief Performs a CUDA map over a vector and stores its result at
     * another vector.
     *
     * @note Input and output vectors can be the same.
     *
     * The reduce operations performed in a logarithmic way, reducing the
     * size of the vector by a factor of two in every iteration. So, this
     * function needs <tt>O(log N)</tt> sequential iterations to perform the
     * required operation.
     *
     * @tparam T - The type for input vector.
     * @tparam O - The type for output vector.
     * @tparam F - An operator implemented as a functor.
     *
     * @param input - The input vector.
     * @param input_stride - The stride between consecutive values at input.
     * @param input_shift - The first valid position at input vector.
     * @param output - The output vector.
     * @param output_stride - The stride between consecutive values at output.
     * @param output_shift - The first valid position at output vector.
     * @param map_op - The functor operator with the unary map.
     */
    template<typename T, typename O, typename F>
    void genericCudaMap1Call(unsigned int N,
                             const GPUMirroredMemoryBlock<T> *input,
                             unsigned int input_stride,
                             unsigned int input_shift,
                             GPUMirroredMemoryBlock<O> *output,
                             unsigned int output_stride,
                             unsigned int output_shift,
                             F map_op) {
      if (N == 0u) return; // Nothing to do!
      const T *input_ptr = input->getGPUForRead() + input_shift;
      O *output_ptr = output->getGPUForWrite() + output_shift;
      int num_threads, num_blocks;
      computeBlockAndGridSizesForArray(N, num_threads, num_blocks);
      genericCudaMap1Kernel<<<num_blocks, num_threads, 0, GPUHelper::getCurrentStream()>>>
        (input_ptr, input_stride, output_ptr, output_stride, N, map_op);
    } // function genericCudaMap1Call

    /**
     * @brief Performs a CUDA map over two vectors and stores its result at
     * another vector.
     *
     * @note Input1, input2 and output vectors can be the same.
     *
     * The reduce operations is performed in a logarithmic way, reducing the
     * size of the vector by a factor of two in every iteration. So, this
     * function needs <tt>O(log N)</tt> sequential iterations to perform the
     * required operation.
     *
     * @tparam T1 - The type for input1 vector.
     * @tparam T2 - The type for input2 vector.
     * @tparam F - An operator implemented as a functor.
     *
     * @param input1 - The first input vector.
     * @param input1_stride - The stride between consecutive values at input1.
     * @param input1_shift - The first valid position at input1.
     * @param input2 - The second input vector.
     * @param input2_stride - The stride between consecutive values at input2.
     * @param input2_shift - The first valid position at input2.
     * @param output - The output vector.
     * @param output_stride - The stride between consecutive values at output.
     * @param output_shift - The first valid position at output vector.
     * @param map_op - The functor operator with the binary map.
     */
    template<typename T1, typename T2, typename O, typename F>
    void genericCudaMap2Call(unsigned int N,
                             const GPUMirroredMemoryBlock<T1> *input1,
                             unsigned int input1_stride,
                             unsigned int input1_shift,
                             const GPUMirroredMemoryBlock<T2> *input2,
                             unsigned int input2_stride,
                             unsigned int input2_shift,
                             GPUMirroredMemoryBlock<O> *output,
                             unsigned int output_stride,
                             unsigned int output_shift,
                             F map_op) {
      if (N == 0u) return; // Nothing to do!
      const T1 *input1_ptr = input1->getGPUForRead() + input1_shift;
      const T2 *input2_ptr = input2->getGPUForRead() + input2_shift;
      O *output_ptr = output->getGPUForWrite() + output_shift;
      dim3 block, grid;
      int num_threads, num_blocks;
      computeBlockAndGridSizesForArray(N, num_threads, num_blocks);
      genericCudaMap2Kernel<<<num_blocks, num_threads, 0, GPUHelper::getCurrentStream()>>>
        (input1_ptr, input1_stride, input2_ptr, input2_stride,
         output_ptr, output_stride, N, map_op);
    } // function genericCudaMap1Call
    
  } // namespace CUDA

} // namespace AprilMath

#endif // USE_CUDA

#endif // CUDA_KERNEL_TEMPLATES_H
