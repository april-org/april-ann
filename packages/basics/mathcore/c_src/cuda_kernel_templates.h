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

    // From CUDA Handbook:
    //   - http://www.cudahandbook.com/
    //   - https://github.com/ArchaeaSoftware/cudahandbook
    
    /**
     * @brief Executes the CUDA reduce kernel over a vector position.
     *
     * @tparam T - The type for input vectors.
     *
     * @tparam O - The type for output vectors.
     *
     * @tparam F - An operator implemented as a functor.
     *
     * @tparam P - An operator implemented as a functor.
     *
     * @param input - The input vector.
     *
     * @param input_stride - The stride between consecutive values at input.
     *
     * @param output - The output vector.
     *
     * @param reduction_top - Bound for the reduction, it must be the floor
     * power of two of size parameter.
     *
     * @param size - Size of the input vector.
     *
     * @param reduce_op - The functor operator with the reduction over two
     * values.
     *
     * @param partials_reduce_op - The functor operator with the reduction over
     * two partial values.
     *
     * @param zero - The initial value for the reduction.
     */
    template<typename T, typename O, typename F, typename P,
             bool overwrite_output>
    __global__ void genericCudaReduceKernel(const T *input,
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
        reduce_op(result, input[i * input_stride], which_result, static_cast<int32_t>(i));
      }
      partials[tid] = result;
      which_partials[tid] = which_result;
      __syncthreads();
      for ( int activeThreads = blockDim.x>>1; activeThreads; 
            activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
          reduce_op(partials[tid], partials[tid+activeThreads],
                    which_partials[tid],
                    which_partials[tid+activeThreads]);
          
        }
        __syncthreads();
      }
      if ( tid == 0 ) {
        // Overwrites the output
        output[blockIdx.x * output_stride] = partials[0];
        which[blockIdx.x * which_stride] = which_partials[0];
      }
    }

    template<typename T, typename F, bool overwrite_output>
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
                    which_partials[tid],
                    which_partials[tid+activeThreads]);
          
        }
        __syncthreads();
      }
      if ( tid == 0 ) {
        if (overwrite_output)  {
          output[blockIdx.x * output_stride] = zero;
          which[blockIdx.x * which_stride] = -1;
        }
        reduce_op(output[blockIdx.x * output_stride], partials[0],
                  which[blockIdx.x * which_stride],
                  which_partials[0]);
      }
    }
    
    /**
     * @brief Performs a CUDA reduce over a vector and stores its result at
     * another vector.
     *
     * The reduce operations performed in a logarithmic way, reducing the
     * size of the vector by a factor of two in every iteration. So, this
     * function needs <tt>O(log N)</tt> sequential iterations to perform the
     * required operation.
     *
     * @tparam T - The type for input vector.
     *
     * @tparam O - The type for output vector.
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
     * @param dest - The output vector.
     *
     * @param dest_shift - The first valid position at dest vector.
     *
     * @param reduce_op - The functor operator with the reduce one input values
     * into an output accumulator.
     *
     * @param partials_reduce_op - The functor operator with the reduce over two
     * output values.
     *
     * @note It is assumed that dest memory has been initialized properly to
     * zero value, allowing to chain multiple reductions over the same dest
     * pointer.
     */
    template<typename T, typename O, typename F, typename P>
    void genericCudaReduceCall(unsigned int N,
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
      int numThreads, threadSize, numBlocks;
      computeReductionSize(static_cast<int>(N), numThreads,
                           threadSize, numBlocks);
      const T *input_ptr = input->getGPUForRead() + input_shift;
      O *dest_ptr = dest->getGPUForWrite() + dest_shift;
      if (numBlocks > 1) {
        AprilUtils::SharedPtr< GPUMirroredMemoryBlock<O> >
          blocks(new GPUMirroredMemoryBlock<O>(static_cast<unsigned int>(numBlocks)));
        O *blocks_ptr = blocks->getGPUForWrite();
        genericCudaReduceKernel<T,O,F,P,false>
          <<<numBlocks, numThreads, numThreads*sizeof(O), GPUHelper::getCurrentStream()>>>
          (input_ptr, input_stride, blocks_ptr, 1u, N, reduce_op,
           partials_reduce_op, zero);
        computeSecondReductionSize(numBlocks, numThreads);
        if (set_dest_to_zero) {
          genericCudaReduceKernel<T,O,F,P,true>
            <<<1, numThreads, numThreads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (blocks_ptr, 1u, dest_ptr, 1u,
             static_cast<unsigned int>(numBlocks), reduce_op,
             partials_reduce_op, zero);
        }
        else {
          genericCudaReduceKernel<T,O,F,P,false>
            <<<1, numThreads, numThreads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (blocks_ptr, 1u, dest_ptr, 1u,
             static_cast<unsigned int>(numBlocks), reduce_op,
             partials_reduce_op, zero);
        }
      } // numBlocks > 1
      else { // numBlocks == 1
        if (set_dest_to_zero) {
          genericCudaReduceKernel<T,O,F,P,true>
            <<<1, numThreads, numThreads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (input_ptr, input_stride, dest_ptr, 1u, N, reduce_op,
             partials_reduce_op, zero);
        }
        else {
          genericCudaReduceKernel<T,O,F,P,false>
            <<<1, numThreads, numThreads*sizeof(O), GPUHelper::getCurrentStream()>>>
            (input_ptr, input_stride, dest_ptr, 1u, N, reduce_op,
             partials_reduce_op, zero);
        }
      }
      // TODO: check return value (cudaError_t)
      // cudaDeviceSynchronize();
    } // function genericCudaReduceCall
    
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
          which->putValue(which_shift, -1);
        }
        return;
      }
      int numThreads, threadSize, numBlocks;
      computeReductionSize(static_cast<int>(N), numThreads,
                           threadSize, numBlocks);        
      const T *input_ptr = input->getGPUForRead() + input_shift;
      int32_t *which_ptr = which->getGPUForReadAndWrite() + which_shift;
      T *dest_ptr = dest->getGPUForReadAndWrite() + dest_shift;

      AprilUtils::SharedPtr< GPUMirroredMemoryBlock<T> >
        blocks(new GPUMirroredMemoryBlock<T>(static_cast<unsigned int>(numBlocks)));
      AprilUtils::SharedPtr< GPUMirroredMemoryBlock<int32_t> >
        which_blocks(new GPUMirroredMemoryBlock<int32_t>(static_cast<unsigned int>(numBlocks)));
      T *blocks_ptr = blocks->getGPUForWrite();
      int32_t *which_blocks_ptr = which_blocks->getGPUForWrite();
      
      // First pass
      genericCudaReduceMinMaxKernel<<<numBlocks, numThreads, numThreads*(sizeof(T)+sizeof(int32_t)),
        GPUHelper::getCurrentStream()>>>
        (input_ptr, input_stride, which_blocks_ptr, 1u, blocks_ptr, 1u,
         N, reduce_op, zero);
      computeSecondReductionSize(numBlocks, numThreads);
      if (set_dest_to_zero) {
        // Second pass
        genericCudaReduceMinMaxKernel2<T,F,true><<<numBlocks, numThreads, numThreads*(sizeof(T)+sizeof(int32_t)),
          GPUHelper::getCurrentStream()>>>
          (which_blocks_ptr, 1u, blocks_ptr, 1u,
           which_ptr, 1u,  dest_ptr, 1u,
           static_cast<unsigned int>(numBlocks), reduce_op, zero);
      }
      else  {
        // Second pass
        genericCudaReduceMinMaxKernel2<T,F,false><<<numBlocks, numThreads, numThreads*(sizeof(T)+sizeof(int32_t)),
          GPUHelper::getCurrentStream()>>>
          (which_blocks_ptr, 1u, blocks_ptr, 1u,
           which_ptr, 1u,  dest_ptr, 1u,
           static_cast<unsigned int>(numBlocks), reduce_op, zero);
      }
      // TODO: check return value (cudaError_t)
      // cudaDeviceSynchronize();
    } // function genericCudaReduceMinMaxCall

    /////////////////////////////// MAP ////////////////////////////////////
    
    /**
     * @brief Executes the CUDA map kernel over a vector position.
     *
     * @tparam T - The type for input vector.
     *
     * @tparam O - The type for output vector.
     *
     * @tparam F - An operator implemented as a functor.
     *
     * @param input - The input vector.
     *
     * @param input_stride - The stride between consecutive values at input.
     *
     * @param output - The output vector.
     *
     * @param output_stride - The stride between consecutive values at output.
     *
     * @param size - Size of the input and output vectors.
     *
     * @param map_op - The functor operator with the unary map function.
     */
    template<typename T, typename O, typename F>
    __global__ void genericCudaMap1Kernel(const T *input,
                                          unsigned int input_stride,
                                          O *output,
                                          unsigned int output_stride,
                                          unsigned int size,
                                          F map_op) {
      unsigned int idx     = getArrayIndex(blockIdx, blockDim, threadIdx);
      unsigned int x_pos   = idx * input_stride;
      unsigned int y_pos   = idx * output_stride;
      if (idx < size) output[y_pos] = map_op(input[x_pos]);
      // else nothing to do, the data is out of the bounds
      __syncthreads();
    }

    /**
     * @brief Executes the CUDA map kernel over two vectors.
     *
     * @tparam T1 - The type for input1 vector.
     *
     * @tparam T2 - The type for input2 vector.
     *
     * @tparam O - The type for output vector.
     *
     * @tparam F - An operator implemented as a functor.
     *
     * @param input1 - The first input vector.
     *
     * @param input1_stride - The stride between consecutive values at input1.
     *
     * @param input2 - The second input vector.
     *
     * @param input2_stride - The stride between consecutive values at input2.
     *
     * @param output - The output vector.
     *
     * @param output_stride - The stride between consecutive values at output.
     *
     * @param size - Size of the input and output vectors.
     *
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
      unsigned int idx     = getArrayIndex(blockIdx, blockDim, threadIdx);
      unsigned int x1_pos  = idx * input1_stride;
      unsigned int x2_pos  = idx * input2_stride;
      unsigned int y_pos   = idx * output_stride;
      if (idx < size) output[y_pos] = map_op(input1[x1_pos], input2[x2_pos]);
      // else nothing to do, the data is out of the bounds
      __syncthreads();
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
     *
     * @tparam O - The type for output vector.
     *
     * @tparam F - An operator implemented as a functor.
     *
     * @param input - The input vector.
     *
     * @param input_stride - The stride between consecutive values at input.
     *
     * @param input_shift - The first valid position at input vector.
     *
     * @param output - The output vector.
     *
     * @param output_stride - The stride between consecutive values at output.
     *
     * @param output_shift - The first valid position at output vector.
     *
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
      dim3 block, grid;
      computeBlockAndGridSizesForArray(N, block, grid);
      genericCudaMap1Kernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
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
     *
     * @tparam T2 - The type for input2 vector.
     *
     * @tparam F - An operator implemented as a functor.
     *
     * @param input1 - The first input vector.
     *
     * @param input1_stride - The stride between consecutive values at input1.
     *
     * @param input1_shift - The first valid position at input1.
     *
     * @param input2 - The second input vector.
     *
     * @param input2_stride - The stride between consecutive values at input2.
     *
     * @param input2_shift - The first valid position at input2.
     *
     * @param output - The output vector.
     *
     * @param output_stride - The stride between consecutive values at output.
     *
     * @param output_shift - The first valid position at output vector.
     *
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
      computeBlockAndGridSizesForArray(N, block, grid);
      genericCudaMap2Kernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
        (input1_ptr, input1_stride, input2_ptr, input2_stride,
         output_ptr, output_stride, N, map_op);
    } // function genericCudaMap1Call
    
  } // namespace CUDA

} // namespace AprilMath

#endif // USE_CUDA

#endif // CUDA_KERNEL_TEMPLATES_H
