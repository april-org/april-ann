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
#include "maxmin.h"
#include "smart_ptr.h"
#include "unused_variable.h"

namespace AprilMath {
  
  namespace CUDA {

    /////////////////////////////// REDUCE ////////////////////////////////////    
    
    /**
     * @brief Executes the CUDA reduce kernel over a vector position.
     *
     * @tparam T - The type for input and output vectors.
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
     * @param reduction_top - Bound for the reduction, it must be the floor
     * power of two of size parameter.
     *
     * @param size - Size of the input vector.
     *
     * @param reduce_op - The functor operator with the reduction over two
     * values.
     *
     * @note The reduce_op functor must be associative, commutative and
     * idempotent.
     */
    template<typename T, typename F>
    __global__ void genericCudaReduceKernel(const T *input,
                                            unsigned int input_stride,
                                            T *output,
                                            unsigned int output_stride,
                                            unsigned int reduce_top,
                                            unsigned int size,
                                            F reduce_op) {
      unsigned int idx     = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int idx_2   = (reduce_top + idx);
      unsigned int x_pos   = idx   * input_stride;
      unsigned int y_pos   = idx   * output_stride;
      unsigned int x_pos_2 = idx_2 * input_stride;
      if (idx < size) {
        if (idx_2 < size) {
          // reduce when both indices are under size bound
          output[y_pos] = reduce_op(input[x_pos], input[x_pos_2]);
        }
        else {
          // copy as it is in the input vector
          output[y_pos] = input[x_pos];
        }
      } // if (idx < size)
      // else nothing to do, the data is out of the bounds
      __syncthreads();
    }
    
    template<typename T, typename F>
    __global__ void genericCudaReduceMinMaxKernel(const T *input,
                                                  unsigned int input_stride,
                                                  int32_t *which,
                                                  unsigned int which_stride,
                                                  T *output,
                                                  unsigned int output_stride,
                                                  unsigned int reduce_top,
                                                  unsigned int size,
                                                  F reduce_op) {
      unsigned int idx     = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int idx_2   = (reduce_top + idx);
      unsigned int x_pos   = idx   * input_stride;
      unsigned int y_pos   = idx   * output_stride;
      unsigned int y_pos_2 = idx   * which_stride;
      unsigned int x_pos_2 = idx_2 * input_stride;
      unsigned int w=0;
      if (idx < size) {
        if (idx_2 < size) {
          // reduce when both indices are under size bound
          output[y_pos] = reduce_op(input[x_pos], input[x_pos_2], w);
          // idx+1 and idx_2+1 because in Lua starts at 1
          if (w == 0) which[y_pos_2] = idx+1;
          else which[y_pos_2] = idx_2+1;
        }
        else {
          // copy as it is in the input vector
          output[y_pos]  = input[x_pos];
          // idx+1 because in Lua starts at 1
          which[y_pos_2] = idx+1;
        }
      } // if (idx < size)
      // else nothing to do, the data is out of the bounds
      __syncthreads();
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
     * @param dest - The output vector.
     *
     * @param dest_shift - The first valid position at dest vector.
     *
     * @param reduce_op - The functor operator with the reduce over two
     * values.
     *
     * @note The reduce_op functor must be associative, commutative and
     * idempotent.
     */
    template<typename T, typename F>
    void genericCudaReduceCall(unsigned int N,
                               const GPUMirroredMemoryBlock<T> *input,
                               unsigned int input_stride,
                               unsigned int input_shift,
                               T zero,
                               GPUMirroredMemoryBlock<T> *dest,
                               unsigned int dest_shift,
                               F reduce_op) {
      switch(N) {
      case 0u:
        dest->putValue(dest_shift, zero);
        break;
      case 1u:
        dest->copyFromBlock(dest_shift, input, input_shift, 1u);
        break;
      default:
        {
          unsigned int size = N;
          unsigned int reduce_top = AprilUtils::ceilingPowerOfTwo(N) >> 1;
          AprilUtils::SharedPtr< GPUMirroredMemoryBlock<O> >
            output(new GPUMirroredMemoryBlock<O>(N));
          const T *input_ptr = input->getGPUForRead() + input_shift;
          T *output_ptr = output->getGPUForWrite();
          dim3 block, grid;
          do {
            // Prepare reduce_top kernels, that is, size/2 kernels.
            computeBlockAndGridSizesForArray(reduce_top, block, grid);
            // Execute reduce_top kernels with size as upper bound.
            genericCudaReduceKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
              (input_ptr, input_stride, output_ptr, 1u,
               reduce_top, size,
               reduce_op);
            // After first iteration it uses only output_ptr.
            input_ptr       = output_ptr;
            size            = reduce_top;
            reduce_top >>= 1;
          } while(reduce_top > 0);
          // TODO: check return value (cudaError_t)
          cudaDeviceSynchronize();
          dest->copyFromBlock(dest_shift, output.get(), 0u, 1u);
        } // default:
      } // switch(N)
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
                                     F reduce_op) {
      switch(N) {
      case 0u:
        dest->putValue(dest_shift, zero);
        which->putValue(which_shift, -1);
        break;
      case 1u:
        dest->copyFromBlock(dest_shift, input, input_shift, 1u);
        which->putValue(which_shift, 0);
        break;
      default:
        {
          unsigned int size = N;
          unsigned int reduce_top = AprilUtils::ceilingPowerOfTwo(N) >> 1;
          AprilUtils::SharedPtr< GPUMirroredMemoryBlock<int32_t> >
            output_which(new GPUMirroredMemoryBlock<int32_t>(N));
          AprilUtils::SharedPtr< GPUMirroredMemoryBlock<T> >
            output(new GPUMirroredMemoryBlock<T>(N));
          const T *input_ptr = input->getGPUForRead() + input_shift;
          int32_t *output_which_ptr = output_which->getGPUForWrite();
          T *output_ptr = output->getGPUForWrite();
          dim3 block, grid;
          do {
            // Prepare reduce_top kernels, that is, size/2 kernels.
            computeBlockAndGridSizesForArray(reduce_top, block, grid);
            // Execute reduce_top kernels with size as upper bound.
            genericCudaReduceMinMaxKernel<<<grid, block, 0, GPUHelper::getCurrentStream()>>>
              (input_ptr, input_stride,
               output_which_ptr, 1u,
               output_ptr, 1u,
               reduce_top, size,
               reduce_op);
            // After first iteration it uses only output_ptr.
            input_ptr       = output_ptr;
            size            = reduce_top;
            reduce_top >>= 1;
          } while(reduce_top > 0);
          // TODO: check return value (cudaError_t)
          cudaDeviceSynchronize();
          dest->copyFromBlock(dest_shift, output.get(), 0u, 1u);
          which->copyFromBlock(which_shift, output_which.get(), 0u, 1u);
        } // default:
      } // switch(N)
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
      unsigned int idx =
        getArrayIndex(blockIdx.x * blockDim.x + threadIdx.x);
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
      unsigned int idx =
        getArrayIndex(blockIdx.x * blockDim.x + threadIdx.x);
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
        (input_ptr, input_stride, output_ptr, output_stride, size, map_op);
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
         output_ptr, output_stride, size, map_op);
    } // function genericCudaMap1Call
    
  } // namespace CUDA

} // namespace AprilMath

#endif // USE_CUDA

#endif // CUDA_KERNEL_TEMPLATES_H
