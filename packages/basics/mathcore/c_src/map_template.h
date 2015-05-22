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

  template<typename T1, typename T2, typename T3,
           typename O, typename F>
  void genericMap3Call(unsigned int N,
                       const GPUMirroredMemoryBlock<T1> *input1,
                       unsigned int input1_stride,
                       unsigned int input1_shift,
                       const GPUMirroredMemoryBlock<T2> *input2,
                       unsigned int input2_stride,
                       unsigned int input2_shift,
                       const GPUMirroredMemoryBlock<T3> *input3,
                       unsigned int input3_stride,
                       unsigned int input3_shift,
                       GPUMirroredMemoryBlock<O> *output,
                       unsigned int output_stride,
                       unsigned int output_shift,
                       bool use_gpu,
                       F map_op) {
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif

    // FIXME: Review why this code is not compiling with nvcc (CUDA) :S
    /*
/home/pako/programas/april-ann/build_release_cuda_and_mkl/packages/basics/mathcore/include/map_template.h(126):
            error: no instance of function template
            "AprilMath::CUDA::genericCudaMap3Call" matches the argument list
            argument types are: (unsigned int, const
            AprilMath::GPUMirroredMemoryBlock<AprilMath::ComplexF> *, unsigned
            int, unsigned int, const
            AprilMath::GPUMirroredMemoryBlock<__nv_bool> *, unsigned int,
            unsigned int, const
            AprilMath::GPUMirroredMemoryBlock<AprilMath::ComplexF> *, unsigned
            int, unsigned int,
            AprilMath::GPUMirroredMemoryBlock<AprilMath::ComplexF> *, unsigned
            int, unsigned int,
            AprilMath::MatrixExt::Operations::maskedCopyOp<AprilMath::ComplexF>)

          detected during:

            instantiation of "void AprilMath::genericMap3Call(unsigned int,
const AprilMath::GPUMirroredMemoryBlock<T1> *, unsigned int, unsigned int, const
AprilMath::GPUMirroredMemoryBlock<T2> *, unsigned int, unsigned int, const
AprilMath::GPUMirroredMemoryBlock<T3> *, unsigned int, unsigned int,
AprilMath::GPUMirroredMemoryBlock<O> *, unsigned int, unsigned int, __nv_bool,
F) [with T1=AprilMath::ComplexF, T2=__nv_bool, T3=AprilMath::ComplexF,
O=AprilMath::ComplexF,
F=AprilMath::MatrixExt::Operations::maskedCopyOp<AprilMath::ComplexF>]" (214):
here

            instantiation of "void AprilMath::ScalarToSpanMap3<T1, T2, T3, O,
OP>::operator()(unsigned int, const AprilMath::GPUMirroredMemoryBlock<T1> *,
unsigned int, unsigned int, const AprilMath::GPUMirroredMemoryBlock<T2> *,
unsigned int, unsigned int, const AprilMath::GPUMirroredMemoryBlock<T3> *,
unsigned int, unsigned int, AprilMath::GPUMirroredMemoryBlock<O> *, unsigned
int, unsigned int, __nv_bool) const [with T1=AprilMath::ComplexF, T2=__nv_bool,
T3=AprilMath::ComplexF, O=AprilMath::ComplexF,
OP=AprilMath::MatrixExt::Operations::maskedCopyOp<AprilMath::ComplexF>]"
c_src/map_matrix.impl.h(371): here

            instantiation of "Basics::Matrix<O>
*AprilMath::MatrixExt::MatrixSpanMap3(const Basics::Matrix<T1> *, const
Basics::Matrix<T2> *, const Basics::Matrix<T3> *, const OP &, Basics::Matrix<O>
*, int, unsigned int) [with T1=AprilMath::ComplexF, T2=__nv_bool,
T3=AprilMath::ComplexF, O=AprilMath::ComplexF,
OP=AprilMath::ScalarToSpanMap3<AprilMath::ComplexF, __nv_bool,
AprilMath::ComplexF, AprilMath::ComplexF,
AprilMath::MatrixExt::Operations::maskedCopyOp<AprilMath::ComplexF>>]"
c_src/map_matrix.impl.h(114): here

            instantiation of "Basics::Matrix<O>
*AprilMath::MatrixExt::MatrixScalarMap3(const Basics::Matrix<T1> *, const
Basics::Matrix<T2> *, const Basics::Matrix<T3> *, const OP &, Basics::Matrix<O>
*, int, unsigned int) [with T1=AprilMath::ComplexF, T2=__nv_bool,
T3=AprilMath::ComplexF, O=AprilMath::ComplexF,
OP=AprilMath::MatrixExt::Operations::maskedCopyOp<AprilMath::ComplexF>]"
c_src/matrix_ext_operations.cu(406): here

            instantiation of "Basics::Matrix<T>
*AprilMath::MatrixExt::Operations::matMaskedCopy(Basics::Matrix<T> *, const
Basics::Matrix<__nv_bool> *, const Basics::Matrix<T> *, Basics::Matrix<T> *)
[with T=AprilMath::ComplexF]" c_src/matrix_ext_operations.cu(515): here
     */
    // #ifdef USE_CUDA
    //     if (use_gpu) {
    //       CUDA::genericCudaMap3Call(N,
    //                                 input1, input1_stride, input1_shift,
    //                                 input2, input2_stride, input2_shift,
    //                                 input3, input3_stride, input3_shift,
    //                                 output, output_stride, output_shift,
    //                                 map_op);
    //     }
    //     else {
    // #endif
      const T1 *input1_mem = input1->getPPALForRead() + input1_shift;
      const T2 *input2_mem = input2->getPPALForRead() + input2_shift;
      const T3 *input3_mem = input3->getPPALForRead() + input3_shift;
      O *output_mem = output->getPPALForWrite() + output_shift;
      for (unsigned int i=0; i<N; ++i,
             output_mem+=output_stride,
             input1_mem+=input1_stride,
             input2_mem+=input2_stride,
             input3_mem+=input3_stride) {
        *output_mem = map_op(*input1_mem, *input2_mem, *input3_mem);
      }
      // #ifdef USE_CUDA
      //     }
      // #endif
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

  template<typename T1, typename T2, typename T3,
           typename O, typename OP>
  struct ScalarToSpanMap3 {
    const OP functor;
    ScalarToSpanMap3(const OP &functor) : functor(functor) { }
    void operator()(unsigned int N,
                    const GPUMirroredMemoryBlock<T1> *input1,
                    unsigned int input1_stride,
                    unsigned int input1_shift,
                    const GPUMirroredMemoryBlock<T2> *input2,
                    unsigned int input2_stride,
                    unsigned int input2_shift,
                    const GPUMirroredMemoryBlock<T3> *input3,
                    unsigned int input3_stride,
                    unsigned int input3_shift,
                    GPUMirroredMemoryBlock<O> *output,
                    unsigned int output_stride,
                    unsigned int output_shift,
                    bool use_cuda) const {
      genericMap3Call(N, input1, input1_stride, input1_shift,
                      input2, input2_stride, input2_shift,
                      input3, input3_stride, input3_shift,
                      output, output_stride, output_shift,
                      use_cuda, functor);
    }
  };
  
} // namespace AprilMath

#endif // MAP_TEMPLATE_H
