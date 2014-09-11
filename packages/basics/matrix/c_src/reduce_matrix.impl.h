/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
 *
 * The APRIL-ANN toolkit is free software; you can redistribute it and/or modify it
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
#ifndef REDUCE_MATRIX_IMPL_H
#define REDUCE_MATRIX_IMPL_H

// Must be defined here.
#include "matrix.h"

#include "april_assert.h"
#include "omp_utils.h"
#include "reduce_matrix.h"
#include "reduce_template.h"
#include "smart_ptr.h"

namespace AprilMath {

  namespace MatrixExt {

    template<typename T, typename OP>
    Basics::Matrix<T> * MatrixScalarReduceMinMaxOverDimension(const Basics::Matrix<T> *input,
                                                              int dim,
                                                              const OP &scalar_red_functor,
                                                              const T &zero,
                                                              Basics::Matrix<int32_t> *which,
                                                              Basics::Matrix<T> *dest,
                                                              bool set_dest_to_zero) {
      ScalarToSpanReduceMinMax<T,OP> span_functor(scalar_red_functor);
      return MatrixSpanReduceMinMaxOverDimension(input, dim, span_functor,
                                                 scalar_red_functor, zero,
                                                 which, dest, set_dest_to_zero);
    }

    template<typename T, typename O, typename OP1, typename OP2>
    void MatrixSpanReduceMinMax(const Basics::Matrix<T> *input,
                                const OP1 &inter_span_red_functor,
                                const OP2 &intra_span_red_functor,
                                const O &zero,
                                AprilMath::GPUMirroredMemoryBlock<int32_t> *which,
                                unsigned int which_raw_pos,
                                AprilMath::GPUMirroredMemoryBlock<O> *dest,
                                unsigned int dest_raw_pos,
                                bool set_dest_to_zero) {
      april_assert(input != 0);
      if (dest == 0) ERROR_EXIT(128, "Expected a non-NULL dest pointer\n");
      if (which == 0) ERROR_EXIT(128, "Expected a non-NULL which pointer\n");
      bool cuda_flag = input->getCudaFlag() || dest->getCudaFlag() ||
        which->getCudaFlag();

      // This specialization has a problem to compute the argmin or argmax when
      // matrices are in col_major

      // // Contiguous memory block
      // if (input->getIsContiguous()) {
      //   inter_span_red_functor(static_cast<unsigned int>(input->size()),
      //                          input->getRawDataAccess(), 1u,
      //                          static_cast<unsigned int>(input->getOffset()),
      //                          input->getCudaFlag(),
      //                          zero,
      //                          intra_span_red_functor,
      //                          which,
      //                          which_raw_pos,
      //                          dest,
      //                          dest_raw_pos,
      //                          set_dest_to_zero);
      // }
      
      // One dimension
      if (input->getNumDim() == 1) {
        inter_span_red_functor(static_cast<unsigned int>(input->size()),
                               input->getRawDataAccess(),
                               static_cast<unsigned int>(input->getStrideSize(0)),
                               static_cast<unsigned int>(input->getOffset()),
                               cuda_flag,
                               zero,
                               intra_span_red_functor,
                               which,
                               which_raw_pos,
                               dest,
                               dest_raw_pos,
                               set_dest_to_zero);
      }
      // This specialization has a problem to compute the argmin or argmax when
      // matrices are in col_major
      
      // General case
      else {
        ERROR_EXIT(128, "NOT IMPLEMENTED\n");
        /*
          typename Basics::Matrix<T>::span_iterator span_it(input);
          unsigned int size   = static_cast<unsigned int>(span_it.getSize());
          unsigned int stride = static_cast<unsigned int>(span_it.getStride());
          const int N = span_it.numberOfIterations();
          for (int i=0; i<N; ++i) {
          april_assert(span_it != input->end_span_iterator());
          inter_span_red_functor(size,
          input->getRawDataAccess(),
          stride,
          span_it.getOffset(),
          input->getCudaFlag(),
          zero,
          intra_span_red_functor,
          which,
          which_raw_pos,
          dest,
          dest_raw_pos,
          set_dest_to_zero);
          set_dest_to_zero = false; // only use it in the first iteration
          ++span_it;
          }
          april_assert(span_it == input->end_span_iterator());
        */
      }
    } // function MatrixSpanReduceMinMax

    template<typename T, typename OP1, typename OP2>
    Basics::Matrix<T> * MatrixSpanReduceMinMaxOverDimension(const Basics::Matrix<T> *input,
                                                            int dim,
                                                            const OP1 &inter_span_red_functor,
                                                            const OP2 &intra_span_red_functor,
                                                            const T &zero,
                                                            Basics::Matrix<int32_t> *which,
                                                            Basics::Matrix<T> *dest,
                                                            bool set_dest_to_zero) {
      april_assert(input != 0);
      bool cuda_flag = input->getCudaFlag() || (dest && dest->getCudaFlag()) ||
        (which && which->getCudaFlag());
      const int numDim      = input->getNumDim();
      const int *matrixSize = input->getDimPtr();
      AprilUtils::UniquePtr<int []> result_dims( new int[numDim] );
      /**** INPUT span ****/
      AprilUtils::UniquePtr<int []> span_order( new int[numDim] );
      span_order[0] = dim;
      int result_size=1;
      for (int i=0; i<dim; ++i) {
        span_order[numDim-i-1] = i;
        result_dims[i]  = matrixSize[i];
        result_size    *= result_dims[i];
      }
      result_dims[dim] = 1;
      for (int i=dim+1; i<numDim; ++i) {
        span_order[numDim-i] = i;
        result_dims[i]  = matrixSize[i];
        result_size    *= result_dims[i];
      }
      /******************************/
      Basics::Matrix<T> *result  = dest;
      Basics::Matrix<int32_t> *result2 = which;
      if (result == 0) {
        result = new Basics::Matrix<T>(numDim, result_dims.get(),
                                       input->getMajorOrder());
        set_dest_to_zero = true;
      }
      if (result2 == 0) {
        result2 = new Basics::Matrix<int32_t>(numDim,
                                              result_dims.get());
        set_dest_to_zero = true;
      }
#ifdef USE_CUDA
      result->setUseCuda(cuda_flag);
      result2->setUseCuda(cuda_flag);
#endif
      if (result->size()  != result_size ||
          result2->size() != result_size) {
        // else if (!result->sameDim(result_dims, numDim))
        ERROR_EXIT2(256, "Incorrect size at the given dest matrtix, "
                    "expected %d, found %d\n", result_size, result->size());
      }
      typename Basics::Matrix<T>::span_iterator span_it(input, span_order.get());
      typename Basics::Matrix<int32_t>::pos_iterator it2(result2);
      unsigned int span_size   = static_cast<unsigned int>(span_it.getSize());
      unsigned int span_stride = static_cast<unsigned int>(span_it.getStride());
      april_assert(span_it.numberOfIterations() == result->size());
      // traverse in row major order
      for (typename Basics::Matrix<T>::pos_iterator it(result);
           !it.isEnd(); ++it, ++it2, ++span_it) {
        april_assert(span_it != input->end_span_iterator());
        april_assert(!it2.isEnd());
        inter_span_red_functor(span_size,
                               input->getRawDataAccess(),
                               span_stride,
                               static_cast<unsigned int>(span_it.getOffset()),
                               cuda_flag,
                               zero,
                               intra_span_red_functor,
                               result2->getRawDataAccess(),
                               it2.getRawPos(),
                               result->getRawDataAccess(),
                               it.getRawPos(),
                               set_dest_to_zero);
      }
      april_assert(span_it == input->end_span_iterator());
      april_assert(it2.isEnd());
      return result;
    }

    template<typename T, typename O, typename OP1, typename OP2>
    Basics::Matrix<O> * MatrixScalarReduce1OverDimension(const Basics::Matrix<T> *input,
                                                         int dim,
                                                         const OP1 &scalar_red_functor,
                                                         const OP2 &partials_red_functor,
                                                         const O &zero,
                                                         Basics::Matrix<O> *dest,
                                                         bool set_dest_to_zero) {
      ScalarToSpanReduce1<T,O,OP1> span_functor(scalar_red_functor);
      return MatrixSpanReduce1OverDimension(input, dim, span_functor,
                                            partials_red_functor, zero,
                                            dest,
                                            set_dest_to_zero);
    }

    template<typename T, typename O, typename OP1, typename OP2>
    Basics::Matrix<O> * MatrixSpanReduce1OverDimension(const Basics::Matrix<T> *input,
                                                       int dim,
                                                       const OP1 &inter_span_red_functor,
                                                       const OP2 &intra_span_red_functor,
                                                       const O &zero,
                                                       Basics::Matrix<O> *dest,
                                                       bool set_dest_to_zero) {
      april_assert(input != 0);
      bool cuda_flag = input->getCudaFlag() || (dest && dest->getCudaFlag());
      const int numDim      = input->getNumDim();
      const int *matrixSize = input->getDimPtr();
      AprilUtils::UniquePtr<int []> result_dims( new int[numDim] );
      /**** INPUT span ****/
      AprilUtils::UniquePtr<int []> span_order( new int[numDim] );
      span_order[0] = dim;
      int result_size=1;
      for (int i=0; i<dim; ++i) {
        span_order[numDim-i-1] = i;
        result_dims[i]  = matrixSize[i];
        result_size    *= result_dims[i];
      }
      result_dims[dim] = 1;
      for (int i=dim+1; i<numDim; ++i) {
        span_order[numDim-i] = i;
        result_dims[i]  = matrixSize[i];
        result_size    *= result_dims[i];
      }
      /******************************/
      Basics::Matrix<O> *result = dest;
      if (result == 0) {
        result = new Basics::Matrix<O>(numDim, result_dims.get(),
                                       input->getMajorOrder());
        set_dest_to_zero = true;
      }
      else if (result->size() != result_size) {
        // else if (!result->sameDim(result_dims, numDim))
        ERROR_EXIT2(256, "Incorrect size at the given dest matrix, "
                    "expected %d, found %d\n", result_size, result->size());
#ifdef USE_CUDA
        result->setUseCuda(cuda_flag);
#endif
      }
      typename Basics::Matrix<T>::span_iterator span_it(input, span_order.get());
      unsigned int span_size   = static_cast<unsigned int>(span_it.getSize());
      unsigned int span_stride = static_cast<unsigned int>(span_it.getStride());
      april_assert(span_it.numberOfIterations() == result->size());
      // traverse in row major order
      for (typename Basics::Matrix<O>::pos_iterator it(result);
           !it.isEnd(); ++it, ++span_it) {
        april_assert(span_it != input->end_span_iterator());
        inter_span_red_functor(span_size,
                               input->getRawDataAccess(),
                               span_stride,
                               static_cast<unsigned int>(span_it.getOffset()),
                               cuda_flag,
                               zero, intra_span_red_functor,
                               result->getRawDataAccess(), it.getRawPos(),
                               set_dest_to_zero);
      }
      april_assert(span_it == input->end_span_iterator());
      return result;
    }
    
    template<typename T, typename OP>
    void MatrixScalarSumReduce1(const Basics::Matrix<T> *input,
                                const OP &scalar_red_functor,
                                AprilMath::GPUMirroredMemoryBlock<T> *dest,
                                unsigned int dest_raw_pos,
                                bool set_dest_to_zero,
                                int N_th, unsigned int SIZE_th) {
      ScalarToSpanReduce1<T,T,OP> span_functor(scalar_red_functor);
      MatrixSpanSumReduce1(input, span_functor, dest, dest_raw_pos,
                           N_th, SIZE_th,
                           set_dest_to_zero);
    }
    
    template<typename T, typename O, typename OP1, typename OP2>
    O MatrixSpanReduce1(const Basics::Matrix<T> *input,
                        const OP1 &inter_span_red_functor,
                        const OP2 &intra_span_red_functor,
                        const O &zero) {
      AprilUtils::SharedPtr< AprilMath::GPUMirroredMemoryBlock<O> > dest( new AprilMath::GPUMirroredMemoryBlock<O>(1u) );
      MatrixSpanReduce1(input, inter_span_red_functor,
                        intra_span_red_functor, zero,
                        dest.get(), 0u, true);
      return dest->get(0);
    }

    template<typename T, typename O, typename OP1, typename OP2>
    void MatrixSpanReduce1(const Basics::Matrix<T> *input,
                           const OP1 &inter_span_red_functor,
                           const OP2 &intra_span_red_functor,
                           const O &zero,
                           AprilMath::GPUMirroredMemoryBlock<O> *dest,
                           unsigned int dest_raw_pos,
                           bool set_dest_to_zero) {
      april_assert(input != 0);
      if (dest == 0) ERROR_EXIT(128, "Expected a non-NULL dest pointer\n");
      bool cuda_flag = input->getCudaFlag() || dest->getCudaFlag();
      // Contiguous memory block or one dimension.
      if (input->getIsContiguous() || input->getNumDim() == 1) {
        unsigned int size = static_cast<unsigned int>(input->size());
        unsigned int input_offset = static_cast<unsigned int>(input->getOffset());
        unsigned int input_stride;
        if (input->getIsContiguous()) {
          // Contiguous.
          input_stride = 1u;
        }
        else {
          // One dimension.
          input_stride = static_cast<unsigned int>(input->getStrideSize(0));
        }
        inter_span_red_functor(size,
                               input->getRawDataAccess(),
                               input_stride, input_offset,
                               cuda_flag,
                               zero, intra_span_red_functor,
                               dest, dest_raw_pos,
                               set_dest_to_zero);
      }
      // General case
      else {
        typename Basics::Matrix<T>::span_iterator span_it(input);
        unsigned int size   = static_cast<unsigned int>(span_it.getSize());
        unsigned int stride = static_cast<unsigned int>(span_it.getStride());
        const int N = span_it.numberOfIterations();
        for (int i=0; i<N; ++i) {
          april_assert(span_it != input->end_span_iterator());
          inter_span_red_functor(size,
                                 input->getRawDataAccess(),
                                 stride,
                                 span_it.getOffset(),
                                 cuda_flag,
                                 zero, intra_span_red_functor,
                                 dest, dest_raw_pos,
                                 set_dest_to_zero);
          set_dest_to_zero = false; // use only in the first iteration
          ++span_it;
        }
        april_assert(span_it == input->end_span_iterator());
      }
    } // function MatrixSpanReduce1

    template<typename T, typename O, typename OP1, typename OP2>
    O MatrixScalarReduce1(const Basics::Matrix<T> *input,
                          const OP1 &scalar_red_functor,
                          const OP2 &partials_red_functor,
                          const O &zero) {
      AprilUtils::SharedPtr< AprilMath::GPUMirroredMemoryBlock<O> > dest( new AprilMath::GPUMirroredMemoryBlock<O>(1) );
      MatrixScalarReduce1(input, scalar_red_functor, partials_red_functor,
                          zero, dest.get(), 0u, true);
      return dest->get(0);
    }

    template<typename T, typename O, typename OP1, typename OP2>
    void MatrixScalarReduce1(const Basics::Matrix<T> *input,
                             const OP1 &scalar_red_functor,
                             const OP2 &partials_red_functor,
                             const O &zero,
                             AprilMath::GPUMirroredMemoryBlock<O> *dest,
                             unsigned int dest_raw_pos,
                             bool set_dest_to_zero) {
      ScalarToSpanReduce1<T,O,OP1> span_functor(scalar_red_functor);
      MatrixSpanReduce1(input, span_functor, partials_red_functor,
                        zero, dest, dest_raw_pos, set_dest_to_zero);
    }
    

    template<typename T, typename OP>
    T MatrixSpanSumReduce1(const Basics::Matrix<T> *input,
                           const OP &inter_span_red_functor) {
      AprilUtils::SharedPtr< AprilMath::GPUMirroredMemoryBlock<T> > dest( new AprilMath::GPUMirroredMemoryBlock<T>(1) );
      MatrixSpanSumReduce1(input, inter_span_red_functor, dest.get(), 0u, true);
      return dest->get(0);
    }
    
    template<typename T, typename OP>
    void MatrixSpanSumReduce1(const Basics::Matrix<T> *input,
                              const OP &inter_span_red_functor,
                              AprilMath::GPUMirroredMemoryBlock<T> *dest,
                              unsigned int dest_raw_pos,
                              bool set_dest_to_zero,
                              int N_th,
                              unsigned int SIZE_th) {
      april_assert(input != 0);
      if (dest == 0) ERROR_EXIT(128, "Expected a non-NULL dest pointer\n");
#ifdef NO_OMP
      UNUSED_VARIABLE(N_th);
      UNUSED_VARIABLE(SIZE_th);
#endif
      bool cuda_flag = input->getCudaFlag() || dest->getCudaFlag();
      // Contiguous memory block or one dimension.
      if (input->getIsContiguous() || input->getNumDim() == 1) {
        unsigned int size = static_cast<unsigned int>(input->size());
        unsigned int input_offset = static_cast<unsigned int>(input->getOffset());
        unsigned int input_stride;
        if (input->getIsContiguous()) {
          // Contiguous.
          input_stride = 1u;
        }
        else {
          // One dimension.
          input_stride = static_cast<unsigned int>(input->getStrideSize(0));
        }
        inter_span_red_functor(size,
                               input->getRawDataAccess(),
                               input_stride, input_offset,
                               cuda_flag,
                               T(0.0f), AprilMath::Functors::r_add<T,T>(),
                               dest, dest_raw_pos,
                               set_dest_to_zero);
      }
      // General case
      else {
        typename Basics::Matrix<T>::span_iterator span_it(input);
        unsigned int size   = static_cast<unsigned int>(span_it.getSize());
        unsigned int stride = static_cast<unsigned int>(span_it.getStride());
        const int N = span_it.numberOfIterations();
#ifndef NO_OMP
        // this if controls the execution using OMP only when the number of threads
        // is more than 1 and the iterator size is big enough
        if (OMPUtils::get_num_threads() > 1 && N > N_th && size > SIZE_th) {
          T result;
          if (set_dest_to_zero) result = T(0.0f);
          else dest->getValue(dest_raw_pos, result);
          GPUMirroredMemoryBlock<T> aux(1);
          T partial;
#ifdef USE_CUDA
          // Forces execution of memory copy from GPU to PPAL or viceversa (if
          // needed), avoiding race conditions on the following.
          input->getRawDataAccess()->forceUpdate(cuda_flag);
#endif
#pragma omp parallel for reduction(+:result) firstprivate(span_it)
          for (int i=0; i<N; ++i) {
            span_it.setAtIteration(i);
            inter_span_red_functor(size,
                                   input->getRawDataAccess(),
                                   stride,
                                   span_it.getOffset(),
                                   cuda_flag,
                                   T(0.0f), AprilMath::Functors::r_add<T,T>(),
                                   &aux, 0, true);
            aux.getValue(0, partial);
            result += partial;
          }
          dest->putValue(dest_raw_pos, result);
        }
        else {
#endif
          for (int i=0; i<N; ++i) {
            april_assert(span_it != input->end_span_iterator());
            inter_span_red_functor(size,
                                   input->getRawDataAccess(),
                                   stride,
                                   span_it.getOffset(),
                                   cuda_flag,
                                   T(0.0f), AprilMath::Functors::r_add<T,T>(),
                                   dest, dest_raw_pos,
                                   set_dest_to_zero);
            set_dest_to_zero = false; // use only in the first iteration
            ++span_it;
          }
          april_assert(span_it == input->end_span_iterator());
#ifndef NO_OMP
        }
#endif
      } // General case
    } // function MatrixSpanSumReduce1

    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    O MatrixSpanReduce2(const Basics::Matrix<T1> *input1,
                        const Basics::Matrix<T2> *input2,
                        const OP1 &inter_span_red_functor,
                        const OP2 &intra_span_red_functor,
                        const O &zero) {
      AprilUtils::SharedPtr< AprilMath::GPUMirroredMemoryBlock<O> > dest( new AprilMath::GPUMirroredMemoryBlock<O>(1) );
      MatrixSpanReduce2(input1, input2, inter_span_red_functor,
                        intra_span_red_functor, zero, dest.get(), 0u, true);
      return dest->get(0);
    }

    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    void MatrixSpanReduce2(const Basics::Matrix<T1> *input1,
                           const Basics::Matrix<T2> *input2,
                           const OP1 &inter_span_red_functor,
                           const OP2 &intra_span_red_functor,
                           const O &zero,
                           AprilMath::GPUMirroredMemoryBlock<O> *dest,
                           unsigned int dest_raw_pos,
                           bool set_dest_to_zero) {
      april_assert(input1 != 0 && input2 != 0);
      if (dest == 0) ERROR_EXIT(128, "Expected a non-NULL dest pointer\n");
      if (input1->size() != input2->size()) {
        ERROR_EXIT(128, "Incompatible matrix sizes\n");
      }
      bool cuda_flag = input1->getCudaFlag() || input2->getCudaFlag() ||
        dest->getCudaFlag();
      // Contiguous memory block or one dimension.
      if ( (input1->getIsContiguous() || input1->getNumDim() == 1) &&
           (input2->getIsContiguous() || input2->getNumDim() == 1) ) {
        unsigned int size = static_cast<unsigned int>(input1->size());
        unsigned int input1_offset = static_cast<unsigned int>(input1->getOffset());
        unsigned int input2_offset = static_cast<unsigned int>(input2->getOffset());
        unsigned int input1_stride, input2_stride;
        if (input1->getIsContiguous()) {
          // Contiguous.
          input1_stride = 1u;
        }
        else {
          // One dimension.
          input1_stride = static_cast<unsigned int>(input1->getStrideSize(0));
        }
        if (input2->getIsContiguous()) {
          // Contiguous.
          input2_stride = 1u;
        }
        else {
          // One dimension.
          input2_stride = static_cast<unsigned int>(input2->getStrideSize(0));
        }
        inter_span_red_functor(size,
                               input1->getRawDataAccess(),
                               input1_stride, input1_offset,
                               input2->getRawDataAccess(),
                               input2_stride, input2_offset,
                               cuda_flag,
                               zero, intra_span_red_functor,
                               dest, dest_raw_pos,
                               set_dest_to_zero);
      }
      // General case
      else {
        typename Basics::Matrix<T1>::span_iterator input1_span_it(input1);
        typename Basics::Matrix<T2>::span_iterator input2_span_it(input2);
        const int N = input1_span_it.numberOfIterations();
        april_assert(N == input2_span_it.numberOfIterations());
        const unsigned int size          = static_cast<unsigned int>(input1_span_it.getSize());
        const unsigned int input1_stride = static_cast<unsigned int>(input1_span_it.getStride());
        const unsigned int input2_stride = static_cast<unsigned int>(input2_span_it.getStride());
        april_assert(size == static_cast<unsigned int>(input2_span_it.getSize()));
        for (int i=0; i<N; ++i) {
          april_assert(input1_span_it != input1->end_span_iterator());
          april_assert(input2_span_it != input2->end_span_iterator());
          inter_span_red_functor(size,
                                 input1->getRawDataAccess(),
                                 input1_stride,
                                 input1_span_it.getOffset(),
                                 input2->getRawDataAccess(),
                                 input2_stride,
                                 input2_span_it.getOffset(),
                                 cuda_flag,
                                 zero, intra_span_red_functor,
                                 dest, dest_raw_pos,
                                 set_dest_to_zero);
          set_dest_to_zero = false; // use only in the first iteration
          ++input1_span_it;
          ++input2_span_it;
        }
        april_assert(input1_span_it == input1->end_span_iterator());
        april_assert(input2_span_it == input2->end_span_iterator());
      }
    } // function MatrixSpanReduce2

    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    void MatrixScalarReduce2(const Basics::Matrix<T1> *input1,
                             const Basics::Matrix<T2> *input2,
                             const OP1 &scalar_red_functor,
                             const OP2 &partials_red_functor,
                             const O &zero,
                             AprilMath::GPUMirroredMemoryBlock<O> *dest,
                             unsigned int dest_raw_pos,
                             bool set_dest_to_zero) {
      ScalarToSpanReduce2<T1,T2,O,OP1> span_functor(scalar_red_functor);
      MatrixSpanReduce2(input1, input2, span_functor, partials_red_functor,
                        zero, dest, dest_raw_pos, set_dest_to_zero);      
    }
    
    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    O MatrixScalarReduce2(const Basics::Matrix<T1> *input1,
                          const Basics::Matrix<T2> *input2,
                          const OP1 &scalar_red_functor,
                          const OP2 &partials_red_functor,
                          const O &zero) {
      AprilUtils::SharedPtr< AprilMath::GPUMirroredMemoryBlock<O> > dest( new AprilMath::GPUMirroredMemoryBlock<O>(1) );
      MatrixScalarReduce2(input1, input2, scalar_red_functor,
                          partials_red_functor, zero, dest.get(), 0u, true);
      return dest->get(0);      
    }
    
    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    Basics::Matrix<O> * MatrixScalarReduce2OverDimension(const Basics::Matrix<T1> *input1,
                                                         const Basics::Matrix<T2> *input2,
                                                         int dim,
                                                         const OP1 &scalar_red_functor,
                                                         const OP2 &partials_red_functor,
                                                         const O &zero,
                                                         Basics::Matrix<O> *dest,
                                                         bool set_dest_to_zero) {
    ScalarToSpanReduce2<T1,T2,O,OP1> span_functor(scalar_red_functor);
    return MatrixSpanReduce2OverDimension(input1, input2, dim, span_functor,
                                          partials_red_functor, zero,
                                          dest,
                                          set_dest_to_zero);
  }
    
    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    Basics::Matrix<O> * MatrixSpanReduce2OverDimension(const Basics::Matrix<T1> *input1,
                                                       const Basics::Matrix<T2> *input2,
                                                       int dim,
                                                       const OP1 &inter_span_red_functor,
                                                       const OP2 &intra_span_red_functor,
                                                       const O &zero,
                                                       Basics::Matrix<O> *dest,
                                                       bool set_dest_to_zero) {
      april_assert(input1 != 0 && input2 != 0);
      if (input1->size() != input2->size()) {
        ERROR_EXIT(128, "Incompatible matrix sizes\n");
      }
      bool cuda_flag = input1->getCudaFlag() || input2->getCudaFlag() ||
        (dest && dest->getCudaFlag());
      const int numDim      = input1->getNumDim();
      const int *matrixSize = input1->getDimPtr();
      AprilUtils::UniquePtr<int []> result_dims( new int[numDim] );
      /**** INPUT span ****/
      AprilUtils::UniquePtr<int []> span_order( new int[numDim] );
      span_order[0] = dim;
      int result_size=1;
      for (int i=0; i<dim; ++i) {
        span_order[numDim-i-1] = i;
        result_dims[i]  = matrixSize[i];
        result_size    *= result_dims[i];
      }
      result_dims[dim] = 1;
      for (int i=dim+1; i<numDim; ++i) {
        span_order[numDim-i] = i;
        result_dims[i]  = matrixSize[i];
        result_size    *= result_dims[i];
      }
      /******************************/
      Basics::Matrix<O> *result = dest;
      if (result == 0) {
        result = new Basics::Matrix<O>(numDim, result_dims.get(),
                                       input1->getMajorOrder());
        set_dest_to_zero = true;
      }
      else if (result->size() != result_size) {
        // else if (!result->sameDim(result_dims, numDim))
        ERROR_EXIT2(256, "Incorrect size at the given dest matrix, "
                    "expected %d, found %d\n", result_size, result->size());
#ifdef USE_CUDA
        result->setUseCuda(cuda_flag);
#endif
      }
      typename Basics::Matrix<T1>::span_iterator span1_it(input1, span_order.get());
      typename Basics::Matrix<T2>::span_iterator span2_it(input2, span_order.get());
      unsigned int span_size   = static_cast<unsigned int>(span1_it.getSize());
      unsigned int span1_stride = static_cast<unsigned int>(span1_it.getStride());
      unsigned int span2_stride = static_cast<unsigned int>(span2_it.getStride());
      april_assert(span1_it.numberOfIterations() == result->size());
      april_assert(span2_it.getSize() == span1_it.getSize());
      april_assert(span2_it.numberOfIterations() == span1_it.numberOfIterations());
      // traverse in row major order
      for (typename Basics::Matrix<O>::pos_iterator it(result);
           !it.isEnd(); ++it, ++span1_it, ++span2_it) {
        april_assert(span1_it != input1->end_span_iterator());
        april_assert(span2_it != input2->end_span_iterator());
        inter_span_red_functor(span_size,
                               input1->getRawDataAccess(),
                               span1_stride,
                               static_cast<unsigned int>(span1_it.getOffset()),
                               input2->getRawDataAccess(),
                               span2_stride,
                               static_cast<unsigned int>(span2_it.getOffset()),
                               cuda_flag,
                               zero, intra_span_red_functor,
                               result->getRawDataAccess(), it.getRawPos(),
                               set_dest_to_zero);
      }
      april_assert(span1_it == input1->end_span_iterator());
      april_assert(span2_it == input2->end_span_iterator());
      return result;
    }
    
  } // namespace MatrixExt
  
} // namespace AprilMath

#endif // REDUCE_MATRIX_IMPL_H
