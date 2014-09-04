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

namespace AprilMath {

  namespace MatrixExt {

    template<typename T, typename OP>
    Basics::Matrix<T> * MatrixScalarReduceMinMaxOverDimension(Basics::Matrix<T> *input,
                                                              int dim,
                                                              const OP &scalar_red_functor,
                                                              const T &zero,
                                                              Basics::Matrix<int32_t> *which,
                                                              Basics::Matrix<T> *dest) {
      ScalarToSpanReduceMinMax<T,OP> span_functor((const OP &)scalar_red_functor);
      return MatrixSpanReduceMinMaxOverDimension(input, dim, span_functor,
                                                 (const OP &)scalar_red_functor, zero,
                                                 which, dest);
    }

    template<typename T, typename O, typename OP1, typename OP2>
    O MatrixSpanReduceMinMax(const Basics::Matrix<T> *input,
                             const OP1 &inter_span_red_functor,
                             const OP2 &intra_span_red_functor,
                             const O &zero,
                             Basics::Matrix<int32_t> *which,
                             unsigned int which_raw_pos,
                             Basics::Matrix<O> *dest,
                             unsigned int dest_raw_pos) {
      O result = zero;
      // Contiguous memory block
      if (input->getIsContiguous()) {
        result = inter_span_red_functor(static_cast<unsigned int>(input->size()),
                                        input->getRawDataAccess(), 1u,
                                        static_cast<unsigned int>(input->getOffset()),
                                        input->getCudaFlag(),
                                        zero,
                                        which->getRawDataAccess(),
                                        which_raw_pos,
                                        dest->getRawDataAccess(),
                                        dest_raw_pos);
      }
      // One dimension
      else if (input->getNumDim() == 1) {
        result = inter_span_red_functor(static_cast<unsigned int>(input->size()),
                                        input->getRawDataAccess(),
                                        static_cast<unsigned int>(input->getStrideSize(0)),
                                        static_cast<unsigned int>(input->getOffset()),
                                        input->getCudaFlag(),
                                        zero,
                                        which->getRawDataAccess(),
                                        which_raw_pos,
                                        dest->getRawDataAccess(),
                                        dest_raw_pos);
      }
      // General case
      else {
        typename Basics::Matrix<T>::span_iterator span_it(input);
        unsigned int size   = static_cast<unsigned int>(span_it.getSize());
        unsigned int stride = static_cast<unsigned int>(span_it.getStride());
        const int N = span_it.numberOfIterations();
        int32_t which_result=0;
        for (int i=0; i<N; ++i) {
          april_assert(span_it != input->end_span_iterator());
          O temp = inter_span_red_functor(size,
                                          input->getRawDataAccess(),
                                          stride,
                                          span_it.getOffset(),
                                          input->getCudaFlag(),
                                          zero,
                                          which->getRawDataAccess(),
                                          which_raw_pos,
                                          dest->getRawDataAccess(),
                                          dest_raw_pos);
          unsigned int aux=0;
          result = intra_span_red_functor(result, temp, aux);
          if (which != 0 && aux == 1) {
            which->getRawDataAccess()->getValue(which_raw_pos, which_result);
          }
          ++span_it;
        }
        april_assert(span_it == input->end_span_iterator());
        if (which != 0) {
          which->getRawDataAccess()->putValue(which_raw_pos, which_result);
        }
      }
      return result;
    } // function MatrixSpanReduceMinMax

    template<typename T, typename OP1, typename OP2>
    Basics::Matrix<T> * MatrixSpanReduceMinMaxOverDimension(Basics::Matrix<T> *input,
                                                            int dim,
                                                            const OP1 &inter_span_red_functor,
                                                            const OP2 &intra_span_red_functor,
                                                            const T &zero,
                                                            Basics::Matrix<int32_t> *which,
                                                            Basics::Matrix<T> *dest) {
      const int numDim      = input->getNumDim();
      const int *matrixSize = input->getDimPtr();
      AprilUtils::UniquePtr<int []> result_dims( new int[numDim] );
      /**** INPUT sliding window ****/
      AprilUtils::UniquePtr<int []> input_w_size( new int[numDim] );
      AprilUtils::UniquePtr<int []> input_w_num_steps( new int[numDim] );
      int result_size=1;
      for (int i=0; i<dim; ++i) {
        input_w_size[i] = 1;
        result_dims[i] = input_w_num_steps[i] = matrixSize[i];
        result_size *= result_dims[i];
      }
      result_dims[dim] = 1;
      input_w_size[dim] = matrixSize[dim];
      input_w_num_steps[dim] = 1;
      for (int i=dim+1; i<numDim; ++i) {
        input_w_size[i] = 1;
        result_dims[i] = input_w_num_steps[i] = matrixSize[i];
        result_size *= result_dims[i];
      }
      typename Basics::Matrix<T>::sliding_window input_w(input,input_w_size.get(),
                                                         0,0,
                                                         input_w_num_steps.get(),0);
      AprilUtils::SharedPtr< Basics::Matrix<T> > slice( input_w.getMatrix() );
      /******************************/
      Basics::Matrix<T> *result  = dest;
      Basics::Matrix<int32_t> *result2 = which;
      if (result == 0) result = new Basics::Matrix<T>(numDim, result_dims.get(),
                                                      input->getMajorOrder());
      if (result2 == 0) result2 = new Basics::Matrix<int32_t>(numDim,
                                                              result_dims.get());
      else if (result->size()  != result_size ||
               result2->size() != result_size) {
        // else if (!result->sameDim(result_dims, numDim))
        ERROR_EXIT2(256, "Incorrect size at the given dest matrtix, "
                    "expected %d, found %d\n", result_size, result->size());
      }
      // Forces to mark PPAL memory as updated, needed to avoid copying the
      // memory block because of the iterator.
      T *force_ppal_bit = result->getRawDataAccess()->getPPALForWrite();
      UNUSED_VARIABLE(force_ppal_bit);
      int32_t *force_ppal_bit2 = result2->getRawDataAccess()->getPPALForWrite();
      UNUSED_VARIABLE(force_ppal_bit2);
      typename Basics::Matrix<int32_t>::iterator it2(result2->begin());
      // traverse in row major order
      for (typename Basics::Matrix<T>::iterator it(result->begin());
           it!=result->end(); ++it, ++it2) {
        april_assert(it2 != result2->end());
        input_w.getMatrix(slice.get());
        MatrixSpanReduceMinMax<float,float>(slice.get(),
                                            inter_span_red_functor,
                                            intra_span_red_functor,
                                            zero,
                                            result2,
                                            static_cast<unsigned int>(it2.getRawPos()),
                                            result,
                                            static_cast<unsigned int>(it.getRawPos()));
        input_w.next();
      }
      april_assert(it2 == result2->end());
      return result;
    }

    template<typename T, typename OP>
    Basics::Matrix<T> * MatrixScalarReduceOverDimension(Basics::Matrix<T> *input,
                                                        int dim,
                                                        const OP &scalar_red_functor,
                                                        const T &zero,
                                                        Basics::Matrix<T> *dest) {
      ScalarToSpanReduce<T,OP> span_functor(scalar_red_functor);
      return MatrixSpanReduceOverDimension(input, dim, span_functor,
                                           scalar_red_functor, zero,
                                           dest);
    }

    template<typename T, typename OP1, typename OP2>
    Basics::Matrix<T> * MatrixSpanReduceOverDimension(Basics::Matrix<T> *input,
                                                      int dim,
                                                      const OP1 &inter_span_red_functor,
                                                      const OP2 &intra_span_red_functor,
                                                      const T &zero,
                                                      Basics::Matrix<T> *dest) {
      const int numDim      = input->getNumDim();
      const int *matrixSize = input->getDimPtr();
      AprilUtils::UniquePtr<int []> result_dims( new int[numDim] );
      /**** INPUT sliding window ****/
      AprilUtils::UniquePtr<int []> input_w_size( new int[numDim] );
      AprilUtils::UniquePtr<int []> input_w_num_steps( new int[numDim] );
      int result_size=1;
      for (int i=0; i<dim; ++i) {
        input_w_size[i] = 1;
        result_dims[i] = input_w_num_steps[i] = matrixSize[i];
        result_size *= result_dims[i];
      }
      result_dims[dim] = 1;
      input_w_size[dim] = matrixSize[dim];
      input_w_num_steps[dim] = 1;
      for (int i=dim+1; i<numDim; ++i) {
        input_w_size[i] = 1;
        result_dims[i] = input_w_num_steps[i] = matrixSize[i];
        result_size *= result_dims[i];
      }
      typename Basics::Matrix<T>::sliding_window input_w(input,input_w_size.get(),
                                                         0,0,
                                                         input_w_num_steps.get(),0);
      AprilUtils::SharedPtr< Basics::Matrix<T> > slice( input_w.getMatrix() );
      /******************************/
      Basics::Matrix<T> *result = dest;
      if (result == 0) result = new Basics::Matrix<T>(numDim, result_dims.get(),
                                                      input->getMajorOrder());
      else if (result->size() != result_size)
        // else if (!result->sameDim(result_dims, numDim))
        ERROR_EXIT2(256, "Incorrect size at the given dest matrtix, "
                    "expected %d, found %d\n", result_size, result->size());
      // Forces to mark PPAL memory as updated, needed to avoid copying the
      // memory block because of the iterator.
      T *force_ppal_bit = result->getRawDataAccess()->getPPALForWrite();
      UNUSED_VARIABLE(force_ppal_bit);
      // traverse in row major order
      for (typename Basics::Matrix<T>::iterator it(result->begin());
           it!=result->end(); ++it) {
        input_w.getMatrix(slice.get());
        MatrixSpanReduce1(slice.get(),
                          inter_span_red_functor,
                          intra_span_red_functor,
                          zero,
                          result,
                          it.getRawPos());
        input_w.next();
      }
      return result;
    }

    template<typename T, typename OP>
    T MatrixScalarSumReduce1(const Basics::Matrix<T> *input,
                             const OP &scalar_red_functor,
                             Basics::Matrix<T> *dest,
                             unsigned int dest_raw_pos,
                             int N_th, unsigned int SIZE_th) {
      ScalarToSpanReduce<T,OP> span_functor(scalar_red_functor);
      return MatrixSpanSumReduce1(input, span_functor, dest, dest_raw_pos,
                                  N_th, SIZE_th);
    }

    template<typename T, typename O, typename OP1, typename OP2>
    O MatrixSpanReduce1(const Basics::Matrix<T> *input,
                        const OP1 &inter_span_red_functor,
                        const OP2 &intra_span_red_functor,
                        const O &zero,
                        Basics::Matrix<O> *dest,
                        unsigned int dest_raw_pos) {
      O result = zero;
      // Contiguous memory block
      if (input->getIsContiguous()) {
        result = inter_span_red_functor(static_cast<unsigned int>(input->size()),
                                        input->getRawDataAccess(), 1u,
                                        static_cast<unsigned int>(input->getOffset()),
                                        input->getCudaFlag(),
                                        zero, 0, 0);
      }
      // One dimension
      else if (input->getNumDim() == 1) {
        result = inter_span_red_functor(static_cast<unsigned int>(input->size()),
                                        input->getRawDataAccess(),
                                        static_cast<unsigned int>(input->getStrideSize(0)),
                                        static_cast<unsigned int>(input->getOffset()),
                                        input->getCudaFlag(),
                                        zero, 0, 0);
      }
      // General case
      else {
        typename Basics::Matrix<T>::span_iterator span_it(input);
        unsigned int size   = static_cast<unsigned int>(span_it.getSize());
        unsigned int stride = static_cast<unsigned int>(span_it.getStride());
        const int N = span_it.numberOfIterations();
        for (int i=0; i<N; ++i) {
          april_assert(span_it != input->end_span_iterator());
          O temp = inter_span_red_functor(size,
                                          input->getRawDataAccess(),
                                          stride,
                                          span_it.getOffset(),
                                          input->getCudaFlag(),
                                          zero, 0, 0);
          result = intra_span_red_functor(result, temp);
          ++span_it;
        }
        april_assert(span_it == input->end_span_iterator());
      }
      if (dest != 0) {
        dest->getRawDataAccess()->putValue(dest_raw_pos, result);
      }
      return result;
    } // function MatrixSpanReduce1

    template<typename T, typename OP>
    T MatrixScalarReduce1(const Basics::Matrix<T> *input,
                          const OP &scalar_red_functor,
                          const T &zero,
                          Basics::Matrix<T> *dest,
                          unsigned int dest_raw_pos) {
      ScalarToSpanReduce<T,OP> span_functor(scalar_red_functor);
      return MatrixSpanReduce1(input, span_functor, scalar_red_functor,
                               zero, dest, dest_raw_pos);
    }
    
    template<typename T, typename OP>
    T MatrixSpanSumReduce1(const Basics::Matrix<T> *input,
                           const OP &inter_span_red_functor,
                           Basics::Matrix<T> *dest,
                           unsigned int dest_raw_pos,
                           int N_th,
                           unsigned int SIZE_th) {
#ifdef NO_OMP
      UNUSED_VARIABLE(N_th);
      UNUSED_VARIABLE(SIZE_th);
#endif
      T result = T(0.0f);
      // Contiguous memory block
      if (input->getIsContiguous()) {
        result = inter_span_red_functor(static_cast<unsigned int>(input->size()),
                                        input->getRawDataAccess(), 1u,
                                        static_cast<unsigned int>(input->getOffset()),
                                        input->getCudaFlag(),
                                        T(0.0f), 0, 0);
      }
      // One dimension
      else if (input->getNumDim() == 1) {
        result = inter_span_red_functor(static_cast<unsigned int>(input->size()),
                                        input->getRawDataAccess(),
                                        static_cast<unsigned int>(input->getStrideSize(0)),
                                        static_cast<unsigned int>(input->getOffset()),
                                        input->getCudaFlag(),
                                        T(0.0f), 0, 0);
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
          // Forces execution of memory copy from GPU to PPAL or viceversa (if
          // needed), avoiding race conditions on the following.
          input->update();
#pragma omp parallel for reduction(+:result) firstprivate(span_it)
          for (int i=0; i<N; ++i) {
            span_it.setAtIteration(i);
            result += inter_span_red_functor(size,
                                             input->getRawDataAccess(),
                                             stride,
                                             span_it.getOffset(),
                                             input->getCudaFlag(),
                                             T(0.0f), 0, 0);
          }
        }
        else {
#endif
          for (int i=0; i<N; ++i) {
            april_assert(span_it != input->end_span_iterator());
            result += inter_span_red_functor(size,
                                             input->getRawDataAccess(),
                                             stride,
                                             span_it.getOffset(),
                                             input->getCudaFlag(),
                                             T(0.0f), 0, 0);
            ++span_it;
          }
          april_assert(span_it == input->end_span_iterator());
#ifndef NO_OMP
        }
#endif
      } // General case
      if (dest != 0) {
        dest->getRawDataAccess()->putValue(dest_raw_pos, result);
      }
      return result;
    } // function MatrixSpanSumReduce1

    template<typename T, typename O, typename OP1, typename OP2>
    O MatrixSpanReduce2(const Basics::Matrix<T> *input1,
                        const Basics::Matrix<T> *input2,
                        const OP1 &inter_span_red_functor,
                        const OP2 &intra_span_red_functor,
                        const O &zero,
                        Basics::Matrix<O> *dest,
                        unsigned int dest_raw_pos) {
      O result = zero;
      // Contiguous memory block
      if (input1->getIsContiguous() &&
          input2->getIsContiguous()) {
        result = inter_span_red_functor(static_cast<unsigned int>(input1->size()),
                                        input1->getRawDataAccess(), 1u,
                                        static_cast<unsigned int>(input1->getOffset()),
                                        input2->getRawDataAccess(), 1u,
                                        static_cast<unsigned int>(input2->getOffset()),
                                        input1->getCudaFlag(),
                                        zero, 0, 0);
      }
      // One dimension
      else if (input1->getNumDim() == 1 && input2->getNumDim() == 1) {
        result = inter_span_red_functor(static_cast<unsigned int>(input1->size()),
                                        input1->getRawDataAccess(),
                                        static_cast<unsigned int>(input1->getStrideSize(0)),
                                        static_cast<unsigned int>(input1->getOffset()),
                                        input2->getRawDataAccess(),
                                        static_cast<unsigned int>(input2->getStrideSize(0)),
                                        static_cast<unsigned int>(input2->getOffset()),
                                        input1->getCudaFlag(),
                                        zero, 0, 0);
      }
      // General case
      else {
        typename Basics::Matrix<T>::span_iterator input1_span_it(input1);
        typename Basics::Matrix<T>::span_iterator input2_span_it(input2);
        const int N = input1_span_it.numberOfIterations();
        april_assert(N == input2_span_it.numberOfIterations());
        const unsigned int size          = static_cast<unsigned int>(input1_span_it.getSize());
        const unsigned int input1_stride = static_cast<unsigned int>(input1_span_it.getStride());
        const unsigned int input2_stride = static_cast<unsigned int>(input2_span_it.getStride());
        april_assert(size == static_cast<unsigned int>(input2_span_it.getSize()));
        for (int i=0; i<N; ++i) {
          april_assert(input1_span_it != input1->end_span_iterator());
          april_assert(input2_span_it != input2->end_span_iterator());
          O temp = inter_span_red_functor(size,
                                          input1->getRawDataAccess(),
                                          input1_stride,
                                          input1_span_it.getOffset(),
                                          input2->getRawDataAccess(),
                                          input2_stride,
                                          input2_span_it.getOffset(),
                                          input1->getCudaFlag(),
                                          zero, 0, 0);
          result = intra_span_red_functor(result, temp);
          ++input1_span_it;
          ++input2_span_it;
        }
        april_assert(input1_span_it == input1->end_span_iterator());
        april_assert(input2_span_it == input2->end_span_iterator());
      }
      if (dest != 0) {
        dest->getRawDataAccess()->putValue(dest_raw_pos, result);
      }
      return result;
    } // function MatrixSpanReduce2

  } // namespace MatrixExt
  
} // namespace AprilMath

#endif // REDUCE_MATRIX_IMPL_H