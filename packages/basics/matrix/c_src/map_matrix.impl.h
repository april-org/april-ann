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
#ifndef MAP_MATRIX_IMPL_H
#define MAP_MATRIX_IMPL_H

// It must be defined here.
#include "matrix.h"

#include "map_matrix.h"
#include "map_template.h"
#include "omp_utils.h"

namespace AprilMath {

  namespace MatrixExt {

    template<typename T, typename O, typename OP>
    Basics::Matrix<O> *MatrixScalarMap1(const Basics::Matrix<T> *input,
                                        const OP &functor,
                                        Basics::Matrix<O> *dest,
                                        const int N_th,
                                        const unsigned int SIZE_th) {
      ScalarToSpanMap1<T,O,OP> span_functor(functor);
      return MatrixSpanMap1(input, span_functor, dest, N_th, SIZE_th);
    }
  
    template<typename T1, typename T2, typename O, typename OP>
    Basics::Matrix<O> *MatrixScalarMap2(const Basics::Matrix<T1> *input1,
                                        const Basics::Matrix<T2> *input2,
                                        const OP &functor,
                                        Basics::Matrix<O> *dest,
                                        const int N_th,
                                        const unsigned int SIZE_th) {
      ScalarToSpanMap2<T1,T2,O,OP> span_functor(functor);
      return MatrixSpanMap2(input1, input2, span_functor, dest, N_th, SIZE_th);
    }
  
    template<typename T, typename O, typename OP>
    Basics::Matrix<O> *MatrixSpanMap1(const Basics::Matrix<T> *input,
                                      const OP &functor,
                                      Basics::Matrix<O> *dest,
                                      const int N_th,
                                      const unsigned int SIZE_th) {
      april_assert(input != 0 && dest != 0);
      if (input->size() != dest->size() || !input->sameDim(dest)) {
        ERROR_EXIT(128, "Incorrect matrix sizes or dimensions\n");
      }
#ifdef NO_OMP
      UNUSED_VARIABLE(N_th);
      UNUSED_VARIABLE(SIZE_th);
#endif
      // Contiguous memory block.
      if (input->getIsContiguous() && dest->getIsContiguous()) {
        functor(static_cast<unsigned int>(input->size()),
                input->getRawDataAccess(),
                1u,
                static_cast<unsigned int>(input->getOffset()),
                dest->getRawDataAccess(),
                1u,
                static_cast<unsigned int>(dest->getOffset()),
                input->getCudaFlag());
      }
      // One dimension.
      else if (input->getNumDim() == 1) {
        functor(static_cast<unsigned int>(input->size()),
                input->getRawDataAccess(),
                static_cast<unsigned int>(input->getStrideSize(0)),
                static_cast<unsigned int>(input->getOffset()),
                dest->getRawDataAccess(),
                static_cast<unsigned int>(dest->getStrideSize(0)),
                static_cast<unsigned int>(dest->getOffset()),
                input->getCudaFlag());
      }
      // General case.
      else {
        typename Basics::Matrix<T>::span_iterator input_span_it(input);
        typename Basics::Matrix<O>::span_iterator dest_span_it(dest,
                                                               input_span_it.getDimOrder());
        const int N = input_span_it.numberOfIterations();
        april_assert(N == dest_span_it.numberOfIterations());
        const unsigned int size         = static_cast<unsigned int>(input_span_it.getSize());
        const unsigned int input_stride = static_cast<unsigned int>(input_span_it.getStride());
        const unsigned int dest_stride  = static_cast<unsigned int>(dest_span_it.getStride());
        april_assert(size == static_cast<unsigned int>(dest_span_it.getSize()));
        // Forces execution of memory copy from GPU to PPAL or viceversa (if
        // needed), avoiding race conditions on the following.
        input->update();
#ifndef NO_OMP
        // This if controls the execution using OMP only when the number of threads
        // is more than 1 and the iterator size is large enough.
        if (OMPUtils::get_num_threads() > 1 && N > N_th && size > SIZE_th) {
#pragma omp parallel for firstprivate(input_span_it) firstprivate(dest_span_it)
          for (int i=0; i<N; ++i) {
            input_span_it.setAtIteration(i);
            dest_span_it.setAtIteration(i);
            //
            functor(size,
                    input->getRawDataAccess(),
                    input_stride,
                    static_cast<unsigned int>(input_span_it.getOffset()),
                    dest->getRawDataAccess(),
                    dest_stride,
                    static_cast<unsigned int>(dest_span_it.getOffset()),
                    input->getCudaFlag());
          } // for every possible span
        } // if num_threads>1 and large enough computation
        else {
#endif
          // sequential code, with less overhead when updating iterator
          for (int i=0; i<N; ++i) {
            april_assert(input_span_it != input->end_span_iterator());
            april_assert(dest_span_it != dest->end_span_iterator());
            functor(size,
                    input->getRawDataAccess(),
                    input_stride,
                    static_cast<unsigned int>(input_span_it.getOffset()),
                    dest->getRawDataAccess(),
                    dest_stride,
                    static_cast<unsigned int>(dest_span_it.getOffset()),
                    input->getCudaFlag());
            //
            ++input_span_it;
            ++dest_span_it;
          } // for every possible span
          april_assert(input_span_it == input->end_span_iterator());
          april_assert(dest_span_it == dest->end_span_iterator());
#ifndef NO_OMP
        } // else (num_threads==1 or not large enough computation)
#endif
      } // General case.
      return dest;
    } // MatrixMap1 function

    template<typename T1, typename T2, typename O, typename OP>
    Basics::Matrix<O> *MatrixSpanMap2(const Basics::Matrix<T1> *input1,
                                      const Basics::Matrix<T2> *input2,
                                      const OP &functor,
                                      Basics::Matrix<O> *dest,
                                      const int N_th,
                                      const unsigned int SIZE_th) {
      april_assert(input1 != 0 && input2 != 0 && dest != 0);
      if (input1->size() != dest->size() || input2->size() != dest->size() ||
          !input1->sameDim(dest) || !input2->sameDim(dest)) {
        ERROR_EXIT(128, "Incorrect matrix sizes or dimensions\n");
      }
#ifdef NO_OMP
      UNUSED_VARIABLE(N_th);
      UNUSED_VARIABLE(SIZE_th);
#endif
      // Contiguous memory block.
      if (input1->getIsContiguous() && input2->getIsContiguous() &&
          dest->getIsContiguous()) {
        functor(static_cast<unsigned int>(input1->size()),
                input1->getRawDataAccess(),
                1u,
                static_cast<unsigned int>(input1->getOffset()),
                input2->getRawDataAccess(),
                1u,
                static_cast<unsigned int>(input2->getOffset()),
                dest->getRawDataAccess(),
                1u,
                static_cast<unsigned int>(dest->getOffset()),
                input1->getCudaFlag());
      }
      // One dimension.
      else if (input1->getNumDim() == 1) {
        functor(static_cast<unsigned int>(input1->size()),
                input1->getRawDataAccess(),
                static_cast<unsigned int>(input1->getStrideSize(0)),
                static_cast<unsigned int>(input1->getOffset()),
                input2->getRawDataAccess(),
                static_cast<unsigned int>(input2->getStrideSize(0)),
                static_cast<unsigned int>(input2->getOffset()),
                dest->getRawDataAccess(),
                static_cast<unsigned int>(dest->getStrideSize(0)),
                static_cast<unsigned int>(dest->getOffset()),
                input1->getCudaFlag());
      }
      // General case.
      else {
        typename Basics::Matrix<T1>::span_iterator input1_span_it(input1);
        typename Basics::Matrix<T2>::span_iterator input2_span_it(input2);
        typename Basics::Matrix<O>::span_iterator dest_span_it(dest,
                                                               input1_span_it.getDimOrder());
        const int N = input1_span_it.numberOfIterations();
        april_assert(N == dest_span_it.numberOfIterations());
        april_assert(N == input2_span_it.numberOfIterations());
        const unsigned int size          = static_cast<unsigned int>(input1_span_it.getSize());
        const unsigned int input1_stride = static_cast<unsigned int>(input1_span_it.getStride());
        const unsigned int input2_stride = static_cast<unsigned int>(input2_span_it.getStride());
        const unsigned int dest_stride   = static_cast<unsigned int>(dest_span_it.getStride());
        april_assert(size == static_cast<unsigned int>(input2_span_it.getSize()));
        april_assert(size == static_cast<unsigned int>(dest_span_it.getSize()));
        // Forces execution of memory copy from GPU to PPAL or viceversa (if
        // needed), avoiding race conditions on the following.
        input1->update();
        input2->update();
#ifndef NO_OMP
        // This if controls the execution using OMP only when the number of threads
        // is more than 1 and the iterator size is large enough.
        if (OMPUtils::get_num_threads() > 1 && N > N_th && size > SIZE_th) {
#pragma omp parallel for firstprivate(input1_span_it) firstprivate(input2_span_it) firstprivate(dest_span_it)
          for (int i=0; i<N; ++i) {
            input1_span_it.setAtIteration(i);
            input2_span_it.setAtIteration(i);
            dest_span_it.setAtIteration(i);
            //
            functor(size,
                    input1->getRawDataAccess(),
                    input1_stride,
                    static_cast<unsigned int>(input1_span_it.getOffset()),
                    input2->getRawDataAccess(),
                    input2_stride,
                    static_cast<unsigned int>(input2_span_it.getOffset()),
                    dest->getRawDataAccess(),
                    dest_stride,
                    static_cast<unsigned int>(dest_span_it.getOffset()),
                    input1->getCudaFlag());
          } // for every possible span
        } // if num_threads>1 and large enough computation
        else {
#endif
          // sequential code, with less overhead when updating iterator
          for (int i=0; i<N; ++i) {
            april_assert(input1_span_it != input1->end_span_iterator());
            april_assert(input2_span_it != input2->end_span_iterator());
            april_assert(dest_span_it != dest->end_span_iterator());
            functor(size,
                    input1->getRawDataAccess(),
                    input1_stride,
                    static_cast<unsigned int>(input1_span_it.getOffset()),
                    input2->getRawDataAccess(),
                    input2_stride,
                    static_cast<unsigned int>(input2_span_it.getOffset()),
                    dest->getRawDataAccess(),
                    dest_stride,
                    static_cast<unsigned int>(dest_span_it.getOffset()),
                    input1->getCudaFlag());
            //
            ++input1_span_it;
            ++input2_span_it;
            ++dest_span_it;
          } // for every possible span
          april_assert(input1_span_it == input1->end_span_iterator());
          april_assert(input2_span_it == input2->end_span_iterator());
          april_assert(dest_span_it == dest->end_span_iterator());
#ifndef NO_OMP
        } // else (num_threads==1 or not large enough computation)
#endif
      } // General case.
      return dest;
    } // MatrixMap2 function

  } // namespace MatrixExt

} // namespace AprilMath

#endif // MAP_MATRIX_IMPL_H
