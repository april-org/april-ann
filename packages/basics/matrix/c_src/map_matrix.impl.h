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

    template<typename T1, typename T2, typename O, typename OP>
    Basics::Matrix<O> *MatrixScalarMap2(const Basics::Matrix<T1> *input1,
                                        const Basics::SparseMatrix<T2> *input2,
                                        const OP &functor,
                                        Basics::Matrix<O> *dest,
                                        const int N_th,
                                        const unsigned int SIZE_th) {
      // TODO: Implement using a CPU/CUDA wrapper
      UNUSED_VARIABLE(N_th);
      UNUSED_VARIABLE(SIZE_th);
      april_assert(input1 != 0 && input2 != 0 && dest != 0);
      if (input1->getNumDim() != 2) {
        ERROR_EXIT(256, "Needs 2-dimensional matrices\n");
      }
      if (!input2->sameDim(input1)) {
        ERROR_EXIT(256, "Incompatible matrix sizes\n");
      }
      if (input2->getSparseFormat() != CSR_FORMAT) {
        ERROR_EXIT(256, "Needs a CSR sparse matrix\n");
      }
      int input2_i, input2_j=0;
      typename Basics::Matrix<O>::iterator dest_it = dest->begin();
      typename Basics::Matrix<T1>::const_iterator input1_it = input1->begin();
      typename Basics::SparseMatrix<T2>::const_iterator input2_it = input2->begin();
      for (int i=0; i<input1->getDimSize(0); ++i) {
        april_assert(input1_it != input1->end());
        april_assert(dest_it != dest->end());
        for (int j=0; j<input1->getDimSize(1); ++j) {
          bool input2_end = (input2_it == input2->end());
          if (!input2_end) input2_it.getCoords(input2_i, input2_j);
          else input2_i = input1->getDimSize(0);
          T2 input2;
          if (input2_i == i && input2_j == j) {
            // non-zero input2
            input2 = *input2_it;
            ++input2_it;
          }
          else {
            input2 = T2(0.0f);
          }
          *dest_it = functor(*input1_it, input2);
          //
          ++input1_it;
          ++dest_it;
        }
      }
      return dest;
    }

    template<typename T1, typename T2, typename T3, typename O, typename OP>
    Basics::Matrix<O> *MatrixScalarMap3(const Basics::Matrix<T1> *input1,
                                        const Basics::Matrix<T2> *input2,
                                        const Basics::Matrix<T3> *input3,
                                        const OP &functor,
                                        Basics::Matrix<O> *dest,
                                        const int N_th,
                                        const unsigned int SIZE_th) {
      ScalarToSpanMap3<T1,T2,T3,O,OP> span_functor(functor);
      return MatrixSpanMap3(input1, input2, input3, span_functor, dest, N_th, SIZE_th);
    }
  
    template<typename T, typename O, typename OP>
    Basics::Matrix<O> *MatrixSpanMap1(const Basics::Matrix<T> *input,
                                      const OP &functor,
                                      Basics::Matrix<O> *dest,
                                      const int N_th,
                                      const unsigned int SIZE_th) {
      april_assert(input != 0 && dest != 0);
      if (input->size() != dest->size()) {
        ERROR_EXIT(128, "Incompatible matrix sizes or dimensions\n");
      }
#ifdef NO_OMP
      UNUSED_VARIABLE(N_th);
      UNUSED_VARIABLE(SIZE_th);
#endif
      bool cuda_flag = input->getCudaFlag() || dest->getCudaFlag();
      // Contiguous memory block or one dimension.
      if ( (input->getIsContiguous() || input->getNumDim() == 1) &&
           (dest->getIsContiguous() || dest->getNumDim() == 1) ) {
        unsigned int size = input->size();
        unsigned int input_stride, input_offset = input->getOffset();
        unsigned int dest_stride, dest_offset = dest->getOffset();
        if (input->getIsContiguous()) input_stride = 1u;
        else input_stride = input->getStrideSize(0);
        if (dest->getIsContiguous()) dest_stride = 1u;
        else dest_stride = dest->getStrideSize(0);
        functor(size,
                input->getRawDataAccess(), input_stride, input_offset,
                dest->getRawDataAccess(), dest_stride, dest_offset,
                cuda_flag);
      }
      // General case.
      else {
        if (!input->sameDim(dest)) {
          ERROR_EXIT(128, "Incompatible matrix sizes or dimensions\n");
        }
        typename Basics::Matrix<T>::span_iterator input_span_it(input);
        typename Basics::Matrix<O>::span_iterator dest_span_it(dest,
                                                               input_span_it.getDimOrder());
        const int N = input_span_it.numberOfIterations();
        april_assert(N == dest_span_it.numberOfIterations());
        const unsigned int size         = static_cast<unsigned int>(input_span_it.getSize());
        const unsigned int input_stride = static_cast<unsigned int>(input_span_it.getStride());
        const unsigned int dest_stride  = static_cast<unsigned int>(dest_span_it.getStride());
        april_assert(size == static_cast<unsigned int>(dest_span_it.getSize()));
#ifdef USE_CUDA
        // Forces execution of memory copy from GPU to PPAL or viceversa (if
        // needed), avoiding race conditions on the following.
        input->getRawDataAccess()->forceSync(cuda_flag);
#endif
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
                    cuda_flag);
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
                    cuda_flag);
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
      if (input1->size() != dest->size() || input2->size() != dest->size()) {
        ERROR_EXIT(128, "Incompatible matrix sizes or dimensions\n");
      }
#ifdef NO_OMP
      UNUSED_VARIABLE(N_th);
      UNUSED_VARIABLE(SIZE_th);
#endif
      bool cuda_flag = input1->getCudaFlag() || input2->getCudaFlag() ||
        dest->getCudaFlag();
      // Contiguous memory block or one dimension.
      if ( (input1->getIsContiguous() || input1->getNumDim() == 1) &&
           (input2->getIsContiguous() || input2->getNumDim() == 1) &&
           (dest->getIsContiguous() || dest->getNumDim() == 1) ) {
        unsigned int size = input1->size();
        unsigned int input1_stride, input1_offset = input1->getOffset();
        unsigned int input2_stride, input2_offset = input2->getOffset();
        unsigned int dest_stride, dest_offset = dest->getOffset();
        if (input1->getIsContiguous()) input1_stride = 1u;
        else input1_stride = input1->getStrideSize(0);
        if (input2->getIsContiguous()) input2_stride = 1u;
        else input2_stride = input2->getStrideSize(0);
        if (dest->getIsContiguous()) dest_stride = 1u;
        else dest_stride = dest->getStrideSize(0);
        functor(size,
                input1->getRawDataAccess(), input1_stride, input1_offset,
                input2->getRawDataAccess(), input2_stride, input2_offset,
                dest->getRawDataAccess(), dest_stride, dest_offset,
                cuda_flag);
      }
      // General case.
      else {
        if (!input1->sameDim(dest) || !input2->sameDim(dest)) {
          ERROR_EXIT(128, "Incompatible matrix sizes or dimensions\n");
        }
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
#ifdef USE_CUDA
        // Forces execution of memory copy from GPU to PPAL or viceversa (if
        // needed), avoiding race conditions on the following.
        input1->getRawDataAccess()->forceSync(cuda_flag);
        input2->getRawDataAccess()->forceSync(cuda_flag);
#endif
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
                    cuda_flag);
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
                    cuda_flag);
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

    template<typename T1, typename T2, typename T3, typename O, typename OP>
    Basics::Matrix<O> *MatrixSpanMap3(const Basics::Matrix<T1> *input1,
                                      const Basics::Matrix<T2> *input2,
                                      const Basics::Matrix<T3> *input3,
                                      const OP &functor,
                                      Basics::Matrix<O> *dest,
                                      const int N_th,
                                      const unsigned int SIZE_th) {
      april_assert(input1 != 0 && input2 != 0 && input3 != 0 && dest != 0);
      if (input1->size() != dest->size() || input2->size() != dest->size() ||
          input3->size() != dest->size()) {
        ERROR_EXIT(128, "Incompatible matrix sizes or dimensions\n");
      }
#ifdef NO_OMP
      UNUSED_VARIABLE(N_th);
      UNUSED_VARIABLE(SIZE_th);
#endif
      bool cuda_flag = input1->getCudaFlag() || input2->getCudaFlag() ||
        input3->getCudaFlag() || dest->getCudaFlag();
      // Contiguous memory block or one dimension.
      if ( (input1->getIsContiguous() || input1->getNumDim() == 1) &&
           (input2->getIsContiguous() || input2->getNumDim() == 1) &&
           (input3->getIsContiguous() || input3->getNumDim() == 1) &&
           (dest->getIsContiguous() || dest->getNumDim() == 1) ) {
        unsigned int size = input1->size();
        unsigned int input1_stride, input1_offset = input1->getOffset();
        unsigned int input2_stride, input2_offset = input2->getOffset();
        unsigned int input3_stride, input3_offset = input3->getOffset();
        unsigned int dest_stride, dest_offset = dest->getOffset();
        if (input1->getIsContiguous()) input1_stride = 1u;
        else input1_stride = input1->getStrideSize(0);
        if (input2->getIsContiguous()) input2_stride = 1u;
        else input2_stride = input2->getStrideSize(0);
        if (input3->getIsContiguous()) input3_stride = 1u;
        else input3_stride = input3->getStrideSize(0);
        if (dest->getIsContiguous()) dest_stride = 1u;
        else dest_stride = dest->getStrideSize(0);
        functor(size,
                input1->getRawDataAccess(), input1_stride, input1_offset,
                input2->getRawDataAccess(), input2_stride, input2_offset,
                input3->getRawDataAccess(), input3_stride, input3_offset,
                dest->getRawDataAccess(), dest_stride, dest_offset,
                cuda_flag);
      }
      // General case.
      else {
        if (!input1->sameDim(dest) || !input2->sameDim(dest) ||
            !input3->sameDim(dest)) {
          ERROR_EXIT(128, "Incompatible matrix sizes or dimensions\n");
        }
        typename Basics::Matrix<T1>::span_iterator input1_span_it(input1);
        typename Basics::Matrix<T2>::span_iterator input2_span_it(input2);
        typename Basics::Matrix<T3>::span_iterator input3_span_it(input3);
        typename Basics::Matrix<O>::span_iterator dest_span_it(dest,
                                                               input1_span_it.getDimOrder());
        const int N = input1_span_it.numberOfIterations();
        april_assert(N == dest_span_it.numberOfIterations());
        april_assert(N == input2_span_it.numberOfIterations());
        april_assert(N == input3_span_it.numberOfIterations());
        const unsigned int size          = static_cast<unsigned int>(input1_span_it.getSize());
        const unsigned int input1_stride = static_cast<unsigned int>(input1_span_it.getStride());
        const unsigned int input2_stride = static_cast<unsigned int>(input2_span_it.getStride());
        const unsigned int input3_stride = static_cast<unsigned int>(input3_span_it.getStride());
        const unsigned int dest_stride   = static_cast<unsigned int>(dest_span_it.getStride());
        april_assert(size == static_cast<unsigned int>(input2_span_it.getSize()));
        april_assert(size == static_cast<unsigned int>(input3_span_it.getSize()));
        april_assert(size == static_cast<unsigned int>(dest_span_it.getSize()));
#ifdef USE_CUDA
        // Forces execution of memory copy from GPU to PPAL or viceversa (if
        // needed), avoiding race conditions on the following.
        input1->getRawDataAccess()->forceSync(cuda_flag);
        input2->getRawDataAccess()->forceSync(cuda_flag);
        input3->getRawDataAccess()->forceSync(cuda_flag);
#endif
#ifndef NO_OMP
        // This if controls the execution using OMP only when the number of threads
        // is more than 1 and the iterator size is large enough.
        if (OMPUtils::get_num_threads() > 1 && N > N_th && size > SIZE_th) {
#pragma omp parallel for firstprivate(input1_span_it) firstprivate(input2_span_it) firstprivate(input3_span_it) firstprivate(dest_span_it)
          for (int i=0; i<N; ++i) {
            input1_span_it.setAtIteration(i);
            input2_span_it.setAtIteration(i);
            input3_span_it.setAtIteration(i);
            dest_span_it.setAtIteration(i);
            //
            functor(size,
                    input1->getRawDataAccess(),
                    input1_stride,
                    static_cast<unsigned int>(input1_span_it.getOffset()),
                    input2->getRawDataAccess(),
                    input2_stride,
                    static_cast<unsigned int>(input2_span_it.getOffset()),
                    input3->getRawDataAccess(),
                    input3_stride,
                    static_cast<unsigned int>(input3_span_it.getOffset()),
                    dest->getRawDataAccess(),
                    dest_stride,
                    static_cast<unsigned int>(dest_span_it.getOffset()),
                    cuda_flag);
          } // for every possible span
        } // if num_threads>1 and large enough computation
        else {
#endif
          // sequential code, with less overhead when updating iterator
          for (int i=0; i<N; ++i) {
            april_assert(input1_span_it != input1->end_span_iterator());
            april_assert(input2_span_it != input2->end_span_iterator());
            april_assert(input3_span_it != input3->end_span_iterator());
            april_assert(dest_span_it != dest->end_span_iterator());
            functor(size,
                    input1->getRawDataAccess(),
                    input1_stride,
                    static_cast<unsigned int>(input1_span_it.getOffset()),
                    input2->getRawDataAccess(),
                    input2_stride,
                    static_cast<unsigned int>(input2_span_it.getOffset()),
                    input3->getRawDataAccess(),
                    input3_stride,
                    static_cast<unsigned int>(input3_span_it.getOffset()),
                    dest->getRawDataAccess(),
                    dest_stride,
                    static_cast<unsigned int>(dest_span_it.getOffset()),
                    cuda_flag);
            //
            ++input1_span_it;
            ++input2_span_it;
            ++input3_span_it;
            ++dest_span_it;
          } // for every possible span
          april_assert(input1_span_it == input1->end_span_iterator());
          april_assert(input2_span_it == input2->end_span_iterator());
          april_assert(input3_span_it == input3->end_span_iterator());
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
