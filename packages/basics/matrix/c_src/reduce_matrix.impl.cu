#include "april_assert.h"
#include "matrix.h"
#include "omp_utils.h"
#include "reduce_matrix.h"
#include "reduce_template.h"
namespace april_math {

  template<typename T, typename O, typename OP1, typename OP2>
  O MatrixScalarReduce1(const basics::Matrix<T> *input,
                        const OP1 &scalar_red_functor,
                        const OP2 &intra_span_red_functor,
                        const O &zero,
                        basics::Matrix<O> *dest,
                        unsigned int dest_raw_pos) {
    ScalarToSpanReduce1<T,O,OP> span_functor(scalar_red_functor);
    return MatrixSpanReduce1(input, span_functor, intra_span_red_functor,
                             zero, dest, dest_raw_pos);
  }

  template<typename T, typename O, typename OP>
  O MatrixScalarSumReduce1(const basics::Matrix<T> *input,
                           const OP &scalar_red_functor,
                           basics::Matrix<O> *dest,
                           unsigned int dest_raw_pos,
                           int N_th, unsigned int SIZE_th) {
    ScalarToSpanReduce1<T,O,OP> span_functor(scalar_red_functor);
    return MatrixSpanSumReduce1(input, span_functor, dest, dest_raw_pos,
                                N_th, SIZE_th);
  }

  template<typename T, typename O, typename OP1, typename OP2>
  O MatrixSpanReduce1(const basics::Matrix<T> *input,
                      const OP1 &inter_span_red_functor,
                      const OP2 &intra_span_red_functor,
                      const O &zero,
                      basics::Matrix<O> *dest,
                      unsigned int dest_raw_pos) {
    O result = zero;
    // Contiguous memory block
    if (input->getIsContiguous()) {
      result = inter_span_red_functor(static_cast<unsigned int>(input->size()),
                                      input->getRawDataAccess(), 1u,
                                      static_cast<unsigned int>(input->getOffset()),
                                      input->getCudaFlag(),
                                      zero);
    }
    // One dimension
    else if (input->getNumDim() == 1) {
      result = inter_span_red_functor(static_cast<unsigned int>(input->size()),
                                      input->getRawDataAccess(),
                                      static_cast<unsigned int>(input->getStrideSize(0)),
                                      static_cast<unsigned int>(input->getOffset()),
                                      input->getCudaFlag(),
                                      zero);
    }
    // General case
    else {
      typename basics::Matrix<T>::span_iterator span_it(input);
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
                                        zero);
        result = intra_span_red_functor(result, temp);
        ++span_it;
      }
      april_assert(span_it == input->end_span_iterator());
    }
    if (dest != 0) {
      dest->getRawDataAccess()->putValue(dest_raw_pos, result);
    }
  } // function MatrixSpanReduce1

  template<typename T, typename O, typename OP>
  O MatrixSpanSumReduce1(const basics::Matrix<T> *input,
                         const OP &inter_span_red_functor,
                         basics::Matrix<O> *dest,
                         unsigned int dest_raw_pos,
                         int N_th,
                         unsigned int SIZE_th) {
    O result = T(0.0f);
    // Contiguous memory block
    if (input->getIsContiguous()) {
      result = inter_span_red_functor(static_cast<unsigned int>(input->size()),
                                      input->getRawDataAccess(), 1u,
                                      static_cast<unsigned int>(input->getOffset()),
                                      input->getCudaFlag(),
                                      T(0.0f));
    }
    // One dimension
    else if (input->getNumDim() == 1) {
      result = inter_span_red_functor(static_cast<unsigned int>(input->size()),
                                      input->getRawDataAccess(),
                                      static_cast<unsigned int>(input->getStrideSize(0)),
                                      static_cast<unsigned int>(input->getOffset()),
                                      input->getCudaFlag(),
                                      T(0.0f));
    }
    // General case
    else {
      typename basics::Matrix<T>::span_iterator span_it(input);
      unsigned int size   = static_cast<unsigned int>(span_it.getSize());
      unsigned int stride = static_cast<unsigned int>(span_it.getStride());
      const int N = span_it.numberOfIterations();
#ifndef NO_OMP
      // this if controls the execution using OMP only when the number of threads
      // is more than 1 and the iterator size is big enough
      if (omp_utils::get_num_threads() > 1 && N > N_th && size > SIZE_th) {
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
                                           T(0.0f));
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
                                           T(0.0f));
          ++span_it;
        }
        april_assert(span_it == input->end_span_iterator());
      }
    }
    if (dest != 0) {
      dest->getRawDataAccess()->putValue(dest_raw_pos, result);
    }
  } // function MatrixSpanSumReduce1

  template<typename T1, typename T2, typename O, typename OP1, typename OP2>
  O MatrixScalarReduce2(const basics::Matrix<T1> *input1,
                        const basics::Matrix<T2> *input2,
                        const OP &scalar_red_functor,
                        const OP &intra_span_red_functor,
                        const O &zero,
                        basics::Matrix<O> *dest,
                        unsigned int dest_raw_pos) {
    ScalarToSpanReduce2<T1,T2,O,OP> span_functor(scalar_red_functor);
    return MatrixSpanReduce2(input1, input2,
                             span_functor, intra_span_red_functor,
                             zero, dest, dest_raw_pos);
  } // function MatrixScalarReduce2
 
  template<typename T1, typename T2, typename O, typename OP1, typename OP2>
  O MatrixSpanReduce2(const basics::Matrix<T1> *input1,
                      const basics::Matrix<T2> *input2,
                      const OP1 &inter_span_red_functor,
                      const OP2 &intra_span_red_functor,
                      const O &zero,
                      basics::Matrix<O> *dest,
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
                                      zero);
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
                                      zero);
    }
    // General case
    else {
      typename basics::Matrix<T1>::span_iterator input1_span_it(input1);
      typename basics::Matrix<T1>::span_iterator input2_span_it(input2);
      typename basics::Matrix<O>::span_iterator dest_span_it(dest,
                                                             input1_span_it.getDimOrder());
      const int N = input1_span_it.numberOfIterations();
      april_assert(N == static_cast<unsigned int>(dest_span_it.numberOfIterations()));
      april_assert(N == static_cast<unsigned int>(input2_span_it.numberOfIterations()));
      const unsigned int size          = static_cast<unsigned int>(input1_span_it.getSize());
      const unsigned int input1_stride = static_cast<unsigned int>(input1_span_it.getStride());
      const unsigned int input2_stride = static_cast<unsigned int>(input2_span_it.getStride());
      const unsigned int dest_stride   = static_cast<unsigned int>(dest_span_it.getStride());
      april_assert(size == static_cast<unsigned int>(input2_span_it.getSize()));
      april_assert(size == static_cast<unsigned int>(dest_span_it.getSize()));
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
                                        input->getCudaFlag(),
                                        zero);
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
  } // function MatrixSpanReduce2

} // namespace april_math
