#include "april_assert.h"
#include "matrix.h"
#include "omp_utils.h"
#include "reduce_matrix.h"
#include "reduce_template.h"
namespace april_math {

  template<typename T, typename O, typename OP>
  O MatrixScalarReduce1(const basics::Matrix<T> *input,
                        const OP &scalar_red_functor,
                        const O &zero,
                        basics::Matrix<T> *dest,
                        unsigned int dest_raw_pos) {
    ScalarToSpanReduce1<T,O,OP> span_functor(scalar_red_functor);
    return MatrixSpanReduce1(input, span_functor, scalar_red_functor,
                             zero, dest, dest_raw_pos);
  }

  template<typename T, typename O, typename OP>
  O MatrixScalarSumReduce1(const basics::Matrix<T> *input,
                           const OP &scalar_red_functor,
                           basics::Matrix<T> *dest,
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
                      basics::Matrix<T> *dest,
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
                         basics::Matrix<T> *dest,
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

} // namespace april_math
