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
#ifndef REDUCE_MATRIX_H
#define REDUCE_MATRIX_H

#include "cmath_overloads.h"
#include "mathcore.h"

#define DEFAULT_N_TH 100
#define DEFAULT_SIZE_TH 100u

namespace Basics {
  // forward declaration
  template <typename T>
  class Matrix;
}

namespace AprilMath {
  
  /**
   * @code
   * // An advanced example of what can be done using a reduction defined
   * // in this namespace: AprilMath::MatrixExt::MatrixScalarReduce1
   * // The example uses the AprilMath::Function::r_add reduction, which uses
   * // operator+= to reduce two values which can be of different types.
   * #include "reduce_matrix.h"
   * struct MeanReduceResult {
   *   float sum;
   *   int N;
   *   MeanReduceResult() : sum(0.0f), N(0) { }
   *   float getMean() const { return sum/N; }
   *   // Called from AprilMath::Functors::r_add<MeanReduceResult,float>,
   *   APRIL_CUDA_EXPORT MeanReduceResult &operator+=(const float &b) const {
   *     acc.N++;
   *     acc.sum += b;
   *     return *this;
   *   }
   *   // Called from AprilMath::Functors::r_add<MeanReduceResult,MeanReduceResult>
   *   APRIL_CUDA_EXPORT MeanReduceResult &operator+=(const MeanReduceResult &b) const {
   *     acc.sum += b.sum;
   *     acc.N   += b.N;
   *     return *this;
   *   }
   * }
   * // a matrix of 1 dimension and 1 element
   * MeanReduceResult result =
   *   MatrixScalarReduce1(my_matrix_float, // a pointer to a MatrixFloat instance
   *                       AprilMath::Functors::r_add<MeanReduceResult,float>,
   *                       AprilMath::Functors::r_add<MeanReduceResult,MeanReduceResult>,
   *                       MeanReduceResult());
   * printf("Mean result: %f\n", result.getMean());
   * @endcode
   */
  namespace MatrixExt {
  
    template<typename T, typename O, typename OP1, typename OP2>
    Basics::Matrix<O> * MatrixScalarReduceOverDimension(Basics::Matrix<T> *input,
                                                        int dim,
                                                        const OP1 &scalar_red_functor,
                                                        const OP2 &partials_red_functor,
                                                        const O &zero,
                                                        Basics::Matrix<O> *dest);
    
    template<typename T, typename O, typename OP1, typename OP2>
    Basics::Matrix<O> * MatrixSpanReduceOverDimension(Basics::Matrix<T> *input,
                                                      int dim,
                                                      const OP1 &inter_span_red_functor,
                                                      const OP2 &intra_span_red_functor,
                                                      const O &zero,
                                                      Basics::Matrix<O> *dest);
    
    template<typename T, typename OP>
    Basics::Matrix<T> * MatrixScalarReduceMinMaxOverDimension(Basics::Matrix<T> *input,
                                                              int dim,
                                                              const OP &scalar_red_functor,
                                                              const T &zero,
                                                              Basics::Matrix<int32_t> *which,
                                                              Basics::Matrix<T> *dest);
    
    template<typename T, typename OP1, typename OP2>
    Basics::Matrix<T> * MatrixSpanReduceMinMaxOverDimension(Basics::Matrix<T> *input,
                                                            int dim,
                                                            const OP1 &inter_span_red_functor,
                                                            const OP2 &intra_span_red_functor,
                                                            const T &zero,
                                                            Basics::Matrix<int32_t> *which,
                                                            Basics::Matrix<T> *dest);
    
    template<typename T, typename O, typename OP1, typename OP2>
    void MatrixScalarReduce1(const Basics::Matrix<T> *input,
                             const OP1 &scalar_red_functor,
                             const OP2 &partials_red_functor,
                             const O &zero,
                             Basics::Matrix<O> *dest,
                             unsigned int dest_raw_pos=0);
    
    template<typename T, typename O, typename OP1, typename OP2>
    O MatrixScalarReduce1(const Basics::Matrix<T> *input,
                          const OP1 &scalar_red_functor,
                          const OP2 &partials_red_functor,
                          const O &zero);
    
    template<typename T, typename O, typename OP1, typename OP2>
    void MatrixSpanReduce1(const Basics::Matrix<T> *input,
                           const OP1 &inter_span_red_functor,
                           const OP2 &intra_span_red_functor,
                           const O &zero,
                           Basics::Matrix<O> *dest,
                           unsigned int dest_raw_pos=0);

    template<typename T, typename O, typename OP1, typename OP2>
    O MatrixSpanReduce1(const Basics::Matrix<T> *input,
                        const OP1 &inter_span_red_functor,
                        const OP2 &intra_span_red_functor,
                        const O &zero);

    template<typename T, typename O, typename OP1, typename OP2>
    void MatrixSpanReduceMinMax(const Basics::Matrix<T> *input,
                                const OP1 &inter_span_red_functor,
                                const OP2 &intra_span_red_functor,
                                const O &zero,
                                Basics::Matrix<int32_t> *which,
                                unsigned int which_raw_pos,
                                Basics::Matrix<O> *dest,
                                unsigned int dest_raw_pos=0);
    
    template<typename T, typename O, typename OP1, typename OP2>
    void MatrixSpanReduce2(const Basics::Matrix<T> *input1,
                           const Basics::Matrix<T> *input2,
                           const OP1 &inter_span_red_functor,
                           const OP2 &intra_span_red_functor,
                           const O &zero,
                           Basics::Matrix<O> *dest,
                           unsigned int dest_raw_pos=0);

    template<typename T, typename O, typename OP1, typename OP2>
    O MatrixSpanReduce2(const Basics::Matrix<T> *input1,
                        const Basics::Matrix<T> *input2,
                        const OP1 &inter_span_red_functor,
                        const OP2 &intra_span_red_functor,
                        const O &zero);
    
    template<typename T, typename OP>
    void MatrixScalarSumReduce1(const Basics::Matrix<T> *input,
                                const OP &scalar_red_functor,
                                Basics::Matrix<T> *dest,
                                unsigned int dest_raw_pos=0,
                                int N_th = DEFAULT_N_TH,
                                unsigned int SIZE_th = DEFAULT_SIZE_TH);
    
    template<typename T, typename OP>
    void MatrixSpanSumReduce1(const Basics::Matrix<T> *input,
                              const OP &inter_span_red_functor,
                              Basics::Matrix<T> *dest,
                              unsigned int dest_raw_pos=0,
                              int N_th = DEFAULT_N_TH,
                              unsigned int SIZE_th = DEFAULT_N_TH);

    template<typename T, typename OP>
    T MatrixSpanSumReduce1(const Basics::Matrix<T> *input,
                           const OP &inter_span_red_functor);
    
  } // namespace MatrixExt

} // namespace AprilMath

#undef DEFAULT_N_TH
#undef DEFAULT_SIZE_TH

#include "reduce_matrix.impl.h"

#endif // MAP_MATRIX_H
