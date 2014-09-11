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
  
  namespace MatrixExt {
  

    template<typename T, typename O, typename OP1, typename OP2>
      Basics::Matrix<O> * MatrixScalarReduce1OverDimension(const Basics::Matrix<T> *input,
                                                           int dim,
                                                           const OP1 &scalar_red_functor,
                                                           const OP2 &partials_red_functor,
                                                           const O &zero,
                                                           Basics::Matrix<O> *dest=0,
                                                           bool set_dest_to_zero=true);
    
    template<typename T, typename O, typename OP1, typename OP2>
    Basics::Matrix<O> * MatrixSpanReduce1OverDimension(const Basics::Matrix<T> *input,
                                                       int dim,
                                                       const OP1 &inter_span_red_functor,
                                                       const OP2 &intra_span_red_functor,
                                                       const O &zero,
                                                       Basics::Matrix<O> *dest=0,
                                                       bool set_dest_to_zero=true);

    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    Basics::Matrix<O> * MatrixScalarReduce2OverDimension(const Basics::Matrix<T1> *input1,
                                                         const Basics::Matrix<T2> *input2,
                                                         int dim,
                                                         const OP1 &scalar_red_functor,
                                                         const OP2 &partials_red_functor,
                                                         const O &zero,
                                                         Basics::Matrix<O> *dest=0,
                                                         bool set_dest_to_zero=true);
    
    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    Basics::Matrix<O> * MatrixSpanReduce2OverDimension(const Basics::Matrix<T1> *input1,
                                                       const Basics::Matrix<T2> *input2,
                                                       int dim,
                                                       const OP1 &inter_span_red_functor,
                                                       const OP2 &intra_span_red_functor,
                                                       const O &zero,
                                                       Basics::Matrix<O> *dest=0,
                                                       bool set_dest_to_zero=true);
    
    template<typename T, typename OP>
    Basics::Matrix<T> * MatrixScalarReduceMinMaxOverDimension(const Basics::Matrix<T> *input,
                                                              int dim,
                                                              const OP &scalar_red_functor,
                                                              const T &zero,
                                                              Basics::Matrix<int32_t> *which=0,
                                                              Basics::Matrix<T> *dest=0,
                                                              bool set_dest_to_zero=true);
    
    template<typename T, typename OP1, typename OP2>
    Basics::Matrix<T> * MatrixSpanReduceMinMaxOverDimension(const Basics::Matrix<T> *input,
                                                            int dim,
                                                            const OP1 &inter_span_red_functor,
                                                            const OP2 &intra_span_red_functor,
                                                            const T &zero,
                                                            Basics::Matrix<int32_t> *which=0,
                                                            Basics::Matrix<T> *dest=0,
                                                            bool set_dest_to_zero=true);
    
    template<typename T, typename O, typename OP1, typename OP2>
    void MatrixScalarReduce1(const Basics::Matrix<T> *input,
                             const OP1 &scalar_red_functor,
                             const OP2 &partials_red_functor,
                             const O &zero,
                             AprilMath::GPUMirroredMemoryBlock<O> *dest,
                             unsigned int dest_raw_pos=0,
                             bool set_dest_to_zero=true);
    
    /**
     * @brief Reduces all the elements of a given Basics::Matrix @c input storing
     * the result in the given @c dest_raw_pos position of @c dest
     * AprilMath::GPUMirroredMemoryBlock.
     *
     * @tparam T - The reduction input type.
     * @tparam O - The reduction output type.
     * @tparam OP1 - The functor for reduction of input values. It has to implement
     * the <tt>APRIL_CUDA_EXPORT void operator()(O &acc, const T &b) const</tt> which
     * reduces the given reduction output accumulator @c acc with the given input
     * value @c b and stores the result in @c acc.
     * @tparam Op2 - The functor for reduction of intermediate output values. It
     * has to implement the <tt>APRIL_CUDA_EXPORT void operator()(O &acc, const O &b) const</tt>
     * which reduces a given reduction output accumulator @c acc with the given output
     * reduction value @c b, storing its result in @c acc.
     * 
     * @param input -
     * @param scalar_red_functor -
     * @param partials_red_functor -
     * @param zero -
     * @param set_dest_to_zero -
     *
     * @return The output reduction value.
     *
     * @note This function is a wrapper over
     * AprilMath::MatrixExt::MatrixScalarReduce1 passing @c dest parameter as a
     * new allocated AprilMath::GPUMirroredMemoryBlock and @c dest_raw_pos=0 .
     *
     * @code
     * // The example uses the AprilMath::Function::r_add reduction, which uses
     * // operator+= to reduce two values, the input type T and the reduction type O.
     * #include "reduce_matrix.h"
     * namespace MyNameSpace {
     *   struct MeanReduceResult {
     *     float sum;
     *     int N;
     *     MeanReduceResult() : sum(0.0f), N(0) { }
     *     float getMean() const { return sum/N; }
     *     // Called from AprilMath::Functors::r_add<MeanReduceResult,float>,
     *     APRIL_CUDA_EXPORT MeanReduceResult &operator+=(const float &b) const {
     *       acc.N++;
     *       acc.sum += b;
     *       return *this;
     *     }
     *     // Called from AprilMath::Functors::r_add<MeanReduceResult,MeanReduceResult>
     *     APRIL_CUDA_EXPORT MeanReduceResult &operator+=(const MeanReduceResult &b) const {
     *       acc.sum += b.sum;
     *       acc.N   += b.N;
     *       return *this;
     *     }
     *   };
     * }
     * SPECIALIZE_CUDA_SHARED_MEMORY(MyNameSpace::MeanReduceResult,MeanReduceResult);
     * // a matrix of 1 dimension and 1 element
     * MyNameSpace::MeanReduceResult result =
     *   MatrixScalarReduce1(my_matrix_float, // a pointer to a MatrixFloat instance
     *                       AprilMath::Functors::r_add<MyNameSpace::MeanReduceResult,float>,
     *                       AprilMath::Functors::r_add<MyNameSpace::MeanReduceResult,MyNameSpace::MeanReduceResult>,
     *                       MyNameSpace::MeanReduceResult());
     * printf("Mean result: %f\n", result.getMean());
     * @endcode
     */
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
                           AprilMath::GPUMirroredMemoryBlock<O> *dest,
                           unsigned int dest_raw_pos=0,
                           bool set_dest_to_zero=true);

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
                                AprilMath::GPUMirroredMemoryBlock<int32_t> *which,
                                unsigned int which_raw_pos,
                                AprilMath::GPUMirroredMemoryBlock<O> *dest,
                                unsigned int dest_raw_pos=0,
                                bool set_dest_to_zero=true);
    
    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    void MatrixSpanReduce2(const Basics::Matrix<T1> *input1,
                           const Basics::Matrix<T2> *input2,
                           const OP1 &inter_span_red_functor,
                           const OP2 &intra_span_red_functor,
                           const O &zero,
                           AprilMath::GPUMirroredMemoryBlock<O> *dest,
                           unsigned int dest_raw_pos=0,
                           bool set_dest_to_zero=true);

    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    O MatrixSpanReduce2(const Basics::Matrix<T1> *input1,
                        const Basics::Matrix<T2> *input2,
                        const OP1 &inter_span_red_functor,
                        const OP2 &intra_span_red_functor,
                        const O &zero);

    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    void MatrixScalarReduce2(const Basics::Matrix<T1> *input1,
                             const Basics::Matrix<T2> *input2,
                             const OP1 &inter_span_red_functor,
                             const OP2 &intra_span_red_functor,
                             const O &zero,
                             AprilMath::GPUMirroredMemoryBlock<O> *dest,
                             unsigned int dest_raw_pos=0,
                             bool set_dest_to_zero=true);
    
    template<typename T1, typename T2, typename O, typename OP1, typename OP2>
    O MatrixScalarReduce2(const Basics::Matrix<T1> *input1,
                          const Basics::Matrix<T2> *input2,
                          const OP1 &inter_span_red_functor,
                          const OP2 &intra_span_red_functor,
                          const O &zero);
    
    template<typename T, typename OP>
    void MatrixScalarSumReduce1(const Basics::Matrix<T> *input,
                                const OP &scalar_red_functor,
                                AprilMath::GPUMirroredMemoryBlock<T> *dest,
                                unsigned int dest_raw_pos=0,
                                bool set_dest_to_zero=true,
                                int N_th = DEFAULT_N_TH,
                                unsigned int SIZE_th = DEFAULT_SIZE_TH);
    
    template<typename T, typename OP>
    void MatrixSpanSumReduce1(const Basics::Matrix<T> *input,
                              const OP &inter_span_red_functor,
                              AprilMath::GPUMirroredMemoryBlock<T> *dest,
                              unsigned int dest_raw_pos=0,
                              bool set_dest_to_zero=true,
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
