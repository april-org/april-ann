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
  
    template<typename T, typename OP>
    Basics::Matrix<T> * MatrixScalarReduceOverDimension(Basics::Matrix<T> *input,
                                                        int dim,
                                                        const OP &scalar_red_functor,
                                                        const T &zero,
                                                        Basics::Matrix<T> *dest = 0);
    
    template<typename T, typename OP1, typename OP2>
    Basics::Matrix<T> * MatrixSpanReduceOverDimension(Basics::Matrix<T> *input,
                                                      int dim,
                                                      const OP1 &inter_span_red_functor,
                                                      const OP2 &intra_span_red_functor,
                                                      const T &zero,
                                                      Basics::Matrix<T> *dest = 0);
    
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

    template<typename T, typename OP>
    T MatrixScalarReduce1(const Basics::Matrix<T> *input,
                          const OP &scalar_red_functor,
                          const T &zero,
                          Basics::Matrix<T> *dest = 0,
                          unsigned int dest_raw_pos = 0);
  
    template<typename T, typename O, typename OP1, typename OP2>
    O MatrixSpanReduce1(const Basics::Matrix<T> *input,
                        const OP1 &inter_span_red_functor,
                        const OP2 &intra_span_red_functor,
                        const O &zero,
                        Basics::Matrix<O> *dest = 0,
                        unsigned int dest_raw_pos = 0);

    template<typename T, typename O, typename OP1, typename OP2>
    O MatrixSpanReduceMinMax(const Basics::Matrix<T> *input,
                             const OP1 &inter_span_red_functor,
                             const OP2 &intra_span_red_functor,
                             const O &zero,
                             Basics::Matrix<int32_t> *which,
                             unsigned int which_raw_pos,
                             Basics::Matrix<O> *dest,
                             unsigned int dest_raw_pos);
        
    template<typename T, typename O, typename OP1, typename OP2>
    O MatrixSpanReduce2(const Basics::Matrix<T> *input1,
                        const Basics::Matrix<T> *input2,
                        const OP1 &inter_span_red_functor,
                        const OP2 &intra_span_red_functor,
                        const O &zero,
                        Basics::Matrix<O> *dest = 0,
                        unsigned int dest_raw_pos = 0);

    template<typename T, typename OP>
    T MatrixScalarSumReduce1(const Basics::Matrix<T> *input,
                             const OP &scalar_red_functor,
                             Basics::Matrix<T> *dest = 0,
                             unsigned int dest_raw_pos = 0,
                             int N_th = DEFAULT_N_TH,
                             unsigned int SIZE_th = DEFAULT_SIZE_TH);

    template<typename T, typename OP>
    T MatrixSpanSumReduce1(const Basics::Matrix<T> *input,
                           const OP &inter_span_red_functor,
                           Basics::Matrix<T> *dest = 0,
                           unsigned int dest_raw_pos = 0,
                           int N_th = DEFAULT_N_TH,
                           unsigned int SIZE_th = DEFAULT_N_TH);

  } // namespace MatrixExt

} // namespace AprilMath

#undef DEFAULT_N_TH
#undef DEFAULT_SIZE_TH

#include "reduce_matrix.impl.h"

#endif // MAP_MATRIX_H
