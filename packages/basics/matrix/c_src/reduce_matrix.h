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

#ifndef MATRIX_H
#error "Requires \#include \"matrix.h\""
#endif

#define DEFAULT_N_TH 100
#define DEFAULT_SIZE_TH 100u

namespace basics {
  // forward declaration
  template <typename T>
  class Matrix;
}

namespace april_math {

  template<typename T, typename O, typename OP>
  O MatrixScalarReduce1(const basics::Matrix<T> *input,
                        const OP &scalar_red_functor,
                        const O &zero,
                        basics::Matrix<T> *dest = 0,
                        unsigned int dest_raw_pos = 0);
  
  template<typename T, typename O, typename OP1, typename OP2>
  O MatrixSpanReduce1(const basics::Matrix<T> *input,
                      const OP1 &inter_span_red_functor,
                      const OP2 &intra_span_red_functor,
                      const O &zero,
                      basics::Matrix<T> *dest = 0,
                      unsigned int dest_raw_pos = 0);

  template<typename T, typename O, typename OP>
  O MatrixScalarSumReduce1(const basics::Matrix<T> *input,
                           const OP &scalar_red_functor,
                           basics::Matrix<T> *dest,
                           unsigned int dest_raw_pos,
                           int N_th = DEFAULT_N_TH,
                           unsigned int SIZE_th = DEFAULT_SIZE_TH);

  template<typename T, typename O, typename OP>
  O MatrixSpanSumReduce1(const basics::Matrix<T> *input,
                         const OP &inter_span_red_functor,
                         basics::Matrix<T> *dest,
                         unsigned int dest_raw_pos,
                         int N_th = DEFAULT_N_TH,
                         unsigned int SIZE_th = DEFAULT_N_TH);

} // namespace april_math

#undef DEFAULT_N_TH
#undef DEFAULT_SIZE_TH

#endif // MAP_MATRIX_H
