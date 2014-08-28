/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014,Francisco Zamora-Martinez
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
#include "map_matrix.impl.cu"
#include "matrixFloat.h"

#define INSTANTIATE_MATRIX_SCALAR_MAP1(TYPE,FUNCTOR)                    \
  template basics::Matrix<TYPE> *MatrixScalarMap1<TYPE,TYPE>(const basics::Matrix<TYPE> *, \
                                                             basics::Matrix<TYPE> *, \
                                                             const FUNCTOR &, \
                                                             const int, \
                                                             const unsigned int)

#define INSTANTIATE_MATRIX_SCALAR_MAP1_O(TYPE,OUTPUT,FUNCTOR)           \
  template basics::Matrix<TYPE> *MatrixScalarMap1<TYPE,OUTPUT>(const basics::Matrix<TYPE> *, \
                                                               basics::Matrix<OUTPUT> *, \
                                                               const FUNCTOR &, \
                                                               const int, \
                                                               const unsigned int)

#define INSTANTIATE_MATRIX_SCALAR_MAP2(TYPE,FUNCTOR)                    \
  template basics::Matrix<TYPE> *MatrixScalarMap2<TYPE,TYPE,TYPE>(const basics::Matrix<TYPE> *, \
                                                                  const basics::Matrix<TYPE> *, \
                                                                  basics::Matrix<TYPE> *, \
                                                                  const FUNCTOR &, \
                                                                  const int, \
                                                                  const unsigned int)

namespace april_math {  

  INSTANTIATE_MATRIX_SCALAR_MAP1(float, m_float_unary_float_map_t);
  INSTANTIATE_MATRIX_SCALAR_MAP1(double, m_double_unary_double_map_t);
  INSTANTIATE_MATRIX_SCALAR_MAP1(ComplexF, m_complexf_unary_complexf_map_t);
  INSTANTIATE_MATRIX_SCALAR_MAP1_O(float, double, m_float_unary_double_map_t);
  INSTANTIATE_MATRIX_SCALAR_MAP1_O(float, ComplexF, m_float_unary_complexf_map_t);

} // namespace april_math
