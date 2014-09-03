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

#include "cmath_overloads.h"
#include "map_matrix.impl.h"

#define INSTANTIATE_MATRIX_SCALAR_MAP1(TYPE,OUTPUT,OP)                  \
  template Basics::Matrix< OUTPUT > *MatrixScalarMap1< TYPE, OUTPUT, OP >(const Basics::Matrix< TYPE > *, \
                                                                          const OP &, \
                                                                          Basics::Matrix< OUTPUT > *, \
                                                                          const int, \
                                                                          const unsigned int)
    
#define INSTANTIATE_MATRIX_SCALAR_MAP2(TYPE,OUTPUT,OP)                  \
  template Basics::Matrix< OUTPUT > *MatrixScalarMap2< TYPE, TYPE, OUTPUT, OP >(const Basics::Matrix< TYPE > *, \
                                                                                const Basics::Matrix< TYPE > *, \
                                                                                const OP &, \
                                                                                Basics::Matrix< OUTPUT > *, \
                                                                                const int, \
                                                                                const unsigned int)

#define INSTANTIATE_MATRIX_SPAN_MAP1(TYPE,OUTPUT,OP)                    \
  template Basics::Matrix< OUTPUT > *MatrixSpanMap1< TYPE, OUTPUT, OP >(const Basics::Matrix< TYPE > *, \
                                                                        const OP &, \
                                                                        Basics::Matrix< OUTPUT > *, \
                                                                        const int, \
                                                                        const unsigned int)

namespace AprilMath {  
  namespace MatrixExt {

    INSTANTIATE_MATRIX_SCALAR_MAP1(char, char, m_curried_fill<char>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(int32_t, int32_t, m_curried_fill<int32_t>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_fill<float>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(double, double, m_curried_fill<double>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(ComplexF, ComplexF, m_curried_fill<ComplexF>);
    
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_float_unary_float_map_t);
    INSTANTIATE_MATRIX_SCALAR_MAP1(double, double, m_double_unary_double_map_t);
    INSTANTIATE_MATRIX_SCALAR_MAP1(ComplexF, ComplexF, m_complexf_unary_complexf_map_t);

    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_pow<float>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_clamp<float>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_mul<float>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_lt<float>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_gt<float>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_eq_nan<float>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_eq<float>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_neq_nan<float>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_neq<float>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_add<float>);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, float, m_curried_div<float>);
    
    INSTANTIATE_MATRIX_SCALAR_MAP1(ComplexF, ComplexF, m_curried_add<ComplexF>);

    INSTANTIATE_MATRIX_SCALAR_MAP1(float, double, m_float_unary_double_map_t);
    INSTANTIATE_MATRIX_SCALAR_MAP1(float, ComplexF, m_float_unary_complexf_map_t);

    INSTANTIATE_MATRIX_SCALAR_MAP2(float, float, m_float_binary_float_map_t);
    INSTANTIATE_MATRIX_SCALAR_MAP2(double, double, m_double_binary_double_map_t);
    INSTANTIATE_MATRIX_SCALAR_MAP2(ComplexF, ComplexF, m_complexf_binary_complexf_map_t);
    
    INSTANTIATE_MATRIX_SPAN_MAP1(float, float, float_float_span_map1_t);
    INSTANTIATE_MATRIX_SPAN_MAP1(double, double, double_double_span_map1_t);
    INSTANTIATE_MATRIX_SPAN_MAP1(ComplexF, ComplexF, complexf_complexf_span_map1_t);
    
    INSTANTIATE_MATRIX_SPAN_MAP1(float, float, CurriedAxpy<float>);
    INSTANTIATE_MATRIX_SPAN_MAP1(float, float, CurriedScal<float>);

    INSTANTIATE_MATRIX_SPAN_MAP1(ComplexF, ComplexF, CurriedAxpy<ComplexF>);
    INSTANTIATE_MATRIX_SPAN_MAP1(ComplexF, ComplexF, CurriedScal<ComplexF>);

  } // namespace MatrixExt
} // namespace AprilMath
