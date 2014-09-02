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

#include "cmath_overloads.h"
#include "map_sparse_matrix.impl.cu"

#define INSTANTIATE_SPARSE_MATRIX_SCALAR_MAP1(TYPE,OP)     \
  template Basics::SparseMatrix< TYPE > *SparseMatrixScalarMap1< TYPE, TYPE, OP >(const Basics::SparseMatrix< TYPE > *, \
                                                                                  const OP &, \
                                                                                  Basics::SparseMatrix< TYPE > *)

#define INSTANTIATE_SPARSE_MATRIX_SCALAR_MAP2(TYPE,OP)                  \
  template Basics::SparseMatrix< TYPE > *SparseMatrixScalarMap2< TYPE, TYPE, OP >(const Basics::SparseMatrix< TYPE > *, \
                                                                                  const Basics::SparseMatrix< TYPE > *, \
                                                                                  const OP &, \
                                                                                  Basics::SparseMatrix< TYPE > *)

namespace AprilMath {  
  namespace MatrixExt {
    
    INSTANTIATE_SPARSE_MATRIX_SCALAR_MAP1(float, m_float_unary_float_map_t);
    INSTANTIATE_SPARSE_MATRIX_SCALAR_MAP1(double, m_double_unary_double_map_t);
    INSTANTIATE_SPARSE_MATRIX_SCALAR_MAP1(ComplexF, m_complexf_unary_complexf_map_t);
    
  }
} // namespace AprilMath
