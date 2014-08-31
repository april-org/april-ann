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

#define INSTANTIATE_SPARSE_MATRIX_SCALAR_MAP1(TYPE,FUNCTOR)     \
  template Matrix<TYPE> *MatrixScalarMap1(const SparseMatrix<TYPE> *, \
                                          SparseMatrix<TYPE> *,             \
                                          const FUNCTOR &);

#define INSTANTIATE_SPARSE_MATRIX_SCALAR_MAP2(TYPE,FUNCTOR)            \
  template Matrix<TYPE> *MatrixScalarMap2(const SparseMatrix<TYPE> *, \
                                          const SparseMatrix<TYPE> *,   \
                                          SparseMatrix<TYPE> *,             \
                                          const FUNCTOR &);

namespace AprilMath {  

  namespace MatrixExt {
    INSTANTIATE_SPARSE_MATRIX_SCALAR_MAP1(float, m_unary_float_map_t);
    INSTANTIATE_SPARSE_MATRIX_SCALAR_MAP1(double, m_unary_double_map_t);
    INSTANTIATE_SPARSE_MATRIX_SCALAR_MAP1(ComplexF, m_unary_complexf_map_t);
  }
  
} // namespace AprilMath
