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
#ifndef MAP_SPARSE_MATRIX_H
#define MAP_SPARSE_MATRIX_H

#ifndef SPARSE_MATRIX_H
#error "Requires \#include \"sparse_matrix.h\""
#endif

namespace basics {
  // forward declaration
  template <typename T>
  class SparseMatrix;
}

namespace april_math {
  
  template<typename T, typename O, typename OP>
  basics::SparseMatrix<O> *SparseMatrixScalarMap1(const basics::SparseMatrix<T> *input,
                                                  const OP &functor,
                                                  basics::SparseMatrix<T> *dest = 0);
  
  template<typename T, typename O, typename OP>
  basics::SparseMatrix<O> *SparseMatrixScalarMap2(const basics::SparseMatrix<T> *input1,
                                                  const basics::SparseMatrix<T> *input2,
                                                  const OP &functor,
                                                  basics::SparseMatrix<T> *dest = 0);
} // namespace april_math

#endif // MAP_SPARSE_MATRIX_H
