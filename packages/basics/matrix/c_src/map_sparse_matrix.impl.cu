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
#ifndef MAP_SPARSE_MATRIX_IMPL_CU
#define MAP_SPARSE_MATRIX_IMPL_CU

// must be defined here
#include "matrix.h"

#include "map_sparse_matrix.h"
#include "map_template.h"

namespace AprilMath {

  namespace MatrixExt {

    template<typename T, typename O, typename OP>
    Basics::SparseMatrix<O> *SparseMatrixScalarMap1(const Basics::SparseMatrix<T> *input,
                                                    const OP &functor,
                                                    Basics::SparseMatrix<O> *dest) {
      if (dest == 0) dest = input->clone();
      april_assert(input != 0 && dest != 0);
      // TODO: check coordinates
      if (input->size() != dest->size() || !input->sameDim(dest) ||
          input->nonZeroSize() != dest->nonZeroSize()) {
        ERROR_EXIT(128, "Incorrect matrix sizes or dimensions\n");
      }
      genericMap1Call(input->nonZeroSize(),
                      input->getRawValuesAccess(), 1u, 0u,
                      dest->getRawValuesAccess(), 1u, 0u,
                      input->getCudaFlag(),
                      functor);
      return dest;
    }
  
    template<typename T, typename O, typename OP>
    Basics::SparseMatrix<O> *SparseMatrixScalarMap2(const Basics::SparseMatrix<T> *input1,
                                                    const Basics::SparseMatrix<T> *input2,
                                                    const OP &functor,
                                                    Basics::SparseMatrix<O> *dest) {
      if (dest == 0) dest = input1->clone();
      // TODO: check coordinates
      if (input1->size() != dest->size() || !input1->sameDim(dest) ||
          input1->nonZeroSize() != dest->nonZeroSize() ||
          input2->size() != dest->size() || !input2->sameDim(dest)) {
        ERROR_EXIT(128, "Incorrect matrix sizes or dimensions\n");
      }
      genericMap2Call(input1->nonZeroSize(),
                      input1->getRawValuesAccess(), 1u, 0u,
                      input2->getRawValuesAccess(), 1u, 0u,
                      dest->getRawValuesAccess(), 1u, 0u,
                      input1->getCudaFlag(),
                      functor);
      return dest;
    }

  } // namespace MatrixExt
  
} // namespace AprilMath

#endif // MAP_SPARSE_MATRIX_IMPL_CU
