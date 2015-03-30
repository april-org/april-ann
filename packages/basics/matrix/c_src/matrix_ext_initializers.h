/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
 * Copyright 2012, Salvador España-Boquera
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
#ifndef MATRIX_H
#include "matrix.h"
#include "sparse_matrix.h"
#endif
#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

namespace AprilMath {

  namespace MatrixExt {

    /**
     * @brief Initializers for Matrix instances.
     *
     * This operations initialize the given Basics::Matrix changing it
     * <b>in-place</b>.
     *
     * @see AprilMath::MatrixExt
     */    
    namespace Initializers {
      
      /// Initialize all the matrix elements to the given value.
      template<typename T>
      Basics::Matrix<T> *matFill(Basics::Matrix<T> *obj, const T value);

      /// Initialize all the matrix elements to the given value.
      template<typename T>
      Basics::SparseMatrix<T> *matFill(Basics::SparseMatrix<T> *obj, const T value);

      /// Initialize all the matrix elements to zero.
      template<typename T>
      Basics::Matrix<T> *matZeros(Basics::Matrix<T> *obj);

      /// Initialize all the matrix elements to zero.
      template<typename T>
      Basics::SparseMatrix<T> *matZeros(Basics::SparseMatrix<T> *obj);

      /// Initialize all the matrix elements to one.
      template<typename T>
      Basics::Matrix<T> *matOnes(Basics::Matrix<T> *obj);

      /// Initialize all the matrix elements to one.
      template<typename T>
      Basics::SparseMatrix<T> *matOnes(Basics::SparseMatrix<T> *obj);

      /// Initialize the matrix diagonal elements to the given value.
      template <typename T>
      Basics::Matrix<T> *matDiag(Basics::Matrix<T> *obj, const T value);

    } // namespace Initializers
    
  } // namespace MatrixExt
} // namespace AprilMath

#endif // MATRIX_OPERATIONS_H
