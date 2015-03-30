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
#include "matrix_ext.h"
#ifndef MATRIX_EXT_LAPACK_H
#define MATRIX_EXT_LAPACK_H

namespace AprilMath {
  
  namespace MatrixExt {

    namespace LAPACK {
      
      //////////////////// LAPACK MATH OPERATIONS ////////////////////

      // FIXME: using WRAPPER for generalized CULA, LAPACK, float and complex
      // numbers

      /// Inverts the given matrix.
      Basics::Matrix<float> *matInv(const Basics::Matrix<float> *obj);

      /// Computes SVD decomposition of the given matrix. S, U, VT are given by reference.
      void matSVD(const Basics::Matrix<float> *obj,
                  Basics::Matrix<float> **U, Basics::SparseMatrix<float> **S,
                  Basics::Matrix<float> **VT);

      // FROM: http://www.r-bloggers.com/matrix-determinant-with-the-lapack-routine-dspsv/
      /// Computes the logarithm of the determinant. The sign is returned by reference.
      AprilUtils::log_float matLogDeterminant(const Basics::Matrix<float> *obj,
                                              float &sign);

      // FROM: http://www.r-bloggers.com/matrix-determinant-with-the-lapack-routine-dspsv/
      /// Computes the determinant.
      double matDeterminant(const Basics::Matrix<float> *obj);

      /**
       * @brief Compute the Cholesky factorization of a real symmetric positive
       * definite matrix A.
       *
       * The factorization has the form:
       * A = U**T *	U,  if UPLO = 'U', or
       * A = L  * L**T,  if      UPLO = 'L',
       * where U is an upper triangular matrix and L is lower triangular.
       *
       * @param obj - The input matrix.
       *
       * @param uplo - A char with 'U' or 'L'.
       */
      Basics::Matrix<float> * matCholesky(const Basics::Matrix<float> *obj,
                                          char uplo);
      
    } // namespace LAPACK
    
  } // namespace MatrixExt
} // namespace AprilMath

#endif // MATRIX_EXT_LAPACK_H
