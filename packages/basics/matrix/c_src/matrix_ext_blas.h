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
#ifndef MATRIX_EXT_BLAS_H
#define MATRIX_EXT_BLAS_H

namespace AprilMath {
  
  namespace MatrixExt {

    namespace BLAS {
      //////////////////// CBLAS MATH OPERATIONS ////////////////////

      /// SCOPY BLAS operation
      template <typename T>
      Basics::Matrix<T> *matCopy(Basics::Matrix<T> *obj,
                                 const Basics::Matrix<T> *other);

      /// SCOPY BLAS operation
      template <typename T>
      Basics::SparseMatrix<T> *matCopy(Basics::SparseMatrix<T> *obj,
                                       const Basics::SparseMatrix<T> *other);

      /// AXPY BLAS operation \f$ y = y + \alpha x \f$ being y=other and x=obj.
      template <typename T>
      Basics::Matrix<T> *matAxpy(Basics::Matrix<T> *obj, const T alpha,
                                 const Basics::Matrix<T> *other);

      /// AXPY BLAS operation \f$ y = y + \alpha x \f$ being y=other and x=obj.
      template <typename T>
      Basics::Matrix<T> *matAxpy(Basics::Matrix<T> *obj, T alpha,
                                 const Basics::SparseMatrix<T> *other);

      /// GEMM BLAS operation \f$ C = \alpha \text{op}(A) \times \text{op}(B) + \beta C \f$
      template <typename T>
      Basics::Matrix<T> *matGemm(Basics::Matrix<T> *C,
                                 CBLAS_TRANSPOSE trans_A,
                                 CBLAS_TRANSPOSE trans_B,
                                 const T alpha,
                                 const Basics::Matrix<T> *otherA,
                                 const Basics::Matrix<T> *otherB,
                                 T beta);

      /// MM Sparse BLAS operation \f$ C = \alpha \text{op}(A) \times \text{op}(B) + \beta C \f$
      template <typename T>
      Basics::Matrix<T> *matSparseMM(Basics::Matrix<T> *C,
                                     CBLAS_TRANSPOSE trans_A,
                                     CBLAS_TRANSPOSE trans_B,
                                     const T alpha,
                                     const Basics::SparseMatrix<T> *otherA,
                                     const Basics::Matrix<T> *otherB,
                                     T beta);

      /// GEMV BLAS operation \f$ Y = \alpha \text{op}(A) \times X + \beta Y \f$
      template <typename T>
      Basics::Matrix<T> *matGemv(Basics::Matrix<T> *Y,
                                 CBLAS_TRANSPOSE trans_A,
                                 const T alpha,
                                 const Basics::Matrix<T> *otherA,
                                 const Basics::Matrix<T> *otherX,
                                 const T beta);

      /// GEMV Sparse BLAS operation \f$ Y = \alpha \text{op}(A) \times X + \beta Y \f$
      template <typename T>
      Basics::Matrix<T> *matGemv(Basics::Matrix<T> *Y, CBLAS_TRANSPOSE trans_A,
                                 const T alpha,
                                 const Basics::SparseMatrix<T> *otherA,
                                 const Basics::Matrix<T> *otherX,
                                 const T beta);

      /// GER BLAS operation \f$ A = \alpha X \otimes Y + A \f$ (outer product)
      template <typename T>
      Basics::Matrix<T> *matGer(Basics::Matrix<T> *A,
                                const T alpha,
                                const Basics::Matrix<T> *otherX,
                                const Basics::Matrix<T> *otherY);

      /// DOT product BLAS operation \f$ X \circ Y \f$
      template <typename T>
      T matDot(const Basics::Matrix<T> *X, const Basics::Matrix<T> *Y);

      /// DOT product Sparse BLAS operation \f$ X \circ Y \f$
      template <typename T>
      T matDot(const Basics::Matrix<T> *X, const Basics::SparseMatrix<T> *Y);

      /// CBLAS norm2 operation \f$ ||A||_2 \f$
      template <typename T>
      float matNorm2(Basics::Matrix<T> *obj);

      /// CBLAS norm2 operation \f$ ||A||_2 \f$
      template <typename T>
      float matNorm2(Basics::SparseMatrix<T> *obj);
      
    } // namespace BLAS
  } // namespace MatrixExt
} // namespace AprilMath

#endif // MATRIX_EXT_BLAS_H
