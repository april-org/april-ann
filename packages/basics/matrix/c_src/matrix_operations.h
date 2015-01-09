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

  /**
   * @brief Linear algebra routines and other math operations for matrices.
   *
   * By default, the zero value must be T(). Additionally, T(0.0f) and T(1.0f)
   * and T(-1.0f) and T(-nan) constructors must be available with correct math
   * values. In case of char buffer or integer matrices these constructors are
   * needed but not operational because math methods are forbidden for these
   * data types.
   */
  namespace MatrixExt {

    ///////// BASIC MAP FUNCTIONS /////////

    /**
     * @brief Valid operations over Matrix instances. They are wrappers over
     * generic functions defined at AprilMath::MatrixExt for map or reduce
     * operations using Matrix instances.
     *
     * This operations have been implemented receiving two Basics::Matrix
     * objects, the source @c obj and the destination @c dest . In case
     * @c dest=0 , @c obj will be taken as destination, computing the operation
     * <b>in-place</b>.
     *
     * @note Most of these operations have been implemented using functions
     * defined in AprilMath and AprilMath::Functors.
     *
     * @see AprilMath::MatrixExt
     */
    namespace Operations {

      /**
       * @brief Computes \f$ p \log p \f$ for every element of the given matrix.
       *
       * This function takes the values of the given matrix as probabilities,
       * so they have to be in the range \f$ [0,1] \f$.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       *
       * @note This operation has been optimized considering properly the case
       * of \f$ p=0 \f$.
       *
       * @see Operations namespace.
       */
      template <typename T>
      Basics::Matrix<T> *matPlogp(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ \log x \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matLog(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ \log (1 + x) \f$ for every element of the given
       * matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matLog1p(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ \exp x \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matExp(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ \sqrt x \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matSqrt(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ \sqrt x \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matSqrt(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);

      /**
       * @brief Computes \f$ x^v \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matPow(Basics::Matrix<T> *obj, const T &value,
                                Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ x^y \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matPow(Basics::SparseMatrix<T> *obj, const T &value,
                                      Basics::SparseMatrix<T> *dest=0);

      /**
       * @brief Computes \f$ tan(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matTan(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ tan(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matTan(Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest=0);

      /**
       * @brief Computes \f$ tanh(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matTanh(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ tanh(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matTanh(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);

      /**
       * @brief Computes \f$ atan(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matAtan(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ atan(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matAtan(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);

      /**
       * @brief Computes \f$ atanh(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matAtanh(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ atanh(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matAtanh(Basics::SparseMatrix<T> *obj,
                                        Basics::SparseMatrix<T> *dest=0);

      /**
       * @brief Computes \f$ sin(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matSin(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ sin(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matSin(Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest=0);

      /**
       * @brief Computes \f$ sinh(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matSinh(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ sinh(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matSinh(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);

      /**
       * @brief Computes \f$ asin(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matAsin(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ asin(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matAsin(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);

      /**
       * @brief Computes \f$ asinh(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matAsinh(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ asinh(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matAsinh(Basics::SparseMatrix<T> *obj,
                                        Basics::SparseMatrix<T> *dest=0);

      /**
       * @brief Computes \f$ cos(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matCos(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ cosh(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matCosh(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ acos(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matAcos(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ acosh(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matAcosh(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ abs(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matAbs(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ abs(x) \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matAbs(Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest=0);

      /**
       * @brief Computes \f$ 1 - x \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matComplement(Basics::Matrix<T> *obj,
                                       Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ \frac{x}{|x|} \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matSign(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ \frac{x}{|x|} \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matSign(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);

      ////////////////// OTHER MAP FUNCTIONS //////////////////

      /**
       * @brief Computes \f$ \max(l, \min(u, x)) \f$ for every element of the
       * given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template<typename T>
      Basics::Matrix<T> *matClamp(Basics::Matrix<T> *obj,
                                  const T lower, const T upper,
                                  Basics::Matrix<T> *dest=0);

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

      /**
       * @brief Computes \f$ v \cdot x \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matScal(Basics::Matrix<T> *obj, const T value,
                                 Basics::Matrix<T> *dest = 0);

      /**
       * @brief Computes \f$ v \cdot x \f$ for every element of the given matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matScal(Basics::SparseMatrix<T> *obj,
                                       const T value,
                                       Basics::SparseMatrix<T> *dest = 0);

      /// CBLAS norm2 operation \f$ ||A||_2 \f$
      template <typename T>
      float matNorm2(Basics::Matrix<T> *obj);

      /// CBLAS norm2 operation \f$ ||A||_2 \f$
      template <typename T>
      float matNorm2(Basics::SparseMatrix<T> *obj);

      /////////////////// MAX MIN REDUCTIONS ///////////////////

      // Min and max over given dimension, be careful, argmin and argmax matrices
      // contains the min/max index at the given dimension, but starting in 1 (not
      // in 0)

      /**
       * @brief Min reduction over a given dimension number.
       *
       * @param obj - The source matrix object.
       * @param dim - The dimension (a value between 0 and @c obj->getNumDim()
       * @param dest - The destination matrix, if it is 0, a new matrix would be allocated.
       * @param argmin - A matrix with the position of the minimums, if not given, it won't be used.
       *
       * @result The given @c dest argument or a new allocated matrix with the min values.
       */
      template <typename T>
      Basics::Matrix<T> *matMin(Basics::Matrix<T> *obj,
                                int dim,
                                Basics::Matrix<T> *dest=0,
                                Basics::Matrix<int32_t> *argmin=0);

      // TODO: use a wrapper for GPU/CPU
      /**
       * @brief Min reduction over a given dimension number.
       *
       * @param obj - The source matrix object.
       * @param dim - The dimension (a value between 0 and @c obj->getNumDim()
       * @param dest - The destination matrix, if it is 0, a new matrix would be allocated.
       * @param argmin - A matrix with the position of the minimums, if not given, it won't be used.
       *
       * @result The given @c dest argument or a new allocated matrix with the min values.
       */
      template <typename T>
      Basics::Matrix<T> *matMin(Basics::SparseMatrix<T> *obj, int dim,
                                Basics::Matrix<T> *dest=0,
                                Basics::Matrix<int32_t> *argmin=0);

      /**
       * @brief Max reduction over a given dimension number.
       *
       * @param obj - The source matrix object.
       * @param dim - The dimension (a value between 0 and @c obj->getNumDim()
       * @param dest - The destination matrix, if it is 0, a new matrix would be allocated.
       * @param argmax - A matrix with the position of the maximums, if not given, it won't be used.
       *
       * @result The given @c dest argument or a new allocated matrix with the min values.
       */
      template <typename T>
      Basics::Matrix<T> *matMax(Basics::Matrix<T> *obj,
                                int dim,
                                Basics::Matrix<T> *dest=0,
                                Basics::Matrix<int32_t> *argmax=0);

      // TODO: use a wrapper for GPU/CPU
      /**
       * @brief Max reduction over a given dimension number.
       *
       * @param obj - The source matrix object.
       * @param dim - The dimension (a value between 0 and @c obj->getNumDim()
       * @param dest - The destination matrix, if it is 0, a new matrix would be allocated.
       * @param argmax - A matrix with the position of the maximums, if not given, it won't be used.
       *
       * @result The given @c dest argument or a new allocated matrix with the min values.
       */
      template <typename T>
      Basics::Matrix<T> *matMax(Basics::SparseMatrix<T> *obj,
                                int dim, Basics::Matrix<T> *dest=0,
                                Basics::Matrix<int32_t> *argmax=0);

      // FIXME: using WRAPPER
      template <typename T>
      T matMin(const Basics::Matrix<T> *obj, int &arg_min, int &arg_min_raw_pos);

      // FIXME: using WRAPPER
      template <typename T>
      T matMin(const Basics::SparseMatrix<T> *obj, int &c0, int &c1);

      // FIXME: using WRAPPER
      /**
       * @brief Max reduction over the whole matrix.
       *
       * @param obj - The source matrix object.
       * @param arg_max - The index value of the element in the matrix.
       * @param arg_max_raw_pos - The index value of the element in the memory block.
       *
       * @result The max value.
       */
      template<typename T>
      T matMax(const Basics::Matrix<T> *obj, int &arg_max, int &arg_max_raw_pos);

      // FIXME: using WRAPPER
      /**
       * @brief Max reduction over the whole matrix.
       *
       * @param obj - The source matrix object.
       * @param c0 - The index in dimension 0.
       * @param c1 - The index in dimension 1.
       *
       * @result The max value.
       */
      template<typename T>
      T matMax(const Basics::SparseMatrix<T> *obj, int &c0, int &c1);

      // FIXME: using WRAPPER
      template<typename T>
      void matMinAndMax(const Basics::Matrix<T> *obj, T &min, T &max);

      template<typename T>
      void matMinAndMax(const Basics::SparseMatrix<T> *obj, T &min, T &max);

      template <typename T>
      Basics::Matrix<T> *matMaxSelDim(const Basics::Matrix<T> *obj,
                                      const int dim,
                                      Int32GPUMirroredMemoryBlock *raw_positions,
                                      const int shift,
                                      Basics::Matrix<T> *result=0);

      //////////////////// BOOLEAN CONDITIONS ////////////////////

      /* BOOLEAN CONDITIONS: this methods transforms the given matrix in a
         ZERO/ONE matrix, depending in the truth of the given condition */

      /**
       * @brief Compares every matrix element with the given value and updates
       * returned @c dest matrix with @c true or @c false depending in the
       * condition \f$ x < v \f$
       *
       * @note If the given @c dest argument is 0, this operation allocates a
       * new destination matrix.
       */
      template <typename T>
      Basics::Matrix<bool> *matLT(Basics::Matrix<T> *obj, const T &value,
                                  Basics::Matrix<bool> *dest=0);

      /**
       * @brief Compares two matrices in a component-wise fashion, updates
       * returned @c dest matrix with @c true or @c false depending in the
       * condition \f$ x < v \f$
       *
       * @note If the given @c dest argument is 0, this operation allocates a
       * new destination matrix.
       */
      template <typename T>
      Basics::Matrix<bool> *matLT(Basics::Matrix<T> *obj,
                                  const Basics::Matrix<T> *other,
                                  Basics::Matrix<bool> *dest=0);

      /**
       * @brief Compares every matrix element with the given value and updates
       * returned @c dest matrix with @c true or @c false depending in the
       * condition \f$ x > v \f$
       *
       * @note If the given @c dest argument is 0, this operation allocates a
       * new destination matrix.
       */
      template <typename T>
      Basics::Matrix<bool> *matGT(Basics::Matrix<T> *obj, const T &value,
                                  Basics::Matrix<bool> *dest=0);

      /**
       * @brief Compares two matrices in a component-wise fashion, updates
       * returned @c dest matrix with @c true or @c false depending in the
       * condition \f$ x > v \f$
       *
       * @note If the given @c dest argument is 0, this operation allocates a
       * new destination matrix.
       */
      template <typename T>
      Basics::Matrix<bool> *matGT(Basics::Matrix<T> *obj,
                                  const Basics::Matrix<T> *other,
                                  Basics::Matrix<bool> *dest=0);

      /**
       * @brief Compares every matrix element with the given value and updates
       * returned @c dest matrix with @c true or @c false depending in the
       * condition \f$ x = v \f$
       *
       * @note If the given @c dest argument is 0, this operation allocates a
       * new destination matrix.
       */
      template <typename T>
      Basics::Matrix<bool> *matEQ(Basics::Matrix<T> *obj, const T &value,
                                  Basics::Matrix<bool> *dest=0);

      /**
       * @brief Compares two matrices in a component-wise fashion, updates
       * returned @c dest matrix with @c true or @c false depending in the
       * condition \f$ x = v \f$
       *
       * @note If the given @c dest argument is 0, this operation allocates a
       * new destination matrix.
       */
      template <typename T>
      Basics::Matrix<bool> *matEQ(Basics::Matrix<T> *obj,
                                  const Basics::Matrix<T> *other,
                                  Basics::Matrix<bool> *dest=0);

      /**
       * @brief Compares every matrix element with the given value and updates
       * returned @c dest matrix with @c true or @c false depending in the
       * condition \f$ x \neq v \f$
       *
       * @note If the given @c dest argument is 0, this operation allocates a
       * new destination matrix.
       */
      template <typename T>
      Basics::Matrix<bool> *matNEQ(Basics::Matrix<T> *obj, const T &value,
                                   Basics::Matrix<bool> *dest=0);

      /**
       * @brief Compares two matrices in a component-wise fashion, updates
       * returned @c dest matrix with @c true or @c false depending in the
       * condition \f$ x \neq v \f$
       *
       * @note If the given @c dest argument is 0, this operation allocates a
       * new destination matrix.
       */
      template <typename T>
      Basics::Matrix<bool> *matNEQ(Basics::Matrix<T> *obj,
                                   const Basics::Matrix<T> *other,
                                   Basics::Matrix<bool> *dest=0);

      //////////////////// OTHER MATH OPERATIONS ////////////////////

      /**
       * @brief Returns the result of \f$ C = A + B \f$
       *
       * @note If the given @c c argument is 0, this operation allocates a
       * new destination matrix, otherwise uses the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matAddition(const Basics::Matrix<T> *a,
                                     const Basics::Matrix<T> *b,
                                     Basics::Matrix<T> *c = 0);

      /**
       * @brief Returns the result of \f$ C = A - B \f$
       *
       * @note If the given @c c argument is 0, this operation allocates a
       * new destination matrix, otherwise uses the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matSubstraction(const Basics::Matrix<T> *a,
                                         const Basics::Matrix<T> *b,
                                         Basics::Matrix<T> *c = 0);

      /**
       * @brief Returns the result of \f$ C = A \times B \f$
       *
       * @note If the given @c c argument is 0, this operation allocates a
       * new destination matrix, otherwise uses the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matMultiply(const Basics::Matrix<T> *a,
                                     const Basics::Matrix<T> *b,
                                     Basics::Matrix<T> *c = 0);

      /// Returns the sum of all the elements of the given matrix.
      template <typename T>
      T matSum(const Basics::Matrix<T> *obj);

      /// Returns the sum of all the elements of the given matrix.
      template <>
      ComplexF matSum(const Basics::Matrix<ComplexF> *obj);

      /// Returns the sum of all the elements of the given matrix.
      template <typename T>
      T matSum(const Basics::SparseMatrix<T> *obj);

      /**
       * @brief Sum reduction over a given dimension number.
       *
       * @param obj - The source matrix object.
       * @param dim - The dimension (a value between 0 and @c obj->getNumDim()
       * @param dest - The destination matrix, if it is 0, a new matrix would be allocated.
       *
       * @result The given @c dest argument or a new allocated matrix with the sum values.
       */
      template <typename T>
      Basics::Matrix<T> *matSum(Basics::Matrix<T> *obj,
                                int dim,
                                Basics::Matrix<T> *dest=0);

      // TODO: Implement using a wrapper for GPU/CPU computation.
      /**
       * @brief Sum reduction over a given dimension number.
       *
       * @param obj - The source matrix object.
       * @param dim - The dimension (a value between 0 and @c obj->getNumDim()
       * @param dest - The destination matrix, if it is 0, a new matrix would be allocated.
       *
       * @result The given @c dest argument or a new allocated matrix with the sum values.
       */
      template <typename T>
      Basics::Matrix<T> *matSum(const Basics::SparseMatrix<T> *obj, int dim,
                                Basics::Matrix<T> *dest=0);

      /**** COMPONENT WISE OPERATIONS ****/

      /// Returns true if \f$ A = B \f$ using the given \f$ \epsilon \f$ as relative error threshold.
      template <typename T>
      bool matEquals(const Basics::Matrix<T> *a, const Basics::Matrix<T> *b,
                     float epsilon);

      /// Returns true if \f$ A = B \f$ using the given \f$ \epsilon \f$ as relative error threshold.
      template <typename T>
      bool matEquals(const Basics::SparseMatrix<T> *a,
                     const Basics::SparseMatrix<T> *b,
                     float epsilon);

      /**
       * @brief Computes component-wise multiplication of two matrices.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matCmul(Basics::Matrix<T> *obj,
                                 const Basics::Matrix<T> *other,
                                 Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ x + v \f$ for every matrix element.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matScalarAdd(Basics::Matrix<T> *obj, const T &v,
                                      Basics::Matrix<T> *dest=0);

      /**
       * @brief Adjust linearly to a given range the values of the matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matAdjustRange(Basics::Matrix<T> *obj,
                                        const T &rmin, const T &rmax,
                                        Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ \frac{v}{x} \f$ for every element x in the matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matDiv(Basics::Matrix<T> *obj, const T &value,
                                Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes a component-wise division of two matrices.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::Matrix<T> *matDiv(Basics::Matrix<T> *obj,
                                const Basics::Matrix<T> *other,
                                Basics::Matrix<T> *dest=0);

      /**
       * @brief Computes \f$ \frac{v}{x} \f$ for every element x in the matrix.
       *
       * @note This operation is done <b>in-place</b> when @c dest=0 , otherwise
       * the operation takes @c obj as input and stores the result in @c dest .
       */
      template <typename T>
      Basics::SparseMatrix<T> *matDiv(Basics::SparseMatrix<T> *obj, const T &value,
                                      Basics::SparseMatrix<T> *dest=0);

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

      /**
       * @brief Computes real FFT with Hamming window using a sliding window.
       *
       * This function uses the given @c wsize and @c wadvance parameters to
       * traverse the given matrix with a sliding window, and for every window
       * applies the Hamming filter and computes real FFT. The result is a
       * matrix with as many columns as FFT bins, and as many rows as windows.
       *
       * @param obj - the source matrix.
       * @param wsize - the source matrix.
       * @param wadvance - the source matrix.
       * @param obj - the source matrix.
       *
       * @result The given @c dest argument or a new allocated matrix if @c
       * dest=0
       *
       * @see AprilMath::RealFFTwithHamming class.
       */
      Basics::Matrix<float> *matRealFFTwithHamming(Basics::Matrix<float> *obj,
                                                   int wsize,
                                                   int wadvance,
                                                   Basics::Matrix<float> *dest=0);

    } // namespace Operations
  } // namespace MatrixExt
} // namespace AprilMath

#include "matrix-conv.impl.h"

#endif // MATRIX_OPERATIONS_H
