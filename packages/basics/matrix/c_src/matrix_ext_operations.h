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
#ifndef MATRIX_EXT_OPERATIONS_H
#define MATRIX_EXT_OPERATIONS_H

namespace AprilMath {
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
      
      /**
       * @brief Computes matrix fill using a given mask matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matMaskedFill(Basics::Matrix<T> *obj,
                                       const Basics::Matrix<bool> *mask,
                                       const T &value,
                                       Basics::Matrix<T> *dest=0);
      
      /**
       * @brief Computes matrix copy using a given mask matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matMaskedCopy(Basics::Matrix<T> *obj1,
                                       const Basics::Matrix<bool> *mask,
                                       const Basics::Matrix<T> *obj2,
                                       Basics::Matrix<T> *dest=0);

    } // namespace Operations
    
  } // namespace MatrixExt
} // namespace AprilMath

#endif // MATRIX_EXT_OPERATIONS_H
