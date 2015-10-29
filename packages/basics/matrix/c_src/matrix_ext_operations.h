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
     * This operations have been overloaded to be **in-place** or over a given
     * @c dest matrix. When given a destination matrix, it should be != 0. In
     * all cases, @c dest matrix is expected to be compatible with shape of @c
     * obj matrix.
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
       * @note This operation has been optimized considering properly the case
       * of \f$ p=0 \f$.
       *
       * @see Operations namespace.
       */
      template <typename T>
      Basics::Matrix<T> *matPlogp(const Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matPlogp(Basics::Matrix<T> *obj) {
        return matPlogp(obj, obj);
      }

      /**
       * @brief Computes \f$ \log x \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matLog(const Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matLog(Basics::Matrix<T> *obj) {
        return matLog(obj, obj);
      }

      /**
       * @brief Computes \f$ \log (1 + x) \f$ for every element of the given
       * matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matLog1p(const Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matLog1p(Basics::Matrix<T> *obj) {
        return matLog1p(obj, obj);
      }

      /**
       * @brief Computes \f$ \exp x \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matExp(const Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest);
      
      template <typename T>
      Basics::Matrix<T> *matExp(Basics::Matrix<T> *obj) {
        return matExp(obj, obj);
      }

      /**
       * @brief Computes \f$ \exp (x) - 1 \f$ for every element of the given
       * matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matExpm1(const Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matExpm1(Basics::Matrix<T> *obj) {
        return matExpm1(obj, obj);
      }

      /**
       * @brief Computes \f$ \sqrt x \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matSqrt(const Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matSqrt(Basics::Matrix<T> *obj) {
        return matSqrt(obj, obj);
      }

      /**
       * @brief Computes \f$ \sqrt x \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matSqrt(const Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matSqrt(Basics::SparseMatrix<T> *obj) {
        return matSqrt(obj, obj);
      }

      /**
       * @brief Computes \f$ x^v \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matPow(const Basics::Matrix<T> *obj, const T &value,
                                Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matPow(Basics::Matrix<T> *obj, const T &value) {
        return matPow(obj, value, obj);
      }

      /**
       * @brief Computes \f$ x^y \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matPow(const Basics::SparseMatrix<T> *obj,
                                      const T &value,
                                      Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matPow(Basics::SparseMatrix<T> *obj, const T &value) {
        return matPow(obj, value, obj);
      }

      /**
       * @brief Computes \f$ tan(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matTan(const Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matTan(Basics::Matrix<T> *obj) {
        return matTan(obj, obj);
      }

      /**
       * @brief Computes \f$ tan(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matTan(const Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matTan(Basics::SparseMatrix<T> *obj) {
        return matTan(obj, obj);
      }

      /**
       * @brief Computes \f$ tanh(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matTanh(const Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matTanh(Basics::Matrix<T> *obj) {
        return matTanh(obj, obj);
      }

      /**
       * @brief Computes \f$ tanh(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matTanh(const Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matTanh(Basics::SparseMatrix<T> *obj) {
        return matTanh(obj, obj);
      }

      /**
       * @brief Computes \f$ atan(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matAtan(const Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matAtan(Basics::Matrix<T> *obj) {
        return matAtan(obj, obj);
      }

      /**
       * @brief Computes \f$ atan(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matAtan(const Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matAtan(Basics::SparseMatrix<T> *obj) {
        return matAtan(obj, obj);
      }

      /**
       * @brief Computes \f$ atanh(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matAtanh(const Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matAtanh(Basics::Matrix<T> *obj) {
        return matAtanh(obj, obj);
      }

      /**
       * @brief Computes \f$ atanh(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matAtanh(const Basics::SparseMatrix<T> *obj,
                                        Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matAtanh(Basics::SparseMatrix<T> *obj) {
        return matAtanh(obj, obj);
      }

      /**
       * @brief Computes \f$ sin(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matSin(const Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matSin(Basics::Matrix<T> *obj) {
        return matSin(obj, obj);
      }

      /**
       * @brief Computes \f$ sin(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matSin(const Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matSin(Basics::SparseMatrix<T> *obj) {
        return matSin(obj, obj);
      }

      /**
       * @brief Computes \f$ sinh(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matSinh(const Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matSinh(Basics::Matrix<T> *obj) {
        return matSinh(obj, obj);
      }

      /**
       * @brief Computes \f$ sinh(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matSinh(const Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matSinh(Basics::SparseMatrix<T> *obj) {
        return matSinh(obj, obj);
      }

      /**
       * @brief Computes \f$ asin(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matAsin(const Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matAsin(Basics::Matrix<T> *obj) {
        return matAsin(obj, obj);
      }

      /**
       * @brief Computes \f$ asin(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matAsin(const Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matAsin(Basics::SparseMatrix<T> *obj) {
        return matAsin(obj, obj);
      }

      /**
       * @brief Computes \f$ asinh(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matAsinh(const Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest);
      template <typename T>
      Basics::Matrix<T> *matAsinh(Basics::Matrix<T> *obj) {
        return matAsinh(obj, obj);
      }

      /**
       * @brief Computes \f$ asinh(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matAsinh(const Basics::SparseMatrix<T> *obj,
                                        Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matAsinh(Basics::SparseMatrix<T> *obj) {
        return matAsinh(obj, obj);
      }

      /**
       * @brief Computes \f$ cos(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matCos(const Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matCos(Basics::Matrix<T> *obj) {
        return matCos(obj, obj);
      }

      /**
       * @brief Computes \f$ cosh(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matCosh(const Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matCosh(Basics::Matrix<T> *obj) {
        return matCosh(obj, obj);
      }

      /**
       * @brief Computes \f$ acos(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matAcos(const Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matAcos(Basics::Matrix<T> *obj) {
        return matAcos(obj, obj);
      }

      /**
       * @brief Computes \f$ acosh(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matAcosh(const Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matAcosh(Basics::Matrix<T> *obj) {
        return matAcosh(obj, obj);
      }

      /**
       * @brief Computes \f$ abs(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matAbs(const Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matAbs(Basics::Matrix<T> *obj) {
        return matAbs(obj, obj);
      }

      /**
       * @brief Computes \f$ abs(x) \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matAbs(const Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matAbs(Basics::SparseMatrix<T> *obj) {
        return matAbs(obj, obj);
      }

      /**
       * @brief Computes \f$ 1 - x \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matComplement(const Basics::Matrix<T> *obj,
                                       Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matComplement(Basics::Matrix<T> *obj) {
        return matComplement(obj, obj);
      }

      /**
       * @brief Computes \f$ \frac{x}{|x|} \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matSign(const Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matSign(Basics::Matrix<T> *obj) {
        return matSign(obj, obj);
      }

      /**
       * @brief Computes \f$ \frac{x}{|x|} \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matSign(const Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matSign(Basics::SparseMatrix<T> *obj) {
        return matSign(obj, obj);
      }

      ////////////////// OTHER MAP FUNCTIONS //////////////////

      /**
       * @brief Computes \f$ \max(l, \min(u, x)) \f$ for every element of the
       * given matrix.
       */
      template<typename T>
      Basics::Matrix<T> *matClamp(const Basics::Matrix<T> *obj,
                                  const T lower, const T upper,
                                  Basics::Matrix<T> *dest);
      
      template<typename T>
      Basics::Matrix<T> *matClamp(Basics::Matrix<T> *obj,
                                  const T lower, const T upper) {
        return matClamp(obj, lower, upper, obj);
      }

      /**
       * @brief Computes \f$ floor(x) \f$ for every element of the
       * given matrix.
       */
      template<typename T>
      Basics::Matrix<T> *matFloor(const Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest);

      template<typename T>
      Basics::Matrix<T> *matFloor(Basics::Matrix<T> *obj) {
        return matFloor(obj, obj);
      }

      /**
       * @brief Computes \f$ ceil(x) \f$ for every element of the
       * given matrix.
       */
      template<typename T>
      Basics::Matrix<T> *matCeil(const Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest);

      template<typename T>
      Basics::Matrix<T> *matCeil(Basics::Matrix<T> *obj) {
        return matCeil(obj, obj);
      }
      
      /**
       * @brief Computes \f$ round(x) \f$ for every element of the
       * given matrix.
       */
      template<typename T>
      Basics::Matrix<T> *matRound(const Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest);

      template<typename T>
      Basics::Matrix<T> *matRound(Basics::Matrix<T> *obj) {
        return matRound(obj, obj);
      }

      /**
       * @brief Computes \f$ v \cdot x \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matScal(const Basics::Matrix<T> *obj, const T value,
                                 Basics::Matrix<T> *dest);

      template <>
      Basics::Matrix<int32_t> *matScal(const Basics::Matrix<int32_t> *obj,
                                       const int32_t value,
                                       Basics::Matrix<int32_t> *dest);

      template <typename T>
      Basics::Matrix<T> *matScal(Basics::Matrix<T> *obj, const T value) {
        return matScal(obj, value, obj);
      }
      
      /**
       * @brief Computes \f$ v \cdot x \f$ for every element of the given matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matScal(const Basics::SparseMatrix<T> *obj,
                                       const T value,
                                       Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matScal(Basics::SparseMatrix<T> *obj,
                                       const T value) {
        return matScal(obj, value, obj);
      }

      /**
       * @brief Computes component-wise multiplication of two matrices.
       */
      template <typename T>
      Basics::Matrix<T> *matCmul(const Basics::Matrix<T> *obj,
                                 const Basics::Matrix<T> *other,
                                 Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matCmul(Basics::Matrix<T> *obj,
                                 const Basics::Matrix<T> *other) {
        return matCmul(obj, other, obj);
      }

      /**
       * @brief Computes \f$ x + v \f$ for every matrix element.
       */
      template <typename T>
      Basics::Matrix<T> *matScalarAdd(const Basics::Matrix<T> *obj, const T &v,
                                      Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matScalarAdd(Basics::Matrix<T> *obj, const T &v) {
        return matScalarAdd(obj, v, obj);
      }

      /**
       * @brief Adjust linearly to a given range the values of the matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matAdjustRange(const Basics::Matrix<T> *obj,
                                        const T &rmin, const T &rmax,
                                        Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matAdjustRange(Basics::Matrix<T> *obj,
                                        const T &rmin, const T &rmax) {
        return matAdjustRange(obj, rmin, rmax, obj);
      }

      /**
       * @brief Computes \f$ \frac{v}{x} \f$ for every element x in the matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matDiv(const Basics::Matrix<T> *obj, const T &value,
                                Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matDiv(Basics::Matrix<T> *obj, const T &value) {
        return matDiv(obj, value, obj);
      }

      /**
       * @brief Computes \f$ \frac{x}{v} \f$ for every element x in the matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matIDiv(const Basics::Matrix<T> *obj, const T &value,
                                 Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matIDiv(Basics::Matrix<T> *obj, const T &value) {
        return matIDiv(obj, value, obj);
      }

      /**
       * @brief Computes a component-wise division of two matrices.
       */
      template <typename T>
      Basics::Matrix<T> *matDiv(const Basics::Matrix<T> *obj,
                                const Basics::Matrix<T> *other,
                                Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matDiv(Basics::Matrix<T> *obj,
                                const Basics::Matrix<T> *other) {
        return matDiv(obj, other, obj);
      }

      /**
       * @brief Computes \f$ \frac{v}{x} \f$ for every element x in the matrix.
       */
      template <typename T>
      Basics::SparseMatrix<T> *matDiv(const Basics::SparseMatrix<T> *obj, const T &value,
                                      Basics::SparseMatrix<T> *dest);

      template <typename T>
      Basics::SparseMatrix<T> *matDiv(Basics::SparseMatrix<T> *obj, const T &value) {
        return matDiv(obj, value, obj);
      }
      

      /**
       * @brief Computes a component-wise mod operation of two matrices.
       */
      template <typename T>
      Basics::Matrix<T> *matMod(const Basics::Matrix<T> *obj,
                                const Basics::Matrix<T> *other,
                                Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matMod(Basics::Matrix<T> *obj,
                                const Basics::Matrix<T> *other) {
        return matMod(obj, other, obj);
      }
      

      /**
       * @brief Computes \f$ x % v \f$ for every element x in the matrix.
       */      
      template <typename T>
      Basics::Matrix<T> *matMod(const Basics::Matrix<T> *obj, const T &value,
                                Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matMod(Basics::Matrix<T> *obj, const T &value) {
        return matMod(obj, value, obj);
      }
      
      /**
       * @brief Computes matrix fill using a given mask matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matMaskedFill(const Basics::Matrix<T> *obj,
                                       const Basics::Matrix<bool> *mask,
                                       const T &value,
                                       Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matMaskedFill(Basics::Matrix<T> *obj,
                                       const Basics::Matrix<bool> *mask,
                                       const T &value) {
        return matMaskedFill(obj, mask, value, obj);
      }
      
      /**
       * @brief Computes matrix copy using a given mask matrix.
       */
      template <typename T>
      Basics::Matrix<T> *matMaskedCopy(const Basics::Matrix<T> *obj1,
                                       const Basics::Matrix<bool> *mask,
                                       const Basics::Matrix<T> *obj2,
                                       Basics::Matrix<T> *dest);

      template <typename T>
      Basics::Matrix<T> *matMaskedCopy(Basics::Matrix<T> *obj1,
                                       const Basics::Matrix<bool> *mask,
                                       const Basics::Matrix<T> *obj2) {
        return matMaskedCopy(obj1, mask, obj2, obj1);
      }
      
    } // namespace Operations
    
  } // namespace MatrixExt
} // namespace AprilMath

#endif // MATRIX_EXT_OPERATIONS_H
