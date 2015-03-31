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
#ifndef MATRIX_EXT_BOOLEAN_H
#define MATRIX_EXT_BOOLEAN_H

namespace AprilMath {
  
  namespace MatrixExt {

    /**
     * @brief Boolean operations over Matrix instances.
     *
     * This operations transform the given matrix in a Basics::MatrixBool
     * instance with True/False depending in the given condition. All the
     * functions receive as first argument the target Basics::Matrix and as last
     * argument an optional destination Basics::MatrixBool. In case of not given
     * last argument, a new Basics::MatrixBool will be allocated.
     *
     * @see AprilMath::MatrixExt
     */
    namespace Boolean {
      
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
      Basics::Matrix<bool> *matLT(const Basics::Matrix<T> *obj, const T &value,
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
      Basics::Matrix<bool> *matLT(const Basics::Matrix<T> *obj,
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
      Basics::Matrix<bool> *matGT(const Basics::Matrix<T> *obj, const T &value,
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
      Basics::Matrix<bool> *matGT(const Basics::Matrix<T> *obj,
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
      Basics::Matrix<bool> *matEQ(const Basics::Matrix<T> *obj, const T &value,
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
      Basics::Matrix<bool> *matEQ(const Basics::Matrix<T> *obj,
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
      Basics::Matrix<bool> *matNEQ(const Basics::Matrix<T> *obj, const T &value,
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
      Basics::Matrix<bool> *matNEQ(const Basics::Matrix<T> *obj,
                                   const Basics::Matrix<T> *other,
                                   Basics::Matrix<bool> *dest=0);
    } // namespace Boolean
    
  } // namespace MatrixExt
} // namespace AprilMath

#endif // MATRIX_EXT_BOOLEAN_H
