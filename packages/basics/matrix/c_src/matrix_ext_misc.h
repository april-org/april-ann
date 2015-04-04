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
#ifndef MATRIX_EXT_MISC_H
#define MATRIX_EXT_MISC_H

namespace AprilMath {

  namespace MatrixExt {

    /**
     * @brief Miscellaneous operations for Matrix instances.
     *
     * This operations have been implemented using other matrix operations,
     * they are wrappers to perform usual computations.
     *
     * @see AprilMath::MatrixExt
     */    
    namespace Misc {
      

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
      
    } // namespace Misc
    
  } // namespace MatrixExt
} // namespace AprilMath

#include "matrix-conv.impl.h"

#endif // MATRIX_EXT_MISC_H
