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
#ifndef MATRIX_EXT_REDUCTIONS_H
#define MATRIX_EXT_REDUCTIONS_H

namespace AprilMath {

  namespace MatrixExt {

    /**
     * @brief Reduction operations for Matrix instances.
     *
     * @see AprilMath::MatrixExt
     */    
    namespace Reductions {
      
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
      Basics::Matrix<T> *matMin(const Basics::Matrix<T> *obj,
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
      Basics::Matrix<T> *matMin(const Basics::SparseMatrix<T> *obj, int dim,
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
      Basics::Matrix<T> *matMax(const Basics::Matrix<T> *obj,
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
      Basics::Matrix<T> *matMax(const Basics::SparseMatrix<T> *obj,
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
       * @param accumulated - Indicates if the sum has to be accumulated to dest.
       *
       * @result The given @c dest argument or a new allocated matrix with the sum values.
       */
      template <typename T>
      Basics::Matrix<T> *matSum(Basics::Matrix<T> *obj,
                                int dim,
                                Basics::Matrix<T> *dest=0,
                                bool accumulated=false);

      // TODO: Implement using a wrapper for GPU/CPU computation.
      /**
       * @brief Sum reduction over a given dimension number.
       *
       * @param obj - The source matrix object.
       * @param dim - The dimension (a value between 0 and @c obj->getNumDim()
       * @param dest - The destination matrix, if it is 0, a new matrix would be allocated.
       * @param accumulated - Indicates if the sum has to be accumulated to dest.
       *
       * @result The given @c dest argument or a new allocated matrix with the sum values.
       */
      template <typename T>
      Basics::Matrix<T> *matSum(const Basics::SparseMatrix<T> *obj, int dim,
                                Basics::Matrix<T> *dest=0,
                                bool accumulated=false);

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
       * @brief Indicates if the matrix numbers are finite or note.
       */
      template <typename T>
      bool matIsFinite(const Basics::Matrix<T> *obj);      
      
    } // namespace Reductions
    
  } // namespace MatrixExt
} // namespace AprilMath

#endif // MATRIX_EXT_REDUCTIONS_H
