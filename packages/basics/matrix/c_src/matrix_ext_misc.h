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

      /////////////////////////////////////////////////////////////////////////

      class BroadcastHelper {
      
        static AprilUtils::UniquePtr<int []> resultShape(const int *a_dim, const int Na,
                                                         const int *b_dim, const int Nb) {
          const int minN = AprilUtils::min(Na, Nb);
          const int maxN = AprilUtils::max(Na, Nb);
          int *shape = new int[maxN];
          for (int i=0; i<minN; ++i) {
            int n = a_dim[i], m = b_dim[i];
            if (n != m && n != 1 && m != 1) {
              ERROR_EXIT(256, "Not aligned matrix shapes\n");
            }
            shape[i] = AprilUtils::max(n, m);
          }
          if (maxN == Na) {
            for (int i=minN; i<maxN; ++i) shape[i] = a_dim[i];
          }
          else {
            for (int i=minN; i<maxN; ++i) shape[i] = b_dim[i];
          }
          return shape;
        }

        template<typename T, typename OP>
        static void broadcast(const OP &func, Basics::Matrix<T> *dest,
                              const Basics::Matrix<T> *other) {
          AprilUtils::SharedPtr< Basics::Matrix<T> > other_squeezed;
          AprilUtils::SharedPtr< Basics::Matrix<T> > dest_slice;
          AprilUtils::SharedPtr< Basics::Matrix<T> > dest_slice_squeezed;
          other_squeezed = other->constSqueeze();
          typename Basics::Matrix<T>::sliding_window
            dest_sw(dest,
                    dest->getDimPtr(),  // sub_matrix_size
                    0,                  // offset
                    dest->getDimPtr()); // step
          while(!dest_sw.isEnd()) {
            dest_slice = dest_sw.getMatrix(dest_slice.get());
            if (dest_slice_squeezed.empty()) {
              dest_slice_squeezed = dest_slice->squeeze();
            }
            AprilUtils::SharedPtr< Basics::Matrix<T> > out;
            out = func(dest_slice_squeezed.get(),
                       static_cast<const Basics::Matrix<T>*>(other_squeezed.get()));
            if (out.get() != dest_slice_squeezed.get()) {
              BLAS::matCopy(dest_slice_squeezed.get(), out.get());
            }
            dest_sw.next();
          }
        }
      
        template<typename T>
        struct copyFunctor {
          Basics::Matrix<T> *operator()(Basics::Matrix<T> *dest,
                                        const Basics::Matrix<T> *source) const {
            return BLAS::matCopy(dest, source);
          }
        };

      public:
        
        /**
         * @see AprilMath::MatrixExt::Misc::matBroadcast
         */
        template<typename T, typename OP>
        static Basics::Matrix<T> *execute(const OP &func,
                                          const Basics::Matrix<T> *a,
                                          const Basics::Matrix<T> *b,
                                          Basics::Matrix<T> *result = 0) {
          const int *a_dim = a->getDimPtr(), *b_dim = b->getDimPtr();
          const int Na = a->getNumDim(), Nb = b->getNumDim();
          const int N = AprilUtils::max(Na, Nb);
          AprilUtils::UniquePtr<int []> shape = resultShape(a_dim, Na, b_dim, Nb);
          if (result == 0) {
            result = new Basics::Matrix<T>(N, shape.get());
          }
          else {
            if (!result->sameDim(shape.get(), N)) {
              ERROR_EXIT(128, "Incompatible shape in result matrix\n");
            }
          }
          if (result == b) {
            broadcast(func, result, a);
          }
          else if (result == a) {
            broadcast(func, result, b);
          }
          else {
            broadcast(copyFunctor<T>(), result, a);
            broadcast(func, result, b);
          }
          return result;
        }
      };
      
      /**
       * @brief Similar to broadcasting in SciPy:
       * http://wiki.scipy.org/EricsBroadcastingDoc
       *
       * The operator is called as: @c out=func(a,b) where 'a' can be input and
       * output at the same time, 'b' is always input data.
       *
       * @note Matrix<T> *func(Matrix<T> *a, const Matrix<T> *b);
       */
      template<typename T, typename OP>
      static Basics::Matrix<T> *matBroadcast(const OP &func,
                                             const Basics::Matrix<T> *a,
                                             const Basics::Matrix<T> *b,
                                             Basics::Matrix<T> *result = 0) {
        return BroadcastHelper::execute(func, a, b, result);
      }
      
      /**
       * @brief Changes the type of a matrix instance.
       */
      template <typename T, typename O>
      Basics::Matrix<O> *matConvertTo(const Basics::Matrix<T> *input,
                                      Basics::Matrix<O> *dest=0);
      
    } // namespace Misc
    
  } // namespace MatrixExt
} // namespace AprilMath

#include "matrix-conv.impl.h"

#endif // MATRIX_EXT_MISC_H
