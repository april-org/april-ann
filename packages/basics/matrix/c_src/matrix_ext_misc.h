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
#include "swap.h"
#ifndef MATRIX_EXT_MISC_H
#define MATRIX_EXT_MISC_H

namespace AprilMath {

  namespace MatrixExt {

    /**
     * @brief Miscellaneous operations for Matrix instances.
     *
     * This operations have been implemented using other matrix operations or
     * ad-hoc implementations. They are wrappers to perform usual computations.
     *
     * @see AprilMath::MatrixExt
     */    
    namespace Misc {
      

      //////////////////// OTHER MATH OPERATIONS ////////////////////

      /// To be called from Lua. Indices start at 1 instead of 0.
      template <typename T>
      Basics::Matrix<int32_t> *matOrder(const Basics::Matrix<T> *m,
                                        Basics::Matrix<int32_t> *dest = 0);

      /// To be called from Lua. Indices start at 1 instead of 0.
      template <typename T>
      Basics::Matrix<int32_t> *matOrderRank(const Basics::Matrix<T> *m,
                                            Basics::Matrix<int32_t> *dest = 0);
      
      /// Changes the order of matrix data over a given dimension. Returns a new allocated matrix.
      template <typename T>
      Basics::Matrix<T> *matIndex(const Basics::Matrix<T> *m, int dim,
                                  const Basics::Matrix<int32_t> *idx);

      /// Changes the order of matrix data over a given dimension. Returns a new allocated matrix.
      template <typename T>
      Basics::Matrix<T> *matIndex(const Basics::Matrix<T> *m, int dim,
                                  const Basics::Matrix<bool> *mask);

      /// Fills the given indices at the given dimension. Returns m.
      template <typename T>
      Basics::Matrix<T> *matIndexedFill(Basics::Matrix<T> *m, int dim,
                                        const Basics::Matrix<int32_t> *idx,
                                        T val);

      /// Fills the given indices at the given dimension. Returns m.
      template <typename T>
      Basics::Matrix<T> *matIndexedFill(Basics::Matrix<T> *m, int dim,
                                        const Basics::Matrix<bool> *mask,
                                        T val);

      /// Copies the given indices at the given dimension. Returns m.
      template <typename T>
      Basics::Matrix<T> *matIndexedCopy(Basics::Matrix<T> *m, int dim,
                                        const Basics::Matrix<int32_t> *idx,
                                        const Basics::Matrix<T> *other);

      /// Copies the given indices at the given dimension. Returns m.
      template <typename T>
      Basics::Matrix<T> *matIndexedCopy(Basics::Matrix<T> *m, int dim,
                                        const Basics::Matrix<bool> *mask,
                                        const Basics::Matrix<T> *other);
      
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
       * This operation implements a generic multiplication which result
       * depends in the rank and shape of the given matrices. So, if
       * a vector and a column vector are given, the operation implements
       * an outer product. If two row vectors are given, the operation
       * implements inner product. If a vector and a matrix is given, it
       * implements matrix-vector product. If two matrices are given, the
       * operation implements matrix-matrix product.
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
                
        /// Computes the shape of the broadcast result matrix.
        static AprilUtils::UniquePtr<int []> resultShape(const int *a_dim, int Na,
                                                         const int *b_dim, int Nb) {
          if (Na > Nb) { // make sure Na < Nb
            AprilUtils::swap(a_dim,b_dim);
            AprilUtils::swap(Na,Nb);
          }
          const int d = Nb - Na; // Na < Nb always here
          AprilUtils::UniquePtr<int []> shape = new int[Nb];
          for (int i=0; i<Na; ++i) {
            int n = a_dim[i], m = b_dim[i+d];
            if (n != m && n != 1 && m != 1) {
              ERROR_EXIT(256, "Not aligned matrix shapes\n");
            }
            shape[i+d] = AprilUtils::max(n, m);
          }
          for (int i=0; i<d; ++i) shape[i] = b_dim[i];
          return shape.release();
        }
        
        /// Performs broadcasting of other matrix into dest matrix using func operation
        template<typename T, typename OP>
        static void oneWayBroadcast(const OP &func,
                                    AprilUtils::SharedPtr<Basics::Matrix<T> > dest,
                                    const Basics::Matrix<T> *other) {
          if (dest->sameDim(other)) { // particular case
            AprilUtils::SharedPtr< Basics::Matrix<T> > out =
              func(dest.get(), other);
            if (out != dest) {
              BLAS::matCopy(dest.get(), out.get());
            }
          }
          else { // general case
            AprilUtils::SharedPtr< Basics::Matrix<T> > other_inflated;
            AprilUtils::SharedPtr< Basics::Matrix<T> > dest_slice;
            if (dest->getNumDim() != other->getNumDim()) {
              // We need to leftInflate 'other' matrix and change other ptr
              other_inflated = other->leftInflate(dest->getNumDim() - other->getNumDim());
              other = other_inflated.get();
            }
            const int *other_dim = other->getDimPtr();
            typename Basics::Matrix<T>::sliding_window
              dest_sw(dest.get(),
                      other_dim,      // sub_matrix_size
                      0,              // offset
                      other_dim);     // step
            while(!dest_sw.isEnd()) {
              dest_slice = dest_sw.getMatrix(dest_slice.get());
              AprilUtils::SharedPtr< Basics::Matrix<T> > out =
                func(dest_slice.get(), other);
              if (out.get() != dest_slice.get()) {
                BLAS::matCopy(dest_slice.get(), out.get());
              }
              dest_sw.next();
            }
          } // general case
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
        static AprilUtils::SharedPtr< Basics::Matrix<T> >
        execute(const OP &func,
                const Basics::Matrix<T> *a,
                const Basics::Matrix<T> *b,
                AprilUtils::SharedPtr< Basics::Matrix<T> > result = 0) {
          const int *a_dim = a->getDimPtr(), *b_dim = b->getDimPtr();
          const int Na = a->getNumDim(), Nb = b->getNumDim();
          const int N = AprilUtils::max(Na, Nb);
          AprilUtils::UniquePtr<int []> shape = resultShape(a_dim, Na, b_dim, Nb);
          if (result.empty()) {
            result = new Basics::Matrix<T>(N, shape.get());
          }
          else {
            if (!result->sameDim(shape.get(), N)) {
              ERROR_EXIT(128, "Incompatible shape in result matrix\n");
            }
          }
          if (result == b) {
            oneWayBroadcast(func, result, a);
          }
          else if (result == a) {
            oneWayBroadcast(func, result, b);
          }
          else {
            oneWayBroadcast(copyFunctor<T>(), result, a);
            oneWayBroadcast(func, result, b);
          }
          return result;
        }
      };
      
      /**
       * @brief Similar to broadcasting in SciPy:
       * http://wiki.scipy.org/EricsBroadcastingDoc
       *
       * The operator is called as: @c out=func(a,b) where 'a' can be input and
       * output at the same time (no const), 'b' is always input data (const).
       *
       * @note Matrix<T> *func(Matrix<T> *a, const Matrix<T> *b);
       */
      template<typename T, typename OP>
      Basics::Matrix<T> *matBroadcast(const OP &func,
                                      const Basics::Matrix<T> *a,
                                      const Basics::Matrix<T> *b,
                                      Basics::Matrix<T> *result = 0) {
        AprilUtils::SharedPtr< Basics::Matrix<T> > result_ref(result);
        result_ref = BroadcastHelper::execute(func, a, b, result_ref);
        return result_ref.weakRelease();
      }
      
      /**
       * @brief Changes the type of a matrix instance.
       */
      template <typename T, typename O>
      Basics::Matrix<O> *matConvertTo(const Basics::Matrix<T> *input,
                                      Basics::Matrix<O> *dest=0);
      
      /**
       * @brief Converts into MatrixInt32 with all non-zero indices. Indices
       * start a 1 instead of 0.
       */
      template <typename T>
      Basics::Matrix<int32_t> *matNonZeroIndices(const Basics::Matrix<T> *input,
                                                 Basics::Matrix<int32_t> *dest=0);
      
    } // namespace Misc
    
  } // namespace MatrixExt
} // namespace AprilMath

#include "matrix-conv.impl.h"

#endif // MATRIX_EXT_MISC_H
