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
     * @see AprilMath::MatrixExt
     */
    namespace Operations {
      
      template <typename T>
      Basics::Matrix<T> *matPlogp(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matLog(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matLog1p(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0);
      
      template <typename T>
      Basics::Matrix<T> *matExp(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matSqrt(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::SparseMatrix<T> *matSqrt(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matPow(Basics::Matrix<T> *obj, const T &value,
                                Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::SparseMatrix<T> *matPow(Basics::SparseMatrix<T> *obj, const T &value,
                                      Basics::SparseMatrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matTan(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::SparseMatrix<T> *matTan(Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matTanh(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::SparseMatrix<T> *matTanh(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matAtan(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::SparseMatrix<T> *matAtan(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matAtanh(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::SparseMatrix<T> *matAtanh(Basics::SparseMatrix<T> *obj,
                                        Basics::SparseMatrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matSin(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::SparseMatrix<T> *matSin(Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matSinh(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::SparseMatrix<T> *matSinh(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matAsin(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::SparseMatrix<T> *matAsin(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matAsinh(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::SparseMatrix<T> *matAsinh(Basics::SparseMatrix<T> *obj,
                                        Basics::SparseMatrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matCos(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matCosh(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matAcos(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matAcosh(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matAbs(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::SparseMatrix<T> *matAbs(Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matComplement(Basics::Matrix<T> *obj,
                                       Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matSign(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::SparseMatrix<T> *matSign(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0);
    
      ////////////////// OTHER MAP FUNCTIONS //////////////////
    
      /// Performs a clamp operation, in-place operation if @c dest=0, otherwise,
      /// the result will be computed at the given @c dest Matrix.
      template<typename T>
      Basics::Matrix<T> *matClamp(Basics::Matrix<T> *obj,
                                  const T lower, const T upper,
                                  Basics::Matrix<T> *dest=0);
    
      template<typename T>
      Basics::Matrix<T> *matFill(Basics::Matrix<T> *obj, const T value);

      template<typename T>
      Basics::SparseMatrix<T> *matFill(Basics::SparseMatrix<T> *obj, const T value);

      template<typename T>
      Basics::Matrix<T> *matZeros(Basics::Matrix<T> *obj);

      template<typename T>
      Basics::SparseMatrix<T> *matZeros(Basics::SparseMatrix<T> *obj);
    
      template<typename T>
      Basics::Matrix<T> *matOnes(Basics::Matrix<T> *obj);

      template<typename T>
      Basics::SparseMatrix<T> *matOnes(Basics::SparseMatrix<T> *obj);
    
      template <typename T>
      Basics::Matrix<T> *matDiag(Basics::Matrix<T> *obj, const T value);
    
      //////////////////// CBLAS MATH OPERATIONS ////////////////////
    
      // SCOPY BLAS operation this = other
      template <typename T>
      Basics::Matrix<T> *matCopy(Basics::Matrix<T> *obj,
                                 const Basics::Matrix<T> *other);

      // SCOPY BLAS operation this = other
      template <typename T>
      Basics::SparseMatrix<T> *matCopy(Basics::SparseMatrix<T> *obj,
                                       const Basics::SparseMatrix<T> *other);
      
      // AXPY BLAS operation this = this + alpha * other
      template <typename T>
      Basics::Matrix<T> *matAxpy(Basics::Matrix<T> *obj, const T alpha,
                                 const Basics::Matrix<T> *other);

      template <typename T>
      Basics::Matrix<T> *matAxpy(Basics::Matrix<T> *obj, T alpha,
                                 const Basics::SparseMatrix<T> *other);
      
      // GEMM BLAS operation C = alpha * op(A)*op(B) + beta*C
      template <typename T>
      Basics::Matrix<T> *matGemm(Basics::Matrix<T> *C,
                                 CBLAS_TRANSPOSE trans_A,
                                 CBLAS_TRANSPOSE trans_B,
                                 const T alpha,
                                 const Basics::Matrix<T> *otherA,
                                 const Basics::Matrix<T> *otherB,
                                 T beta);
      
      // MM Sparse BLAS operation C = alpha * op(A)*op(B) + beta*op(C)
      template <typename T>
      Basics::Matrix<T> *matSparseMM(Basics::Matrix<T> *C,
                                     CBLAS_TRANSPOSE trans_A,
                                     CBLAS_TRANSPOSE trans_B,
                                     CBLAS_TRANSPOSE trans_C,
                                     const T alpha,
                                     const Basics::SparseMatrix<T> *otherA,
                                     const Basics::Matrix<T> *otherB,
                                     T beta);

      // GEMV BLAS operation Y = alpha * op(A)*X + beta*Y
      template <typename T>
      Basics::Matrix<T> *matGemv(Basics::Matrix<T> *Y,
                                 CBLAS_TRANSPOSE trans_A,
                                 const T alpha,
                                 const Basics::Matrix<T> *otherA,
                                 const Basics::Matrix<T> *otherX,
                                 const T beta);
      
      // GEMV Sparse BLAS operation this = alpha * op(A)*X + beta*this
      template <typename T>
      Basics::Matrix<T> *matGemv(Basics::Matrix<T> *Y, CBLAS_TRANSPOSE trans_A,
                                 const T alpha,
                                 const Basics::SparseMatrix<T> *otherA,
                                 const Basics::Matrix<T> *otherX,
                                 const T beta);
  
      // GER BLAS operation A = alpha * X*Y' + A
      template <typename T>
      Basics::Matrix<T> *matGer(Basics::Matrix<T> *A,
                                const T alpha,
                                const Basics::Matrix<T> *otherX,
                                const Basics::Matrix<T> *otherY);

      // DOT BLAS operation value = dot(X, Y)
      template <typename T>
      T matDot(const Basics::Matrix<T> *X, const Basics::Matrix<T> *Y);

      // DOT Sparse BLAS operation value = dot(this, other)
      template <typename T>
      T matDot(const Basics::Matrix<T> *X, const Basics::SparseMatrix<T> *Y);
    
      template <typename T>
      Basics::Matrix<T> *matScal(Basics::Matrix<T> *obj, const T value);

      template <typename T>
      Basics::SparseMatrix<T> *matScal(Basics::SparseMatrix<T> *obj,
                                       const T value);

      template <typename T>
      float matNorm2(Basics::Matrix<T> *obj);
      
      // FIXME: implement using a wrapper
      template <typename T>
      float matNorm2(Basics::SparseMatrix<T> *obj);
    
      /////////////////// MAX MIN REDUCTIONS ///////////////////

      // Min and max over given dimension, be careful, argmin and argmax matrices
      // contains the min/max index at the given dimension, but starting in 1 (not
      // in 0)

      template <typename T>
      Basics::Matrix<T> *matMin(Basics::Matrix<T> *obj,
                                int dim,
                                Basics::Matrix<T> *dest=0,
                                Basics::Matrix<int32_t> *argmin=0);

      // TODO: use a wrapper for GPU/CPU
      template <typename T>
      Basics::Matrix<T> *matMin(Basics::SparseMatrix<T> *obj, int dim,
                                Basics::Matrix<T> *dest=0,
                                Basics::Matrix<int32_t> *argmin=0);
    
      template <typename T>
      Basics::Matrix<T> *matMax(Basics::Matrix<T> *obj,
                                int dim,
                                Basics::Matrix<T> *dest=0,
                                Basics::Matrix<int32_t> *argmax=0);

      // TODO: use a wrapper for GPU/CPU
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
      template<typename T>
      T matMax(const Basics::Matrix<T> *obj, int &arg_max, int &arg_max_raw_pos);
    
      // FIXME: using WRAPPER
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
    
      template <typename T>
      Basics::Matrix<bool> *matLT(Basics::Matrix<T> *obj, const T &value,
                                  Basics::Matrix<bool> *dest=0);

      template <typename T>
      Basics::Matrix<bool> *matLT(Basics::Matrix<T> *obj,
                                  const Basics::Matrix<T> *other,
                                  Basics::Matrix<bool> *dest=0);

      template <typename T>
      Basics::Matrix<bool> *matGT(Basics::Matrix<T> *obj, const T &value,
                                  Basics::Matrix<bool> *dest=0);

      template <typename T>
      Basics::Matrix<bool> *matGT(Basics::Matrix<T> *obj,
                                  const Basics::Matrix<T> *other,
                                  Basics::Matrix<bool> *dest=0);

      template <typename T>
      Basics::Matrix<bool> *matEQ(Basics::Matrix<T> *obj, const T &value,
                                  Basics::Matrix<bool> *dest=0);
    
      template <typename T>
      Basics::Matrix<bool> *matEQ(Basics::Matrix<T> *obj,
                                  const Basics::Matrix<T> *other,
                                  Basics::Matrix<bool> *dest=0);
    
      template <typename T>
      Basics::Matrix<bool> *matNEQ(Basics::Matrix<T> *obj, const T &value,
                                   Basics::Matrix<bool> *dest=0);
    
      template <typename T>
      Basics::Matrix<bool> *matNEQ(Basics::Matrix<T> *obj,
                                   const Basics::Matrix<T> *other,
                                   Basics::Matrix<bool> *dest=0);
    
      //////////////////// OTHER MATH OPERATIONS ////////////////////
    
      template <typename T>
      Basics::Matrix<T> *matAddition(const Basics::Matrix<T> *a,
                                     const Basics::Matrix<T> *b,
                                     Basics::Matrix<T> *c = 0);

      template <typename T>
      Basics::Matrix<T> *matSubstraction(const Basics::Matrix<T> *a,
                                         const Basics::Matrix<T> *b,
                                         Basics::Matrix<T> *c = 0);
    
      template <typename T>
      Basics::Matrix<T> *matMultiply(const Basics::Matrix<T> *a,
                                     const Basics::Matrix<T> *b,
                                     Basics::Matrix<T> *c = 0);    
      template <typename T>
      T matSum(const Basics::Matrix<T> *obj);

      template <>
      ComplexF matSum(const Basics::Matrix<ComplexF> *obj);

      template <typename T>
      T matSum(const Basics::SparseMatrix<T> *obj);
    
      template <typename T>
      Basics::Matrix<T> *matSum(Basics::Matrix<T> *obj,
                                int dim,
                                Basics::Matrix<T> *dest=0);

      // TODO: Implement using a wrapper for GPU/CPU computation.
      template <typename T>
      Basics::Matrix<T> *matSum(const Basics::SparseMatrix<T> *obj, int dim,
                                Basics::Matrix<T> *dest=0);

      /**** COMPONENT WISE OPERATIONS ****/
    
      template <typename T>
      bool matEquals(const Basics::Matrix<T> *a, const Basics::Matrix<T> *b,
                     float epsilon);

      template <typename T>
      bool matEquals(const Basics::SparseMatrix<T> *a,
                     const Basics::SparseMatrix<T> *b,
                     float epsilon);

      template <typename T>
      Basics::Matrix<T> *matCmul(Basics::Matrix<T> *obj,
                                 const Basics::Matrix<T> *other,
                                 Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matScalarAdd(Basics::Matrix<T> *obj, const T &v,
                                      Basics::Matrix<T> *dest=0);
    
      template <typename T>
      Basics::Matrix<T> *matAdjustRange(Basics::Matrix<T> *obj,
                                        const T &rmin, const T &rmax,
                                        Basics::Matrix<T> *dest=0);        
      
      template <typename T>
      Basics::Matrix<T> *matDiv(Basics::Matrix<T> *obj, const T &value,
                                Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::Matrix<T> *matDiv(Basics::Matrix<T> *obj,
                                const Basics::Matrix<T> *other,
                                Basics::Matrix<T> *dest=0);

      template <typename T>
      Basics::SparseMatrix<T> *matDiv(Basics::SparseMatrix<T> *obj, const T &value,
                                      Basics::SparseMatrix<T> *dest=0);

      //////////////////// LAPACK MATH OPERATIONS ////////////////////
    
      // FIXME: using WRAPPER for generalized CULA, LAPACK, float and complex
      // numbers

      Basics::Matrix<float> *matInv(const Basics::Matrix<float> *obj);
    
      void matSVD(const Basics::Matrix<float> *obj,
                  Basics::Matrix<float> **U, Basics::SparseMatrix<float> **S,
                  Basics::Matrix<float> **VT);

      // FROM: http://www.r-bloggers.com/matrix-determinant-with-the-lapack-routine-dspsv/    
      AprilUtils::log_float matLogDeterminant(const Basics::Matrix<float> *obj,
                                              float &sign);

      // FROM: http://www.r-bloggers.com/matrix-determinant-with-the-lapack-routine-dspsv/    
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
      Basics::Matrix<float> *matCholesky(const Basics::Matrix<float> *obj,
                                         char uplo);

      Basics::Matrix<float> *matRealFFTwithHamming(Basics::Matrix<float> *obj,
						   int wsize,
						   int wadvance,
						   Basics::Matrix<float> *dest=0);
      
    } // namespace Operations
  } // namespace MatrixExt
} // namespace AprilMath

#include "matrix-conv.impl.h"

#endif // MATRIX_OPERATIONS_H
