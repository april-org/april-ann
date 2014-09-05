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

#include <climits> // for INT_MAX
#include "cmath_overloads.h"
#include "mathcore.h"
#include "mathcore.h"
#include "maxmin.h"
#include "smart_ptr.h"

// Must to be defined here.
#include "map_matrix.h"
#include "map_sparse_matrix.h"

// Must to be defined here.
#include "reduce_matrix.h"
#include "reduce_sparse_matrix.h"

#define UNARY_SCALAR_CAST T(*)(const T&)
#define BINARY_SCALAR_CAST T(*)(const T&,const T&)
#define BINARY_MINMAX_SCALAR_CAST T(*)(const T&,const T&,unsigned int&)
#define UNARY_SPAN_CAST(T,O) void(*)(unsigned int,const GPUMirroredMemoryBlock< T > *, unsigned int, unsigned int, GPUMirroredMemoryBlock< O > *, unsigned int, unsigned int, bool)

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
    
    /// Useful functors for Matrix operations.
    namespace Functors {
      
      template <typename T>
      struct MatrixNorm2Reductor {
        float operator()(const T &a, const T &b) const {
          return static_cast<float>(m_sqrt(a*a + b*b));
        }
      };
      
      template<typename T>
      struct SpanSumReductor {
        T operator()(unsigned int N,
                     const GPUMirroredMemoryBlock<T> *input,
                     unsigned int input_stride,
                     unsigned int input_shift,
                     bool use_cuda,
                     const T &zero,
                     GPUMirroredMemoryBlock<T> *dest=0,
                     unsigned int dest_raw_pos=0) const {
          UNUSED_VARIABLE(zero);
          return sumReduceCall(N, input, input_stride, input_shift,
                               use_cuda, dest, dest_raw_pos);
        }
      };
      
    } // namespace Functors
    
    ///////// BASIC MAP FUNCTIONS /////////
    
    /**
     * @brief Valid operations over Matrix instances. They are wrappers over
     * generic functions defined at MatrixExt for map or reduce operations using
     * Matrix instances.
     *
     * @see AprilMath::MatrixExt
     */
    namespace Operations {
      
      template <typename T>
      Basics::Matrix<T> *matPlogp(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T> (obj, (UNARY_SCALAR_CAST)m_plogp<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matLog(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_log<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matLog1p(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_log1p<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matExp(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_exp<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matSqrt(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_sqrt<T>, dest);
      }

      template <typename T>
      Basics::SparseMatrix<T> *matSqrt(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_sqrt<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matPow(Basics::Matrix<T> *obj, const T &value,
                                Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_pow<T>(value), dest);
      }

      template <typename T>
      Basics::SparseMatrix<T> *matPow(Basics::SparseMatrix<T> *obj, const T &value,
                                      Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, m_curried_pow<T>(value), dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matTan(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_tan<T>, dest);
      }

      template <typename T>
      Basics::SparseMatrix<T> *matTan(Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_tan<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matTanh(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_tanh<T>, dest);
      }

      template <typename T>
      Basics::SparseMatrix<T> *matTanh(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_tanh<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matAtan(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_atan<T>, dest);
      }

      template <typename T>
      Basics::SparseMatrix<T> *matAtan(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_atan<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matAtanh(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_atanh<T>, dest);
      }

      template <typename T>
      Basics::SparseMatrix<T> *matAtanh(Basics::SparseMatrix<T> *obj,
                                        Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_atanh<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matSin(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_sin<T>, dest);
      }

      template <typename T>
      Basics::SparseMatrix<T> *matSin(Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_sin<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matSinh(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_sinh<T>, dest);
      }

      template <typename T>
      Basics::SparseMatrix<T> *matSinh(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_sinh<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matAsin(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_asin<T>, dest);
      }
    
      template <typename T>
      Basics::SparseMatrix<T> *matAsin(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_asin<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matAsinh(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_asinh<T>, dest);
      }

      template <typename T>
      Basics::SparseMatrix<T> *matAsinh(Basics::SparseMatrix<T> *obj,
                                        Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_asinh<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matCos(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_cos<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matCosh(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_cosh<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matAcos(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_acos<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matAcosh(Basics::Matrix<T> *obj,
                                  Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_acosh<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matAbs(Basics::Matrix<T> *obj,
                                Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_abs<T>, dest);
      }

      template <typename T>
      Basics::SparseMatrix<T> *matAbs(Basics::SparseMatrix<T> *obj,
                                      Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_abs<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matComplement(Basics::Matrix<T> *obj,
                                       Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_complement<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matSign(Basics::Matrix<T> *obj,
                                 Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_sign<T>, dest);
      }
    
      template <typename T>
      Basics::SparseMatrix<T> *matSign(Basics::SparseMatrix<T> *obj,
                                       Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, (UNARY_SCALAR_CAST)m_sign<T>, dest);
      }
    
      ////////////////// OTHER MAP FUNCTIONS //////////////////
    
      /// Performs a clamp operation, in-place operation if @c dest=0, otherwise,
      /// the result will be computed at the given @c dest Matrix.
      template<typename T>
      Basics::Matrix<T> *matClamp(Basics::Matrix<T> *obj,
                                  const T lower, const T upper,
                                  Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_clamp<T>(lower,upper), dest);
      }
    
      template<typename T>
      Basics::Matrix<T> *matFill(Basics::Matrix<T> *obj, const T value) {
        return MatrixScalarMap1<T,T>(obj, m_curried_fill<T>(value), obj);
      }

      template<typename T>
      Basics::SparseMatrix<T> *matFill(Basics::SparseMatrix<T> *obj, const T value) {
        return SparseMatrixScalarMap1<T,T>(obj,  m_curried_fill<T>(value), obj);
      }

      template<typename T>
      Basics::Matrix<T> *matZeros(Basics::Matrix<T> *obj) {
        return matFill(obj, T());
      }

      template<typename T>
      Basics::SparseMatrix<T> *matZeros(Basics::SparseMatrix<T> *obj) {
        return matFill(obj, T());
      }
    
      template<typename T>
      Basics::Matrix<T> *matOnes(Basics::Matrix<T> *obj) {
        return matFill(obj, T(1.0f));
      }

      template<typename T>
      Basics::SparseMatrix<T> *matOnes(Basics::SparseMatrix<T> *obj) {
        return matFill(obj, T(1.0f));
      }
    
      template <typename T>
      Basics::Matrix<T> *matDiag(Basics::Matrix<T> *obj, const T value) {
        if (obj->getCudaFlag()) {
          ERROR_PRINT("WARNING! DIAG OPERATION NOT IMPLENTED FOR CUDA\n");
        }
        for (int i=1; i<obj->getNumDim(); ++i) {
          if (obj->getDimSize(i) != obj->getDimSize(i-1)) {
            ERROR_EXIT(128, "Only allowed for squared matrices\n");
          }
        }
        typename Basics::Matrix<T>::random_access_iterator it(obj);
        AprilUtils::UniquePtr<int []> aux_coords(new int[obj->getNumDim()]);
        for (int i=0; i<obj->getDimSize(0); ++i) {
          for (int j=0; j<obj->getNumDim(); ++j) aux_coords[j] = i;
          it(aux_coords.get(), obj->getNumDim()) = value;
        }
        return obj;
      }
    
      //////////////////// CBLAS MATH OPERATIONS ////////////////////
    
      // SCOPY BLAS operation this = other
      template <typename T>
      Basics::Matrix<T> *matCopy(Basics::Matrix<T> *obj,
                                 const Basics::Matrix<T> *other) {
        return MatrixSpanMap1<T,T>(other,
                                   (UNARY_SPAN_CAST(T,T))doCopy<T>, obj);
      }

      // SCOPY BLAS operation this = other
      template <typename T>
      Basics::SparseMatrix<T> *matCopy(Basics::SparseMatrix<T> *obj,
                                       const Basics::SparseMatrix<T> *other) {
        return SparseMatrixScalarMap1<T,T>(other,
                                           (UNARY_SCALAR_CAST)m_identity<T>,
                                           obj);
      }

      // Specialization for char
      template <>
      Basics::Matrix<char> *matCopy(Basics::Matrix<char> *obj,
                                    const Basics::Matrix<char> *other);

      // Specialization for int32_t
      template <>
      Basics::Matrix<int32_t> *matCopy(Basics::Matrix<int32_t> *obj,
                                       const Basics::Matrix<int32_t> *other);
  
      // AXPY BLAS operation this = this + alpha * other
      template <typename T>
      Basics::Matrix<T> *matAxpy(Basics::Matrix<T> *obj, const T alpha,
                                 const Basics::Matrix<T> *other) {
        if (obj->size() != other->size()) {
          ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d\n",
                      obj->size(), other->size());
        }
#ifdef USE_MKL
        // INT_MAX avoids OMP parallel for
        return MatrixSpanMap1<T,T>(other, CurriedAxpy<T>(alpha), obj, INT_MAX);
#else
        return MatrixSpanMap1<T,T>(other, CurriedAxpy<T>(alpha), obj);
#endif
      }

      template <typename T>
      Basics::Matrix<T> *matAxpy(Basics::Matrix<T> *obj, T alpha,
                                 const Basics::SparseMatrix<T> *other) {
        if (obj->size() != other->size()) {
          ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d\n",
                      obj->size(), other->size());
        }
        if (!obj->isVector()) {
          ERROR_EXIT(128, "sparse AXPY only works with vectors\n");
        }
        if ( (other->getSparseFormat() == CSR_FORMAT &&
              other->getDimSize(0) != 1) ||
             (other->getSparseFormat() == CSC_FORMAT &&
              other->getDimSize(1) != 1) ) {
          ERROR_EXIT(128, "sparse AXPY needs a CSR row-vector or a CSC col-vector\n");
        }
        doSparseAxpy(other->nonZeroSize(), alpha,
                     other->getRawValuesAccess(),
                     other->getRawIndicesAccess(),
                     obj->getRawDataAccess(),
                     static_cast<unsigned int>(obj->getOffset()),
                     static_cast<unsigned int>(obj->getVectorStride()),
                     obj->getCudaFlag());
        return obj;
      }

    
      // GEMM BLAS operation C = alpha * op(A)*op(B) + beta*C
      template <typename T>
      Basics::Matrix<T> *matGemm(Basics::Matrix<T> *C,
                                 CBLAS_TRANSPOSE trans_A,
                                 CBLAS_TRANSPOSE trans_B,
                                 const T alpha,
                                 const Basics::Matrix<T> *otherA,
                                 const Basics::Matrix<T> *otherB,
                                 T beta) {
        if (C->getTransposedFlag()) {
          ERROR_EXIT(128, "GEMM method don't work with transposed C matrix\n");
        }
        if (C == otherA || C == otherB) {
          ERROR_EXIT(128, "GEMM method couldn't receive as A or B argument "
                     "the C argument\n");
        }
        if (C->getNumDim() != 2 ||
            otherA->getNumDim() != 2 ||
            otherB->getNumDim() != 2) {
          ERROR_EXIT(128,"Incorrect number of dimensions, only allowed for numDim=2\n");
        }
        int row_idx_A = 0, col_idx_A = 1, row_idx_B = 0, col_idx_B = 1;
        if (trans_A == CblasTrans) AprilUtils::swap(row_idx_A, col_idx_A);
        if (trans_B == CblasTrans) AprilUtils::swap(row_idx_B, col_idx_B);
        if (C->getDimSize(0) != otherA->getDimSize(row_idx_A) ||
            C->getDimSize(1) != otherB->getDimSize(col_idx_B) ||
            otherA->getDimSize(col_idx_A) != otherB->getDimSize(row_idx_B)) {
          ERROR_EXIT6(128, "Incorrect matrixes dimensions: %dx%d + %dx%d * %dx%d\n",
                      C->getDimSize(0), C->getDimSize(1),
                      otherA->getDimSize(row_idx_A), otherA->getDimSize(col_idx_A),
                      otherB->getDimSize(row_idx_B), otherB->getDimSize(col_idx_B));
        }
        if (C->getMajorOrder() != otherA->getMajorOrder() ||
            otherA->getMajorOrder() != otherB->getMajorOrder()) {
          ERROR_EXIT(128, "Matrices with different major orders\n");
        }
  
        int M=C->getDimSize(0), N=C->getDimSize(1), K=otherA->getDimSize(col_idx_A);
        int lda, ldb, ldc;
        if (C->getMajorOrder() == CblasRowMajor) {
          lda = (!otherA->getTransposedFlag())?(otherA->getStrideSize(0)):(otherA->getStrideSize(1));
          ldb = (!otherB->getTransposedFlag())?(otherB->getStrideSize(0)):(otherB->getStrideSize(1));
          ldc = (!C->getTransposedFlag())?(C->getStrideSize(0)):(C->getStrideSize(1));
        }
        else {
          lda = (!otherA->getTransposedFlag())?(otherA->getStrideSize(1)):(otherA->getStrideSize(0));
          ldb = (!otherB->getTransposedFlag())?(otherB->getStrideSize(1)):(otherB->getStrideSize(0));
          ldc = (!C->getTransposedFlag())?(C->getStrideSize(1)  ):(C->getStrideSize(0));
        }
        if (otherA->getStrideSize(0) + otherA->getStrideSize(1) != lda+1 ||
            otherB->getStrideSize(0) + otherB->getStrideSize(1) != ldb+1 ||
            C->getStrideSize(0)      + C->getStrideSize(1)      != ldc+1) {
          ERROR_EXIT(128, "Contiguous matrices are needed\n");
        }
        if (otherA->getTransposedFlag()) trans_A=NEGATE_CBLAS_TRANSPOSE(trans_A);
        if (otherB->getTransposedFlag()) trans_B=NEGATE_CBLAS_TRANSPOSE(trans_B);
        doGemm(C->getMajorOrder(), trans_A, trans_B,
               M, N, K,
               alpha, otherA->getRawDataAccess(), lda,
               otherB->getRawDataAccess(), ldb,
               beta, C->getRawDataAccess(), ldc,
               otherA->getOffset(), otherB->getOffset(), C->getOffset(),
               C->getCudaFlag());
        return C;
      }

      // MM Sparse BLAS operation C = alpha * op(A)*op(B) + beta*op(C)
      template <typename T>
      Basics::Matrix<T> *matSparseMM(Basics::Matrix<T> *C,
                                     CBLAS_TRANSPOSE trans_A,
                                     CBLAS_TRANSPOSE trans_B,
                                     CBLAS_TRANSPOSE trans_C,
                                     const T alpha,
                                     const Basics::SparseMatrix<T> *otherA,
                                     const Basics::Matrix<T> *otherB,
                                     T beta) {
        if (C == otherB) {
          ERROR_EXIT(128, "Sparse GEMM method couldn't receive as A or B argument "
                     "the C argument\n");
        }
        if (C->getNumDim() != 2 ||
            otherA->getNumDim() != 2 ||
            otherB->getNumDim() != 2) {
          ERROR_EXIT(128,"Incorrect number of dimensions, only allowed for numDim=2\n");
        }
        int row_idx_A = 0, col_idx_A = 1, row_idx_B = 0, col_idx_B = 1;
        int row_idx_C = 0, col_idx_C = 1;
        if (trans_A == CblasTrans) AprilUtils::swap(row_idx_A, col_idx_A);
        if (trans_B == CblasTrans) AprilUtils::swap(row_idx_B, col_idx_B);
        if (trans_C == CblasTrans) AprilUtils::swap(row_idx_C, col_idx_C);
        if (C->getDimSize(row_idx_C) != otherA->getDimSize(row_idx_A) ||
            C->getDimSize(col_idx_C) != otherB->getDimSize(col_idx_B) ||
            otherA->getDimSize(col_idx_A) != otherB->getDimSize(row_idx_B)) {
          ERROR_EXIT6(128, "Incorrect matrixes dimensions: %dx%d + %dx%d * %dx%d\n",
                      C->getDimSize(row_idx_C), C->getDimSize(col_idx_C),
                      otherA->getDimSize(row_idx_A), otherA->getDimSize(col_idx_A),
                      otherB->getDimSize(row_idx_B), otherB->getDimSize(col_idx_B));
        }
        if (C->getMajorOrder() != otherB->getMajorOrder()) {
          ERROR_EXIT(128, "Matrices with different major orders\n"); 
        }
        int M=C->getDimSize(row_idx_C), N=C->getDimSize(col_idx_C), K=otherB->getDimSize(row_idx_B);
        int ldb, ldc;
        if (C->getMajorOrder() == CblasRowMajor) {
          ldb = (!otherB->getTransposedFlag())?(otherB->getStrideSize(0)):(otherB->getStrideSize(1));
          ldc = (!C->getTransposedFlag())?(C->getStrideSize(0)):(C->getStrideSize(1));
        }
        else {
          ldb = (!otherB->getTransposedFlag())?(otherB->getStrideSize(1)):(otherB->getStrideSize(0));
          ldc = (!C->getTransposedFlag())?(C->getStrideSize(1)):(C->getStrideSize(0));
        }
        if (otherB->getStrideSize(0)+ otherB->getStrideSize(1) != ldb+1 ||
            C->getStrideSize(0)     + C->getStrideSize(1)      != ldc+1) {
          ERROR_EXIT(128, "Contiguous matrices are needed\n");
        }
        if (otherB->getTransposedFlag()) trans_B=NEGATE_CBLAS_TRANSPOSE(trans_B);
        if (C->getTransposedFlag())      trans_C=NEGATE_CBLAS_TRANSPOSE(trans_C);
        doSparseMM<T>(C->getMajorOrder(),
                      otherA->getSparseFormat(),
                      trans_A,
                      trans_B,
                      trans_C,
                      M, N, K,
                      alpha,
                      otherA->getRawValuesAccess(),
                      otherA->getRawIndicesAccess(),
                      otherA->getRawFirstIndexAccess(),
                      otherB->getRawDataAccess(), ldb,
                      beta, C->getRawDataAccess(), ldc,
                      otherB->getOffset(), C->getOffset(),
                      C->getCudaFlag());
        return C;
      }

      // GEMV BLAS operation Y = alpha * op(A)*X + beta*Y
      template <typename T>
      Basics::Matrix<T> *matGemv(Basics::Matrix<T> *Y,
                                 CBLAS_TRANSPOSE trans_A,
                                 const T alpha,
                                 const Basics::Matrix<T> *otherA,
                                 const Basics::Matrix<T> *otherX,
                                 const T beta) {
        if (!Y->isVector() || !otherX->isVector() || otherA->getNumDim() != 2) {
          ERROR_EXIT(128,"Incorrect number of dimensions\n");
        }
        int M,N;
        if (otherA->getTransposedFlag()) {
          trans_A=NEGATE_CBLAS_TRANSPOSE(trans_A);
          M=otherA->getDimSize(1);
          N=otherA->getDimSize(0);
        }else {
          M=otherA->getDimSize(0);
          N=otherA->getDimSize(1);
        }
        // SANITY CHECK
        if (trans_A == CblasNoTrans) {
          if (M != Y->size() || N != otherX->size()) {
            ERROR_EXIT4(128, "Incorrect matrixes dimensions: %dx1 + %dx%d * %dx1\n",
                        Y->size(), M, N, otherX->size());
          }
        }
        else {
          if (N != Y->size() || M != otherX->size())
            ERROR_EXIT4(128, "Incorrect matrixes dimensions: %dx1 + %dx%d * %dx1\n",
                        Y->size(), N, M, otherX->size());
        }
        if (Y->getMajorOrder() != otherA->getMajorOrder() ||
            otherA->getMajorOrder() != otherX->getMajorOrder()) {
          ERROR_EXIT(128, "Matrices with different major orders\n");
        }
        //
        int lda=(otherA->getIsDataRowOrdered())?otherA->getStrideSize(0):otherA->getStrideSize(1);
        int ldx=otherX->getVectorStride();
        int ldy=Y->getVectorStride();
        if (otherA->getStrideSize(0) + otherA->getStrideSize(1) != lda+1) {
          ERROR_EXIT(128, "Only allowed with contiguous matrices\n");
        }
        doGemv(Y->getMajorOrder(), trans_A,
               M, N,
               alpha, otherA->getRawDataAccess(), lda,
               otherX->getRawDataAccess(), ldx,
               beta, Y->getRawDataAccess(), ldy,
               otherA->getOffset(), otherX->getOffset(),
               Y->getOffset(),
               Y->getCudaFlag());
        return Y;
      }
    
      // GEMV Sparse BLAS operation this = alpha * op(A)*X + beta*this
      template <typename T>
      Basics::Matrix<T> *matGemv(Basics::Matrix<T> *Y, CBLAS_TRANSPOSE trans_A,
                                 const T alpha,
                                 const Basics::SparseMatrix<T> *otherA,
                                 const Basics::Matrix<T> *otherX,
                                 const T beta) {
        if (!Y->isVector() || !otherX->isVector()) {
          ERROR_EXIT(128,"Incorrect number of dimensions\n");
        }
        int M,N;
        M=otherA->getDimSize(0);
        N=otherA->getDimSize(1);
        // SANITY CHECK
        if (trans_A == CblasNoTrans) {
          if (M != Y->size() || N != otherX->size()) {
            ERROR_EXIT4(128, "Incorrect matrixes dimensions: %dx1 + %dx%d * %dx1\n",
                        Y->size(), M, N, otherX->size());
          }
        }
        else {
          if (N != Y->size() || M != otherX->size()) {
            ERROR_EXIT4(128, "Incorrect matrixes dimensions: %dx1 + %dx%d * %dx1\n",
                        Y->size(), N, M, otherX->size());
          }
        }
        //
        int ldx=otherX->getVectorStride();
        int ldy=Y->getVectorStride();
        doSparseGemv(otherA->getSparseFormat(),
                     trans_A,
                     M, N,
                     alpha,
                     otherA->getRawValuesAccess(),
                     otherA->getRawIndicesAccess(),
                     otherA->getRawFirstIndexAccess(),
                     otherX->getRawDataAccess(), ldx,
                     beta, Y->getRawDataAccess(), ldy,
                     otherX->getOffset(), Y->getOffset(),
                     Y->getCudaFlag());
        return Y;
      }
  
      // GER BLAS operation A = alpha * X*Y' + A
      template <typename T>
      Basics::Matrix<T> *matGer(Basics::Matrix<T> *A,
                                const T alpha,
                                const Basics::Matrix<T> *otherX,
                                const Basics::Matrix<T> *otherY) {
        if (A->getTransposedFlag()) {
          ERROR_EXIT(128, "GER method don't work with transposed A matrix\n");
        }
        if (!otherX->isVector() || !otherY->isVector() || A->getNumDim()!=2) {
          ERROR_EXIT(128,"Incorrect number of dimensions\n");
        }
        int M=otherX->size(), N=otherY->size();
        if (A->getDimSize(0) != M || A->getDimSize(1) != N) {
          ERROR_EXIT4(128, "Incorrect matrixes dimensions: %dx%d + %dx1 * 1x%d\n",
                      A->getDimSize(0), A->getDimSize(1), M, N);
        }
        if (A->getMajorOrder() != otherX->getMajorOrder() ||
            otherX->getMajorOrder() != otherY->getMajorOrder()) {
          ERROR_EXIT(128, "Matrices with different major orders\n");
        }
        int lda=(A->getIsDataRowOrdered())?A->getStrideSize(0):A->getStrideSize(1);
        int ldx=otherX->getVectorStride();
        int ldy=otherY->getVectorStride();
        doGer(A->getMajorOrder(),
              M, N,
              alpha, otherX->getRawDataAccess(), otherX->getOffset(), ldx,
              otherY->getRawDataAccess(), otherY->getOffset(), ldy,
              A->getRawDataAccess(), A->getOffset(), lda,
              A->getCudaFlag());
        return A;
      }

      // DOT BLAS operation value = dot(X, Y)
      template <typename T>
      T matDot(const Basics::Matrix<T> *X, const Basics::Matrix<T> *Y) {
        if (X->size() != Y->size()) {
          ERROR_EXIT2(128, "Incorrect dimensions: %d dot %d\n",
                      X->size(), Y->size());
        }
        if (X->getMajorOrder() != Y->getMajorOrder()) {
          ERROR_EXIT(128, "Matrices with different major orders\n");
        }
        return MatrixSpanReduce2<T>(X, Y,
                                    (SPAN_REDUCE2_CAST(T,T))doDot<T>,
                                    (BINARY_SCALAR_CAST)r_add<T>, T(0.0f));
      }

      // DOT Sparse BLAS operation value = dot(this, other)
      template <typename T>
      T matDot(const Basics::Matrix<T> *X, const Basics::SparseMatrix<T> *Y) {
        if (!X->isVector() || !Y->isVector()) {
          ERROR_EXIT(128,"Incorrect number of dimensions\n");
        }
        if (X->size() != Y->size()) {
          ERROR_EXIT2(128, "Incorrect dimensions: %d dot %d\n",
                      X->size(), Y->size());
        }
        if (Y->getDenseCoordinateSize() != 1) {
          ERROR_EXIT(128, "DOT operation only allowed with sparse matrices with "
                     "dense coordinate size of 1, please, change the sparse "
                     "format\n");
        }
        T ret = doSparseDot(Y->nonZeroSize(),
                            Y->getRawValuesAccess(),
                            Y->getRawIndicesAccess(),
                            X->getRawDataAccess(), X->getOffset(),
                            X->getVectorStride(),
                            X->getCudaFlag());
        return ret;
      }
    
      template <typename T>
      Basics::Matrix<T> *matScal(Basics::Matrix<T> *obj, const T value) {
#ifdef USE_MKL
        // INT_MAX avoids OMP parallel for
        return MatrixSpanMap1<T,T>(obj, CurriedScal<T>(value), obj, INT_MAX);
#else
        return MatrixSpanMap1<T,T>(obj, CurriedScal<T>(value), obj);
#endif
      }

      template <typename T>
      Basics::SparseMatrix<T> *matScal(Basics::SparseMatrix<T> *obj,
                                       const T value) {
        return SparseMatrixScalarMap1<T,T>(obj, m_curried_mul<T>(value), obj);
      }

      template <typename T>
      float matNorm2(Basics::Matrix<T> *obj) {
        return MatrixSpanReduce1(obj, (SPAN_REDUCE_CAST(T,float))doNrm2<T>,
                                 Functors::MatrixNorm2Reductor<T>(),
                                 float(0.0f));
      }
      
      // FIXME: implement using a wrapper
      template <typename T>
      float matNorm2(Basics::SparseMatrix<T> *obj) {
        float result = 0.0f;
        for (typename Basics::SparseMatrix<T>::const_iterator it(obj->begin());
             it != obj->end(); ++it) {
          result += (*it) * (*it);
        }
        return m_sqrt(result);
      }
    
      /////////////////// MAX MIN REDUCTIONS ///////////////////

      // Min and max over given dimension, be careful, argmin and argmax matrices
      // contains the min/max index at the given dimension, but starting in 1 (not
      // in 0)

      template <typename T>
      Basics::Matrix<T> *matMin(Basics::Matrix<T> *obj,
                                int dim,
                                Basics::Matrix<T> *dest=0,
                                Basics::Matrix<int32_t> *argmin=0) {
        if (argmin == 0) {
          return MatrixScalarReduceOverDimension(obj, dim,
                                                 (BINARY_SCALAR_CAST)r_min<T>,
                                                 Limits<T>::max(), dest);
        }
        else {
          return MatrixScalarReduceMinMaxOverDimension(obj, dim,
                                                       (BINARY_MINMAX_SCALAR_CAST)r_min2<T>,
                                                       Limits<T>::max(), argmin, dest);
        }
      }

      // TODO: use a wrapper for GPU/CPU
      template <typename T>
      Basics::Matrix<T> *matMin(Basics::SparseMatrix<T> *obj, int dim,
                                Basics::Matrix<T> *dest=0,
                                Basics::Matrix<int32_t> *argmin=0) {
        if (dim != 0 && dim != 1) {
          ERROR_EXIT1(128, "Incorrect given dimension %d\n", dim);
        }
        int ndim = (dim==0)?(1):(0);
        if (dest) {
          if (dest->getDimSize(dim) != 1 ||
              dest->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
        }
        else {
          int result_dims[2] = { obj->getDimSize(0), obj->getDimSize(1) };
          result_dims[dim] = 1;
          dest = new Basics::Matrix<T>(1, result_dims);
        }
        if (argmin) {
          if (argmin->getDimSize(dim) != 1 ||
              argmin->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
          matZeros(argmin);
        }
        matZeros(dest);
        typename Basics::Matrix<T>::random_access_iterator dest_it(dest);
        int aux_dims[2] = { 0, 0 };
        if (argmin == 0) {
          for (typename Basics::SparseMatrix<T>::const_iterator it(obj->begin());
               it!=obj->end(); ++it) {
            int coords[2];
            it.getCoords(coords[0],coords[1]);
            aux_dims[ndim] = coords[ndim];
            dest_it(aux_dims[0],aux_dims[1]) =
              AprilUtils::min(dest_it(aux_dims[0],aux_dims[1]),(*it));
          }
        }
        else {
          typename Basics::Matrix<int32_t>::random_access_iterator argmin_it(argmin);
          for (typename Basics::SparseMatrix<T>::const_iterator it(obj->begin());
               it!=obj->end(); ++it) {
            int coords[2];
            it.getCoords(coords[0],coords[1]);
            aux_dims[ndim] = coords[ndim];
            if (*it < dest_it(aux_dims[0],aux_dims[1])) {
              dest_it(aux_dims[0],aux_dims[1]) = *it;
              argmin_it(aux_dims[0],aux_dims[1]) = aux_dims[ndim];
            }
          }
        }
        return dest;
      }
    
      template <typename T>
      Basics::Matrix<T> *matMax(Basics::Matrix<T> *obj,
                                int dim,
                                Basics::Matrix<T> *dest=0,
                                Basics::Matrix<int32_t> *argmax=0) {
        if (argmax == 0) {
          return MatrixScalarReduceOverDimension(obj, dim,
                                                 (BINARY_SCALAR_CAST)r_max<T>,
                                                 Limits<T>::min(), dest);
        }
        else {
          return MatrixScalarReduceMinMaxOverDimension(obj, dim,
                                                       (BINARY_MINMAX_SCALAR_CAST)r_max2<T>,
                                                       Limits<T>::min(), argmax, dest);
        }
      }

      // TODO: use a wrapper for GPU/CPU
      template <typename T>
      Basics::Matrix<T> *matMax(Basics::SparseMatrix<T> *obj,
                                int dim, Basics::Matrix<T> *dest=0,
                                Basics::Matrix<int32_t> *argmax=0) {
        if (dim != 0 && dim != 1) {
          ERROR_EXIT1(128, "Incorrect given dimension %d\n", dim);
        }
        int ndim = (dim==0)?(1):(0);
        if (dest) {
          if (dest->getDimSize(dim) != 1 ||
              dest->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
        }
        else {
          int result_dims[2] = { obj->getDimSize(0), obj->getDimSize(1) };
          result_dims[dim] = 1;
          dest = new Basics::Matrix<T>(1, result_dims);
        }
        if (argmax) {
          if (argmax->getDimSize(dim) != 1 ||
              argmax->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
          matZeros(argmax);
        }
        matZeros(dest);
        typename Basics::Matrix<T>::random_access_iterator dest_it(dest);
        int aux_dims[2] = { 0, 0 };
        if (argmax == 0) {
          for (typename Basics::SparseMatrix<T>::const_iterator it(obj->begin());
               it!=obj->end(); ++it) {
            int coords[2];
            it.getCoords(coords[0],coords[1]);
            aux_dims[ndim] = coords[ndim];
            dest_it(aux_dims[0],aux_dims[1]) =
              AprilUtils::max(dest_it(aux_dims[0],aux_dims[1]),(*it));
          }
        }
        else {
          typename Basics::Matrix<int32_t>::random_access_iterator argmax_it(argmax);
          for (typename Basics::SparseMatrix<T>::const_iterator it(obj->begin());
               it!=obj->end(); ++it) {
            int coords[2];
            it.getCoords(coords[0],coords[1]);
            aux_dims[ndim] = coords[ndim];
            if (dest_it(aux_dims[0],aux_dims[1]) < *it) {
              dest_it(aux_dims[0],aux_dims[1]) = *it;
              argmax_it(aux_dims[0],aux_dims[1]) = aux_dims[ndim];
            }
          }
        }
        return dest;
      }
    
      // FIXME: using WRAPPER
      template <typename T>
      T matMin(const Basics::Matrix<T> *obj, int &arg_min, int &arg_min_raw_pos) {
        typename Basics::Matrix<T>::const_iterator it(obj->begin());
        typename Basics::Matrix<T>::const_iterator result =
          AprilUtils::argmin(it, typename Basics::Matrix<T>::const_iterator(obj->end()));
        arg_min = result.getIdx();
        arg_min_raw_pos = result.getRawPos();
        return *result;
      }
    
      // FIXME: using WRAPPER
      template <typename T>
      T matMin(const Basics::SparseMatrix<T> *obj, int &c0, int &c1) {
        typename Basics::SparseMatrix<T>::const_iterator it =
          AprilUtils::argmin(obj->begin(),obj->end());
        it.getCoords(c0,c1);
        return *it;
      }

      // FIXME: using WRAPPER
      template<typename T>
      T matMax(const Basics::Matrix<T> *obj, int &arg_max, int &arg_max_raw_pos) {
        typename Basics::Matrix<T>::const_iterator it(obj->begin());
        typename Basics::Matrix<T>::const_iterator result =
          AprilUtils::argmax(it, typename Basics::Matrix<T>::const_iterator(obj->end()));
        arg_max = result.getIdx();
        arg_max_raw_pos = result.getRawPos();
        return *result;
      }
    
      // FIXME: using WRAPPER
      template<typename T>
      T matMax(const Basics::SparseMatrix<T> *obj, int &c0, int &c1) {
        typename Basics::SparseMatrix<T>::const_iterator it =
          AprilUtils::argmax(obj->begin(),obj->end());
        it.getCoords(c0,c1);
        return *it;
      }

      // FIXME: using WRAPPER
      template<typename T>
      void matMinAndMax(const Basics::Matrix<T> *obj, T &min, T &max) {
        if (obj->getMajorOrder() == CblasRowMajor) {
          typename Basics::Matrix<T>::const_iterator it(obj->begin());
          min = *it;
          max = *it;
          for (; it!=obj->end(); ++it) {
            if (*it < min) min = *it;
            if (*it > max) max = *it;
          }
        }
        else {
          typename Basics::Matrix<T>::const_col_major_iterator it(obj->begin());
          min = *it;
          max = *it;
          for (; it!=obj->end(); ++it) {
            if (*it < min) min = *it;
            if (*it > max) max = *it;
          }
        }
      }
    
      template<typename T>
      void matMinAndMax(const Basics::SparseMatrix<T> *obj, T &min, T &max) {
        typename Basics::SparseMatrix<T>::const_iterator it(obj->begin());
        min = max = *it;
        ++it;
        for (; it != obj->end(); ++it) {
          if ( max < (*it) ) max = *it;
          else if ( (*it) < min ) min = *it;
        }
      }
    
      template <typename T>
      Basics::Matrix<T> *matMaxSelDim(const Basics::Matrix<T> *obj,
                                      const int dim,
                                      Int32GPUMirroredMemoryBlock *raw_positions,
                                      const int shift,
                                      Basics::Matrix<T> *result=0) {
        if (dim < 0 || dim > obj->getNumDim()) {
          ERROR_EXIT2(128, "Incorrect dimension %d, numDim=%d\n",
                      dim, obj->getNumDim());
        }
        if (result == 0) {
          result = new Basics::Matrix<T>(1, obj->getDimSize(dim),
                                         obj->getMajorOrder());
        }
        else {
          if (result->size()!=obj->getDimSize(dim) || result->getNumDim()!=1) {
            ERROR_EXIT1(128, "Incorrect result matrix size, "
                        "expected unidimensional matrix with size %d\n",
                        obj->getDimSize(dim));
          }
        }
#ifdef USE_CUDA
        result->setUseCuda(obj->getCudaFlag());
#endif
        int *argmax = 0;
        if (raw_positions != 0) {
          argmax = raw_positions->getPPALForWrite() + shift;
        }
        switch(obj->getNumDim()) {
        case 1:
          ERROR_EXIT(128, "Impossible to compute maxSelDim when numDim=1\n");
          break;
        case 2:
          {
            const int other_dim = 1 - dim;
            T *res_ptr = result->getRawDataAccess()->getPPALForWrite();
            const T *src_ptr = obj->getRawDataAccess()->getPPALForRead();
            for (int i=0; i<obj->getDimSize(dim); ++i, ++res_ptr) {
              int current_raw_pos = obj->getOffset() + i*obj->getStrideSize(dim);
              int raw_pos_max = current_raw_pos;
              *res_ptr = src_ptr[current_raw_pos];
              current_raw_pos += obj->getStrideSize(other_dim);
              for (int j=1; j<obj->getDimSize(other_dim);
                   ++j, current_raw_pos += obj->getStrideSize(other_dim)) {
                if (src_ptr[current_raw_pos] > *res_ptr) {
                  *res_ptr    = src_ptr[current_raw_pos];
                  raw_pos_max = current_raw_pos;
                }
              }
              if (argmax) argmax[i] = raw_pos_max;
            }
            break;
          }
        case 3:
          {
            int other_dim1 = (dim+1)%3;
            int other_dim2 = (dim+2)%3;
            if (other_dim2 < other_dim1) {
              AprilUtils::swap(other_dim1, other_dim2);
            }
            T *res_ptr = result->getRawDataAccess()->getPPALForWrite();
            const T *src_ptr = obj->getRawDataAccess()->getPPALForRead();
            for (int i=0; i<obj->getDimSize(dim); ++i, ++res_ptr) {
              int raw_pos_max = i*obj->getStrideSize(dim) + obj->getOffset();
              *res_ptr = src_ptr[raw_pos_max];
              for (int j=0; j<obj->getDimSize(other_dim1); ++j) {
                int current_raw_pos = obj->getOffset() + i*obj->getStrideSize(dim) + j*obj->getStrideSize(other_dim1);
                for (int k=0; k<obj->getDimSize(other_dim2);
                     ++k, current_raw_pos += obj->getStrideSize(other_dim2)) {
                  if (src_ptr[current_raw_pos] > *res_ptr) {
                    *res_ptr    = src_ptr[current_raw_pos];
                    raw_pos_max = current_raw_pos;
                  }
                }
              }
              if (argmax) argmax[i] = raw_pos_max;
            }
            break;
          }
        case 4:
          {
            int other_dim1 = (dim+1)%4;
            int other_dim2 = (dim+2)%4;
            int other_dim3 = (dim+3)%4;
            if (other_dim1 > other_dim2)
              AprilUtils::swap(other_dim1, other_dim2);
            if (other_dim2 > other_dim3) {
              AprilUtils::swap(other_dim2, other_dim3);
              if (other_dim1 > other_dim2)
                AprilUtils::swap(other_dim1, other_dim2);
            }
            T *res_ptr = result->getRawDataAccess()->getPPALForWrite();
            const T *src_ptr = obj->getRawDataAccess()->getPPALForRead();
            for (int i=0; i<obj->getDimSize(dim); ++i, ++res_ptr) {
              int raw_pos_max = i*obj->getStrideSize(dim) + obj->getOffset();
              *res_ptr = src_ptr[raw_pos_max];
              for (int j=0; j<obj->getDimSize(other_dim1); ++j) {
                for (int k=0; k<obj->getDimSize(other_dim2); ++k) {
                  int current_raw_pos=obj->getOffset()+i*obj->getStrideSize(dim)+j*obj->getStrideSize(other_dim1)+k*obj->getStrideSize(other_dim2);
                  for (int k2=0; k2<obj->getDimSize(other_dim3);
                       ++k2, current_raw_pos += obj->getStrideSize(other_dim3)) {
                    if (src_ptr[current_raw_pos] > *res_ptr) {
                      *res_ptr    = src_ptr[current_raw_pos];
                      raw_pos_max = current_raw_pos;
                    }
                  }
                }
              }
              if (argmax) argmax[i] = raw_pos_max;
            }
            break;
          }
        default:
          {
            T *res_ptr = result->getRawDataAccess()->getPPALForWrite();
            for (int i=0; i<obj->getDimSize(dim); ++i, ++res_ptr) {
              int aux, argmax_raw_pos;
              AprilUtils::SharedPtr< Basics::Matrix<T> >
                current( const_cast<Basics::Matrix<T>*>(obj)->select(dim, i) );
              matMax(current.get(), aux, argmax_raw_pos);
              if (argmax) argmax[i] = argmax_raw_pos;
            }
          }
        }
        return result;
      }

      //////////////////// BOOLEAN CONDITIONS ////////////////////

      /* BOOLEAN CONDITIONS: this methods transforms the given matrix in a
         ZERO/ONE matrix, depending in the truth of the given condition */
    
      template <typename T>
      Basics::Matrix<T> *matLT(Basics::Matrix<T> *obj, const T &value,
                               Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_lt<T>(value), dest);
      }

      template <typename T>
      Basics::Matrix<T> *matLT(Basics::Matrix<T> *obj,
                               const Basics::Matrix<T> *other,
                               Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap2<T,T,T>(obj, other,
                                       (BINARY_SCALAR_CAST)m_lt<T>, dest);
      }

      template <typename T>
      Basics::Matrix<T> *matGT(Basics::Matrix<T> *obj, const T &value,
                               Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_gt<T>(value), dest);
      }

      template <typename T>
      Basics::Matrix<T> *matGT(Basics::Matrix<T> *obj,
                               const Basics::Matrix<T> *other,
                               Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap2<T,T,T>(obj, other,
                                       (BINARY_SCALAR_CAST)m_gt<T>, dest);
      }

      template <typename T>
      Basics::Matrix<T> *matEQ(Basics::Matrix<T> *obj, const T &value,
                               Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        if (m_isnan(value)) {
          return MatrixScalarMap1<T,T>(obj, m_curried_eq_nan<T>(), dest);
        }
        else {
          return MatrixScalarMap1<T,T>(obj, m_curried_eq<T>(value), dest);
        }
      }
    
      template <typename T>
      Basics::Matrix<T> *matEQ(Basics::Matrix<T> *obj,
                               const Basics::Matrix<T> *other,
                               Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap2<T,T>(obj, other,
                                     (BINARY_SCALAR_CAST)m_eq<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matNEQ(Basics::Matrix<T> *obj, const T &value,
                                Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        if (m_isnan(value)) {
          return MatrixScalarMap1<T,T>(obj, m_curried_neq_nan<T>(), dest);
        }
        else {
          return MatrixScalarMap1<T,T>(obj, m_curried_neq<T>(value), dest);
        }
      }
    
      template <typename T>
      Basics::Matrix<T> *matNEQ(Basics::Matrix<T> *obj,
                                const Basics::Matrix<T> *other,
                                Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap2<T,T>(obj, other,
                                     (BINARY_SCALAR_CAST)m_neq<T>, dest);
      }
    
      //////////////////// OTHER MATH OPERATIONS ////////////////////
    
      template <typename T>
      Basics::Matrix<T> *matAddition(const Basics::Matrix<T> *a,
                                     const Basics::Matrix<T> *b,
                                     Basics::Matrix<T> *c = 0) {
        if (c == 0) c = a->clone();
        return matAxpy(c, T(1.0f), b);
      }

      template <typename T>
      Basics::Matrix<T> *matSubstraction(const Basics::Matrix<T> *a,
                                         const Basics::Matrix<T> *b,
                                         Basics::Matrix<T> *c = 0) {
        if (c == 0) c = a->clone();
        return matAxpy(c, T(-1.0f), b);
      }
    
      template <typename T>
      Basics::Matrix<T> *matMultiply(const Basics::Matrix<T> *a,
                                     const Basics::Matrix<T> *b,
                                     Basics::Matrix<T> *c = 0) {
        if (b->isVector()) {
          if (a->isColVector()) {
            // OUTER product
            int dim[2] = {a->size(),b->size()};
            if (c == 0) {
              c = new Basics::Matrix<T>(2, dim, a->getMajorOrder());
            }
            else if (!c->sameDim(dim, 2)) {
              ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                          dim[0], dim[1]);
            }
#ifdef USE_CUDA
            c->setUseCuda(a->getCudaFlag());
#endif
            matGer(matZeros(c), T(1.0f), a, b);
          }
          else if (!a->isVector()) {
            // Matrix-Vector product
            int dim[2] = {a->getDimSize(0),1};
            if (c == 0) {
              c = new Basics::Matrix<T>(b->getNumDim(), dim, a->getMajorOrder());
            }
            else if (!c->sameDim(dim, b->getNumDim())) {
              ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                          dim[0], dim[1]);
            }
#ifdef USE_CUDA
            c->setUseCuda(use_cuda);
#endif
            matGemv(matZeros(c), CblasNoTrans, T(1.0f), a, b, T());
          }
          else {
            // DOT product
            int dim[2] = {1,1};
            if (c == 0) {
              c = new Basics::Matrix<T>(a->getNumDim(), dim, a->getMajorOrder());
            }
            else if (!c->sameDim(dim, a->getNumDim())) {
              ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                          dim[0], dim[1]);
            }
#ifdef USE_CUDA
            c->setUseCuda(use_cuda);
#endif
            c->getRawDataAccess()->putValue( c->getOffset(), matDot(a, b) );
          }
        }
        else if (a->getNumDim() == 2 && b->getNumDim() == 2 &&
                 a->getDimSize(1) == b->getDimSize(0)) {
          // Matrix-Matrix product
          int dim[2] = {a->getDimSize(0), b->getDimSize(1)};
          if (c == 0) {
            c = new Basics::Matrix<T>(2,dim,a->getMajorOrder());
          }
          else if (!c->sameDim(dim,2)) {
            ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                        dim[0], dim[1]);
          }
#ifdef USE_CUDA
          c->setUseCuda(use_cuda);
#endif
          matGemm(matZeros(c), CblasNoTrans, CblasNoTrans,
                  T(1.0f), a, b, T());
        }
        else {
          ERROR_EXIT(128, "Incompatible matrix sizes\n");
        }
        return c;
      }
    
      template <typename T>
      T matSum(const Basics::Matrix<T> *obj) {
        return MatrixSpanSumReduce1<T>(obj, Functors::SpanSumReductor<T>());
      }

      template <>
      ComplexF matSum(const Basics::Matrix<ComplexF> *obj);

      template <typename T>
      T matSum(const Basics::SparseMatrix<T> *obj) {
        return SparseMatrixScalarReduce1<T>(obj,
                                            (BINARY_SCALAR_CAST)r_add<T>,
                                            T(0.0f));
      }
    
      template <typename T>
      Basics::Matrix<T> *matSum(Basics::Matrix<T> *obj,
                                int dim,
                                Basics::Matrix<T> *dest=0) {
        return MatrixScalarReduceOverDimension(obj, dim,
                                               (BINARY_SCALAR_CAST)r_add<T>,
                                               T(0.0f), dest);
      }

      // TODO: Implement using a wrapper for GPU/CPU computation.
      template <typename T>
      Basics::Matrix<T> *matSum(const Basics::SparseMatrix<T> *obj, int dim,
                                Basics::Matrix<T> *dest=0) {
        if (dim != 0 && dim != 1) {
          ERROR_EXIT1(128, "Incorrect given dimension %d\n", dim);
        }
        int ndim = (dim==0)?(1):(0);
        if (dest) {
          if (dest->getDimSize(dim) != 1 ||
              dest->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
        }
        else {
          int result_dims[2] = { obj->getDimSize(0), obj->getDimSize(1) };
          result_dims[dim] = 1;
          dest = new Basics::Matrix<T>(1, result_dims);
        }
        matZeros(dest);
        typename Basics::Matrix<T>::random_access_iterator dest_it(dest);
        int aux_dims[2] = { 0, 0 };
        for (typename Basics::SparseMatrix<T>::const_iterator it(obj->begin());
             it!=obj->end(); ++it) {
          int coords[2];
          it.getCoords(coords[0],coords[1]);
          aux_dims[ndim] = coords[ndim];
          dest_it(aux_dims[0],aux_dims[1]) += (*it);
        }
        return dest;
      }

      /**** COMPONENT WISE OPERATIONS ****/
    
      template <typename T>
      bool matEquals(const Basics::Matrix<T> *a, const Basics::Matrix<T> *b,
                     float epsilon) {
        if (!a->sameDim(b)) return false;
        typename Basics::Matrix<T>::const_iterator a_it(a->begin());
        typename Basics::Matrix<T>::const_iterator b_it(b->begin());
        while(a_it != a->end() && b_it != b->end()) {
          if (!m_relative_equals(*a_it, *b_it, epsilon)) return false;
          ++a_it;
          ++b_it;
        }
        if (a_it != a->end() || b_it != b->end()) return false;
        return true;
      }

      template <typename T>
      bool matEquals(const Basics::SparseMatrix<T> *a,
                     const Basics::SparseMatrix<T> *b,
                     float epsilon) {
        if (!a->sameDim(b)) return false;
        typename Basics::SparseMatrix<T>::const_iterator a_it(a->begin());
        typename Basics::SparseMatrix<T>::const_iterator b_it(b->begin());
        while(a_it != a->end() && b_it != b->end()) {
          int a_c0, a_c1, b_c0, b_c1;
          a_it.getCoords(a_c0, a_c1);
          b_it.getCoords(b_c0, b_c1);
          if (a_c0 != b_c0 || a_c1 != b_c1 ||
              !m_relative_equals(*a_it, *b_it, epsilon)) return false;
          ++a_it;
          ++b_it;
        }
        if (a_it != a->end() || b_it != b->end()) return false;
        return true;
      }
    
      template <typename T>
      Basics::Matrix<T> *matCmul(Basics::Matrix<T> *obj,
                                 const Basics::Matrix<T> *other,
                                 Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap2<T,T,T>(obj, other, (BINARY_SCALAR_CAST)r_mul<T>, dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matScalarAdd(Basics::Matrix<T> *obj, const T &v,
                                      Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_add<T>(v), dest);
      }
    
      template <typename T>
      Basics::Matrix<T> *matAdjustRange(Basics::Matrix<T> *obj,
                                        const T &rmin, const T &rmax,
                                        Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        T mmin, mmax;
        matMinAndMax(obj, mmin, mmax);
        // especial case, set all values to rmin
        if (mmax - mmin == T()) matFill(dest, rmin);
        else {
          const T ratio = (rmax-rmin)/(mmax-mmin);
          if (mmin > T(0.0f) || mmin < T(0.0f)) matScalarAdd(obj, -mmin, dest);
          matScal(dest, ratio);
          if (rmin > (0.0f) || rmin < (0.0f)) matScalarAdd(dest, rmin);
        }
        return dest;
      }
        
      template <typename T>
      Basics::Matrix<T> *matDiv(Basics::Matrix<T> *obj, const T &value,
                                Basics::Matrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_div<T>(value), dest);
      }

      template <typename T>
      Basics::SparseMatrix<T> *matDiv(Basics::SparseMatrix<T> *obj, const T &value,
                                      Basics::SparseMatrix<T> *dest=0) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, m_curried_div<T>(value), dest);
      }

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

    } // namespace Operations
  } // namespace MatrixExt
} // namespace AprilMath

#undef UNARY_SCALAR_CAST
#undef BINARY_SCALAR_CAST
#undef BINARY_MINMAX_SCALAR_CAST
#undef UNARY_SPAN_CAST

#include "matrix-conv.impl.h"

#endif // MATRIX_OPERATIONS_H
