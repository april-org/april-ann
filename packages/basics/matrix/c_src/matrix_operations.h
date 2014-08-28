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
#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <climits> // for INT_MAX
#include "cmath_overloads.h"
#include "mathcore.h"
#include "matrix.h"
#include "mathcore.h"
#include "maxmin.h"
#include "smart_ptr.h"
#include "sparse_matrix.h"

// Must to be defined here.
#include "map_matrix.h"
#include "map_sparse_matrix.h"

#include "reduce_matrix.h"
#include "reduce_sparse_matrix.h"

namespace april_math {
  
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
    
    template <typename T>
    basics::Matrix<T> *matPlogp(basics::Matrix<T> *obj,
                                basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T> (obj, m_plogp<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matLog(basics::Matrix<T> *obj,
                              basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_log<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matLog1p(basics::Matrix<T> *obj,
                                basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_log1p<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matExp(basics::Matrix<T> *obj,
                              basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_exp<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matSqrt(basics::Matrix<T> *obj,
                               basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_sqrt<T>, dest);
    }

    template <typename T>
    basics::SparseMatrix<T> *matSqrt(basics::SparseMatrix<T> *obj,
                                     basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return SparseMatrixScalarMap1<T,T>(obj, m_sqrt<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matPow(basics::Matrix<T> *obj, const T &value,
                              basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_curried_pow<T>(value), dest);
    }

    template <typename T>
    basics::SparseMatrix<T> *matPow(basics::SparseMatrix<T> *obj, const T &value,
                                    basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return SparseMatrixScalarMap1<T,T>(obj, m_curried_pow<T>(value), dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matTan(basics::Matrix<T> *obj,
                              basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_tan<T>, dest);
    }

    template <typename T>
    basics::SparseMatrix<T> *matTan(basics::SparseMatrix<T> *obj,
                                    basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_tan<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matTanh(basics::Matrix<T> *obj,
                               basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_tanh<T>, dest);
    }

    template <typename T>
    basics::SparseMatrix<T> *matTanh(basics::SparseMatrix<T> *obj,
                                     basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_tanh<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matAtan(basics::Matrix<T> *obj,
                               basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_atan<T>, dest);
    }

    template <typename T>
    basics::SparseMatrix<T> *matAtan(basics::SparseMatrix<T> *obj,
                                     basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_atan<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matAtanh(basics::Matrix<T> *obj,
                                basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_atanh<T>, dest);
    }

    template <typename T>
    basics::SparseMatrix<T> *matAtanh(basics::SparseMatrix<T> *obj,
                                      basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_atanh<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matSin(basics::Matrix<T> *obj,
                              basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_sin<T>, dest);
    }

    template <typename T>
    basics::SparseMatrix<T> *matSin(basics::SparseMatrix<T> *obj,
                                    basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_sin<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matSinh(basics::Matrix<T> *obj,
                               basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_sinh<T>, dest);
    }

    template <typename T>
    basics::SparseMatrix<T> *matSinh(basics::SparseMatrix<T> *obj,
                                     basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_sinh<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matAsin(basics::Matrix<T> *obj,
                               basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_asin<T>, dest);
    }
    
    template <typename T>
    basics::SparseMatrix<T> *matAsin(basics::SparseMatrix<T> *obj,
                                     basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_asin<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matAsinh(basics::Matrix<T> *obj,
                                basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_asinh<T>, dest);
    }

    template <typename T>
    basics::SparseMatrix<T> *matAsinh(basics::SparseMatrix<T> *obj,
                                      basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_asinh<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matCos(basics::Matrix<T> *obj,
                              basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_cos<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matCosh(basics::Matrix<T> *obj,
                               basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_cosh<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matAcos(basics::Matrix<T> *obj,
                               basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_acos<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matAcosh(basics::Matrix<T> *obj,
                                basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_acosh<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matAbs(basics::Matrix<T> *obj,
                              basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_abs<T>, dest);
    }

    template <typename T>
    basics::SparseMatrix<T> *matAbs(basics::SparseMatrix<T> *obj,
                                    basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_abs<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matComplement(basics::Matrix<T> *obj,
                                     basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_complement<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matSign(basics::Matrix<T> *obj,
                               basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_sign<T>, dest);
    }
    
    template <typename T>
    basics::SparseMatrix<T> *matSign(basics::SparseMatrix<T> *obj,
                                     basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_sign<T>, dest);
    }
    
    ////////////////// OTHER MAP FUNCTIONS //////////////////
    
    /// Performs a clamp operation, in-place operation if @c dest=0, otherwise,
    /// the result will be computed at the given @c dest Matrix.
    template<typename T>
    basics::Matrix<T> *matClamp(basics::Matrix<T> *obj,
                                const T lower, const T upper,
                                basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_curried_clamp<T>(lower, upper), dest);
    }
    
    template<typename T>
    basics::Matrix<T> *matFill(basics::Matrix<T> *obj, const T value) {
      return MatrixScalarMap1<T,T>(obj, m_curried_fill<T>(value), obj);
    }

    template<typename T>
    basics::SparseMatrix<T> *matFill(basics::SparseMatrix<T> *obj, const T value) {
      return SparseMatrixScalarMap1<T,T>(obj,  m_curried_fill<T>(value), obj);
    }

    template<typename T>
    basics::Matrix<T> *matZeros(basics::Matrix<T> *obj) {
      return matFill(obj, T());
    }

    template<typename T>
    basics::SparseMatrix<T> *matZeros(basics::SparseMatrix<T> *obj) {
      return matFill(obj, T());
    }
    
    template<typename T>
    basics::Matrix<T> *matOnes(basics::Matrix<T> *obj) {
      return matFill(obj, T(1.0f));
    }

    template<typename T>
    basics::SparseMatrix<T> *matOnes(basics::SparseMatrix<T> *obj) {
      return matFill(obj, T(1.0f));
    }
    
    template <typename T>
    basics::Matrix<T> *matDiag(basics::Matrix<T> *obj, const T value) {
      if (obj->getCudaFlag()) {
        ERROR_PRINT("WARNING! DIAG OPERATION NOT IMPLENTED FOR CUDA\n");
      }
      for (int i=1; i<obj->getNumDim(); ++i) {
        if (obj->getDimSize(i) != obj->getDimSize(i-1)) {
          ERROR_EXIT(128, "Only allowed for squared matrices\n");
        }
      }
      typename basics::Matrix<T>::random_access_iterator it(obj);
      april_utils::UniquePtr<int []> aux_coords(new int[obj->getNumDim()]);
      for (int i=0; i<obj->getDimSize(0); ++i) {
        for (int j=0; j<obj->getNumDim(); ++j) aux_coords[j] = i;
        it(aux_coords, obj->getNumDim()) = value;
      }
      return obj;
    }
    
    //////////////////// CBLAS MATH OPERATIONS ////////////////////
    
    // SCOPY BLAS operation this = other
    template <typename T>
    basics::Matrix<T> *matCopy(basics::Matrix<T> *obj,
                               const basics::Matrix<T> *other) {
      return MatrixSpanMap1<T,T>(other, doCopy<T>, obj);
    }

    // SCOPY BLAS operation this = other
    template <typename T>
    basics::SparseMatrix<T> *matCopy(basics::SparseMatrix<T> *obj,
                                     const basics::SparseMatrix<T> *other) {
      return SparseMatrixScalarMap1<T,T>(other, m_identity<T>, obj);
    }

    // Specialization for char
    template <>
    basics::Matrix<char> *matCopy(basics::Matrix<char> *obj,
                                  const basics::Matrix<char> *other) {
      if (obj->size() != other->size()) {
        ERROR_EXIT(128, "Sizes don't match\n");
      }
      typename basics::Matrix<char>::iterator obj_it(obj->begin());
      typename basics::Matrix<char>::const_iterator other_it(obj->begin());
      while(obj_it != obj->end()) {
        april_assert(obj_it != obj->end());
        april_assert(other_it != other->end());
        *obj_it = *other_it;
        ++obj_it;
        ++other_it;
      }
      april_assert(obj_it == obj->end());
      april_assert(other_it == other->end());
      return obj;
    }
  
    // AXPY BLAS operation this = this + alpha * other
    template <typename T>
    basics::Matrix<T> *matAxpy(basics::Matrix<T> *obj, const T alpha,
                               const basics::Matrix<T> *other) {
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
    
    // GEMM BLAS operation C = alpha * op(A)*op(B) + beta*C
    template <typename T>
    basics::Matrix<T> *matGemm(basics::Matrix<T> *C,
                               CBLAS_TRANSPOSE trans_A,
                               CBLAS_TRANSPOSE trans_B,
                               const T alpha,
                               const basics::Matrix<T> *otherA,
                               const basics::Matrix<T> *otherB,
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
      if (trans_A == CblasTrans) april_utils::swap(row_idx_A, col_idx_A);
      if (trans_B == CblasTrans) april_utils::swap(row_idx_B, col_idx_B);
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
    basics::Matrix<T> *matSparseMM(basics::Matrix<T> *C,
                                   CBLAS_TRANSPOSE trans_A,
                                   CBLAS_TRANSPOSE trans_B,
                                   CBLAS_TRANSPOSE trans_C,
                                   const T alpha,
                                   const basics::SparseMatrix<T> *otherA,
                                   const basics::Matrix<T> *otherB,
                                   T beta) {
      if (C == otherB) {
        ERROR_EXIT(128, "Sparse GEMM method couldn't receive as A or B argument "
                   "the C argument\n");
      }
      if (C->getNumDim() != 2 || otherA->getNumDim() != 2 || otherB->numDim != 2) {
        ERROR_EXIT(128,"Incorrect number of dimensions, only allowed for numDim=2\n");
      }
      int row_idx_A = 0, col_idx_A = 1, row_idx_B = 0, col_idx_B = 1;
      int row_idx_C = 0, col_idx_C = 1;
      if (trans_A == CblasTrans) april_utils::swap(row_idx_A, col_idx_A);
      if (trans_B == CblasTrans) april_utils::swap(row_idx_B, col_idx_B);
      if (trans_C == CblasTrans) april_utils::swap(row_idx_C, col_idx_C);
      if (C->getDimSize(row_idx_C) != otherA->getDimSize(row_idx_A) ||
          C->getDimSize(col_idx_C) != otherB->getDimSize(col_idx_B) ||
          otherA->getDimSize(col_idx_A) != otherB->matrixSize[row_idx_B]) {
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
    basics::Matrix<T> matGemv(basics::Matrix<T> *Y,
                              CBLAS_TRANSPOSE trans_A,
                              const T alpha,
                              const basics::Matrix<T> *otherA,
                              const basics::Matrix<T> *otherX,
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
    basics::Matrix<T> *matGemv(basics::Matrix<T> *Y, CBLAS_TRANSPOSE trans_A,
                               const T alpha,
                               const basics::SparseMatrix<T> *otherA,
                               const basics::Matrix<T> *otherX,
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
    basics::Matrix<T> *matGer(basics::Matrix<T> *A,
                              const T alpha,
                              const basics::Matrix<T> *otherX,
                              const basics::Matrix<T> *otherY) {
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
    T matDot(const basics::Matrix<T> *X, const basics::Matrix<T> *Y) {
      if (X->size() != Y->size()) {
        ERROR_EXIT2(128, "Incorrect dimensions: %d dot %d\n",
                    X->size(), Y->size());
      }
      if (X->getMajorOrder() != Y->getMajorOrder()) {
        ERROR_EXIT(128, "Matrices with different major orders\n");
      }
      return MatrixSpanReduce2<T,T,T>(X, Y, doDot<T>, r_add<T>, T(0.0f));
    }

    // DOT Sparse BLAS operation value = dot(this, other)
    template <typename T>
    T matDot(const basics::Matrix<T> *X, const basics::SparseMatrix<T> *Y) {
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
    basics::Matrix<T> *matScal(basics::Matrix<T> *obj, const T value) {
#ifdef USE_MKL
      // INT_MAX avoids OMP parallel for
      return MatrixSpanMap1<T,T>(obj, CurriedScal<T>(value), obj, INT_MAX);
#else
      return MatrixSpanMap1<T,T>(obj, CurriedScal<T>(value), obj);
#endif
    }

    template <typename T>
    basics::SparseMatrix<T> *matScal(basics::SparseMatrix<T> *obj,
                                     const T value) {
      return SparseMatrixScalarMap1<T,T>(obj, m_curried_mul<T>(value), obj);
    }

    template <typename T>
    struct MatrixNorm2Reductor {
      float operator()(const T &a, const T &b) const {
        return static_cast<float>(m_sqrt(a*a + b*b));
      }
    };

    template <typename T>
    float matNorm2(basics::Matrix<T> *obj) {
      return MatrixSpanReduce1(obj, doNrm2<T>, MatrixNorm2Reductor<T>(), T(0.0f));
    }

    template <typename T>
    struct SparseMatrixNorm2 {
      APRIL_CUDA_EXPORT float operator()(const T &a, const T &b) const {
        return static_cast<float>(a + b*b);
      }
    };
    
    template <typename T>
    float matNorm2(basics::SparseMatrix<T> *obj) {
      return m_sqrt(SparseMatrixScalarReduce1(obj, SparseMatrixNorm2<T>(),
                                              r_add<T>,
                                              T(0.0f)));
    }
    
    /////////////////// MAX MIN REDUCTIONS ///////////////////

    // Min and max over given dimension, be careful, argmin and argmax matrices
    // contains the min/max index at the given dimension, but starting in 1 (not
    // in 0)

    template <typename T>
    basics::Matrix<T> *matMin(const basics::Matrix<T> *obj, int dim,
                              basics::Matrix<T> *dest=0,
                              basics::Matrix<int32_t> *argmin=0) {
      if (argmin == 0) {
        return MatrixScalarReduceOverDimension(obj, dim, r_min<T>,
                                               Limits<T>::max(), dest);
      }
      else {
        return MatrixScalarReduceOverDimension2(obj, dim, r_min2<T>,
                                                Limits<T>::max(), argmin, dest);
      }
    }

    // TODO: use a wrapper for GPU/CPU
    template <typename T>
    basics::Matrix<T> *matMin(const basics::SparseMatrix<T> *obj, int dim,
                              basics::Matrix<T> *dest=0,
                              basics::Matrix<int32_t> *argmin=0) {
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
        dest = new basics::Matrix<T>(1, result_dims);
      }
      if (argmin) {
        if (argmin->getDimSize(dim) != 1 ||
            argmin->getDimSize(ndim) != obj->getDimSize(ndim)) {
          ERROR_EXIT(128, "Incorrect matrix sizes\n");
        }
        matZeros(argmin);
      }
      matZeros(dest);
      typename basics::Matrix<T>::random_access_iterator dest_it(dest);
      int aux_dims[2] = { 0, 0 };
      if (argmin == 0) {
        for (typename basics::SparseMatrix<T>::const_iterator it(obj->begin());
             it!=obj->end(); ++it) {
          int coords[2];
          it.getCoords(coords[0],coords[1]);
          aux_dims[ndim] = coords[ndim];
          dest_it(aux_dims[0],aux_dims[1]) =
            april_utils::min(dest_it(aux_dims[0],aux_dims[1]),(*it));
        }
      }
      else {
        typename basics::Matrix<int32_t>::random_access_iterator argmin_it(argmin);
        for (typename basics::SparseMatrix<T>::const_iterator it(obj->begin());
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
    basics::Matrix<T> *matMax(basics::Matrix<T> *obj,
                              int dim, basics::Matrix<T> *dest=0,
                              basics::Matrix<int32_t> *argmax=0) {
      if (argmax == 0) {
        return MatrixScalarReduceOverDimension(obj, dim, r_max<T>,
                                               Limits<T>::min(), dest);
      }
      else {
        return MatrixScalarReduceOverDimension2(obj, dim, r_max2<T>,
                                                Limits<T>::min(), argmax, dest);
      }
    }

    // TODO: use a wrapper for GPU/CPU
    template <typename T>
    basics::Matrix<T> *matMax(basics::SparseMatrix<T> *obj,
                              int dim, basics::Matrix<T> *dest=0,
                              basics::Matrix<int32_t> *argmax=0) {
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
        dest = new basics::Matrix<T>(1, result_dims);
      }
      if (argmax) {
        if (argmax->getDimSize(dim) != 1 ||
            argmax->getDimSize(ndim) != obj->getDimSize(ndim)) {
          ERROR_EXIT(128, "Incorrect matrix sizes\n");
        }
        matZeros(argmax);
      }
      matZeros(dest);
      typename basics::Matrix<T>::random_access_iterator dest_it(dest);
      int aux_dims[2] = { 0, 0 };
      if (argmax == 0) {
        for (typename basics::SparseMatrix<T>::const_iterator it(obj->begin());
             it!=obj->end(); ++it) {
          int coords[2];
          it.getCoords(coords[0],coords[1]);
          aux_dims[ndim] = coords[ndim];
          dest_it(aux_dims[0],aux_dims[1]) =
            april_utils::max(dest_it(aux_dims[0],aux_dims[1]),(*it));
        }
      }
      else {
        typename basics::Matrix<int32_t>::random_access_iterator argmax_it(argmax);
        for (typename basics::SparseMatrix<T>::const_iterator it(obj->begin());
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
    T matMin(const basics::Matrix<T> *obj, int &arg_min, int &arg_min_raw_pos) {
      typename basics::Matrix<T>::const_iterator it(obj->begin());
      typename basics::Matrix<T>::const_iterator result =
        april_utils::argmin(it, basics::Matrix<T>::const_iterator(obj->end()));
      arg_min = result.getIdx();
      arg_min_raw_pos = result.getRawPos();
      return *result;
    }
    
    // FIXME: using WRAPPER
    template <typename T>
    T matMin(const basics::SparseMatrix<T> *obj, int &c0, int &c1) {
      typename basics::SparseMatrix<T>::const_iterator it =
        april_utils::argmin(obj->begin(),obj->end());
      it.getCoords(c0,c1);
      return *it;
    }

    // FIXME: using WRAPPER
    template<typename T>
    T matMax(const basics::Matrix<T> *obj, int &arg_max, int &arg_max_raw_pos) {
      typename basics::Matrix<T>::const_iterator it(obj->begin());
      typename basics::Matrix<T>::const_iterator result =
        april_utils::argmax(it, basics::Matrix<T>::const_iterator(obj->end()));
      arg_max = result.getIdx();
      arg_max_raw_pos = result.getRawPos();
      return *result;
    }
    
    // FIXME: using WRAPPER
    template<typename T>
    T matMax(const basics::SparseMatrix<T> *obj, int &c0, int &c1) {
      typename basics::SparseMatrix<T>::const_iterator it =
        april_utils::argmax(obj->begin(),obj->end());
      it.getCoords(c0,c1);
      return *it;
    }

    // FIXME: using WRAPPER
    template<typename T>
    void matMinAndMax(const basics::Matrix<T> *obj, T &min, T &max) {
      if (obj->getMajorOrder() == CblasRowMajor) {
        typename basics::Matrix<T>::const_iterator it(obj->begin());
        min = *it;
        max = *it;
        for (; it!=obj->end(); ++it) {
          if (*it < min) min = *it;
          if (*it > max) max = *it;
        }
      }
      else {
        typename basics::Matrix<T>::const_col_major_iterator it(obj->begin());
        min = *it;
        max = *it;
        for (; it!=obj->end(); ++it) {
          if (*it < min) min = *it;
          if (*it > max) max = *it;
        }
      }
    }
    
    template<typename T>
    void matMinAndMax(const basics::SparseMatrix<T> *obj, T &min, T &max) {
      typename basics::SparseMatrix<T>::const_iterator it(obj->begin());
      min = max = *it;
      ++it;
      for (; it != obj->end(); ++it) {
        if ( max < (*it) ) max = *it;
        else if ( (*it) < min ) min = *it;
      }
    }
    
    template <typename T>
    basics::Matrix<T> *matMaxSelDim(const basics::Matrix<T> *obj,
                                    const int dim,
                                    Int32GPUMirroredMemoryBlock *raw_positions,
                                    const int shift,
                                    basics::Matrix<T> *result=0) {
      if (dim < 0 || dim > obj->getNumDim()) {
        ERROR_EXIT2(128, "Incorrect dimension %d, numDim=%d\n",
                    dim, obj->getNumDim());
      }
      if (result == 0) {
        result = new basics::Matrix<T>(1, obj->getDimSize(dim),
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
            april_utils::swap(other_dim1, other_dim2);
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
            april_utils::swap(other_dim1, other_dim2);
          if (other_dim2 > other_dim3) {
            april_utils::swap(other_dim2, other_dim3);
            if (other_dim1 > other_dim2)
              april_utils::swap(other_dim1, other_dim2);
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
            april_utils::SharedPtr< basics::Matrix<T> >
              current( const_cast<basics::Matrix<T>*>(obj)->select(dim, i) );
            current->max(aux, argmax_raw_pos);
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
    basics::Matrix<T> *matLT(basics::Matrix<T> *obj, const T &value,
                             basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_curried_lt<T>(value), dest);
    }

    template <typename T>
    basics::Matrix<T> *matLT(basics::Matrix<T> *obj,
                             const basics::Matrix<T> *other,
                             basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap2<T,T,T>(obj, other, m_lt<T>, dest);
    }

    template <typename T>
    basics::Matrix<T> *matGT(basics::Matrix<T> *obj, const T &value,
                             basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_curried_gt<T>(value), dest);
    }

    template <typename T>
    basics::Matrix<T> *matGT(basics::Matrix<T> *obj,
                             const basics::Matrix<T> *other,
                             basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap2<T,T,T>(obj, other, m_gt<T>, dest);
    }

    template <typename T>
    basics::Matrix<T> *matEQ(basics::Matrix<T> *obj, const T &value,
                             basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      if (m_isnan(value)) {
        return MatrixScalarMap1<T,T>(obj, m_curried_eq_nan<T>(), dest);
      }
      else {
        return MatrixScalarMap1<T,T>(obj, m_curried_eq<T>(value), dest);
      }
    }
    
    template <typename T>
    basics::Matrix<T> *matEQ(basics::Matrix<T> *obj,
                             const basics::Matrix<T> *other,
                             basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, other, m_eq<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matNEQ(basics::Matrix<T> *obj, const T &value,
                              basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      if (m_isnan(value)) {
        return MatrixScalarMap1<T,T>(obj, m_curried_neq_nan<T>(), dest);
      }
      else {
        return MatrixScalarMap1<T,T>(obj, m_curried_neq<T>(value), dest);
      }
    }
    
    template <typename T>
    basics::Matrix<T> *matNEQ(basics::Matrix<T> *obj,
                              const basics::Matrix<T> *other,
                              basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, other, m_neq<T>, dest);
    }
    
    //////////////////// OTHER MATH OPERATIONS ////////////////////
    
    template <typename T>
    basics::Matrix<T> *matAddition(const basics::Matrix<T> *a,
                                   const basics::Matrix<T> *b,
                                   basics::Matrix<T> *c = 0) {
      if (c == 0) c = a->clone();
      return matAxpy(c, T(1.0f), b);
    }

    template <typename T>
    basics::Matrix<T> *matSubstraction(const basics::Matrix<T> *a,
                                       const basics::Matrix<T> *b,
                                       basics::Matrix<T> *c = 0) {
      if (c == 0) c = a->clone();
      return matAxpy(c, T(-1.0f), b);
    }
    
    template <typename T>
    basics::Matrix<T> *matMultiply(const basics::Matrix<T> *a,
                                   const basics::Matrix<T> *b,
                                   basics::Matrix<T> *c = 0) {
      if (b->isVector()) {
        if (a->isColVector()) {
          // OUTER product
          int dim[2] = {a->size(),b->size()};
          if (c == 0) {
            c = new basics::Matrix<T>(2, dim, a->getMajorOrder());
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
            c = new basics::Matrix<T>(b->getNumDim(), dim, a->getMajorOrder());
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
            c = new basics::Matrix<T>(a->getNumDim(), dim, a->getMajorOrder());
          }
          else if (!c->sameDim(dim, a->getNumDim())) {
            ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                        dim[0], dim[1]);
          }
#ifdef USE_CUDA
          c->setUseCuda(use_cuda);
#endif
          c->getRawDataAccess()->putValue( matDot(a, b) );
        }
      }
      else if (a->getNumDim() == 2 && b->getNumDim() == 2 &&
               a->getDimSize(1) == b->getDimSize(0)) {
        // Matrix-Matrix product
        int dim[2] = {a->getDimSize(0), b->getDimSize(1)};
        if (c == 0) {
          c = new basics::Matrix<T>(2,dim,a->getMajorOrder());
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
    T matSum(const basics::Matrix<T> *obj) {
      return MatrixScalarSumReduce1(obj, r_add<T>);
    }

    template <typename T>
    T matSum(const basics::SparseMatrix<T> *obj) {
      return SparseMatrixScalarReduce(obj, T(), r_add<T>, r_add<T>);
    }
    
    template <typename T>
    basics::Matrix<T> *matSum(const basics::Matrix<T> *obj, int dim,
                              basics::Matrix<T> *dest=0) {
      return MatrixScalarSumReduce1Dim(obj, r_add<T>, dim, dest);
    }

    // TODO: Implement using a wrapper for GPU/CPU computation.
    template <typename T>
    basics::Matrix<T> *matSum(const basics::SparseMatrix<T> *obj, int dim,
                              basics::Matrix<T> *dest=0) {
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
        dest = new basics::Matrix<T>(1, result_dims);
      }
      matZeros(dest);
      typename basics::Matrix<T>::random_access_iterator dest_it(dest);
      int aux_dims[2] = { 0, 0 };
      for (typename basics::SparseMatrix<T>::const_iterator it(obj->begin());
           it!=obj->end(); ++it) {
        int coords[2];
        it.getCoords(coords[0],coords[1]);
        aux_dims[ndim] = coords[ndim];
        dest_it(aux_dims[0],aux_dims[1]) += (*it);
      }
      return dest;
    }

    /**** COMPONENT WISE OPERATIONS ****/
    template<typename T>
    struct EqualsReductor {
      const m_curried_relative_equals<T> eq_functor;
      EqualsReductor(const T &epsilon) : eq_functor(epsilon) { }
      APRIL_CUDA_EXPORT bool operator()(const bool &acc,
                                        const T &a, const T &b) {
        return acc && eq_functor(a,b);
      }
    };
    template <typename T>
    bool matEquals(const basics::Matrix<T> *a, const basics::Matrix<T> *b,
                   T epsilon) {
      if (!a->sameDim(b)) return false;
      return MatrixScalarReduce2<T,T,bool>(a, b, EqualsReductor<T>(epsilon),
                                           r_and<bool>, true);
    }

    template <typename T>
    bool matEquals(const basics::SparseMatrix<T> *a,
                   const basics::SparseMatrix<T> *b,
                   T epsilon) {
      UNUSED_VARIABLE(a);
      UNUSED_VARIABLE(b);
      UNUSED_VARIABLE(epsilon);
      ERROR_EXIT(128, "NOT IMPLEMENTED!!!\n");
      return false;
    }
    
    template <typename T>
    basics::Matrix<T> *matCmul(basics::Matrix<T> *obj,
                               const basics::Matrix<T> *other,
                               basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap2<T,T,T>(obj, other, r_mul<T>, dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matScalarAdd(basics::Matrix<T> *obj, const T &v,
                                    basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_curried_add<T>(v), dest);
    }
    
    template <typename T>
    basics::Matrix<T> *matAdjustRange(basics::Matrix<T> *obj,
                                      const T &rmin, const T &rmax,
                                      basics::Matrix<T> *dest=0) {
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
    basics::Matrix<T> *matDiv(basics::Matrix<T> *obj, const T &value,
                              basics::Matrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return MatrixScalarMap1<T,T>(obj, m_curried_div<T>(value), dest);
    }

    template <typename T>
    basics::SparseMatrix<T> *matDiv(basics::SparseMatrix<T> *obj, const T &value,
                                    basics::SparseMatrix<T> *dest=0) {
      if (dest == 0) dest = obj;
      return SparseMatrixScalarMap1<T,T>(obj, m_curried_div<T>(value), dest);
    }

    //////////////////// LAPACK MATH OPERATIONS ////////////////////
    
    // FIXME: using WRAPPER for generalized CULA, LAPACK, float and complex
    // numbers

    basics::Matrix<float> *matInv(const basics::Matrix<float> *obj) {
      if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
        ERROR_EXIT(128, "Only bi-dimensional matrices are allowed\n");
      }
      basics::Matrix<float> *A = obj->clone(CblasColMajor);
      april_utils::UniquePtr<int []> IPIV( new int[obj->getDimSize(0)] );
      int INFO;
      INFO = clapack_sgetrf(CblasColMajor,
                            A->getDimSize(0), A->getDimSize(1),
                            A->getRawDataAccess()->getPPALForReadAndWrite(),
                            A->getStrideSize(1),
                            IPIV.get());
      checkLapackInfo(INFO);
      INFO = clapack_sgetri(CblasColMajor,
                            A->getDimSize(0),
                            A->getRawDataAccess()->getPPALForReadAndWrite(),
                            A->getStrideSize(1),
                            IPIV.get());
      checkLapackInfo(INFO);
      return A;
    }
    
    void matSVD(const basics::Matrix<float> *obj,
                basics::Matrix<float> **U, basics::SparseMatrix<float> **S,
                basics::Matrix<float> **VT) {
      if (obj->getNumDim() != 2) {
        ERROR_EXIT(128, "Only bi-dimensional matrices are allowed\n");
      }
      april_utils::SharedPtr< basics::Matrix<float> > A( obj->clone(CblasColMajor) );
      int INFO;
      const int m = A->getDimSize(0); // cols
      const int n = A->getDimSize(1); // rows
      const int lda = A->getStrideSize(1);
      const int numSV = (m<n) ? m : n;
      const int dimsU[2]  = {m, m};
      const int dimsVT[2] = {n, n};
      *U  = new basics::Matrix<float>(2, dimsU,  CblasColMajor);
      *S  = basics::SparseMatrix<float>::diag(numSV, 0.0f, CSR_FORMAT);
      *VT = new basics::Matrix<float>(2, dimsVT, CblasColMajor);
      INFO = clapack_sgesdd(CblasColMajor, m, n, lda,
                            A->getRawDataAccess()->getPPALForReadAndWrite(),
                            (*U)->getRawDataAccess()->getPPALForWrite(),
                            (*S)->getRawValuesAccess()->getPPALForWrite(),
                            (*VT)->getRawDataAccess()->getPPALForWrite());
      checkLapackInfo(INFO);
    }

    // FROM: http://www.r-bloggers.com/matrix-determinant-with-the-lapack-routine-dspsv/    
    april_utils::log_float matLogDeterminant(const basics::Matrix<float> *obj,
                                             float &sign) {
      if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
        ERROR_EXIT(128, "Only squared bi-dimensional matrices are allowed\n");
      }
      april_utils::SharedPtr< basics::Matrix<float> > A( obj->clone(CblasColMajor) );
      april_utils::UniquePtr<int []> IPIV( new int[A->getDimSize(0)] );
      int INFO;
      INFO = clapack_sgetrf(CblasColMajor,
                            A->getDimSize(0), A->getDimSize(1),
                            A->getRawDataAccess()->getPPALForReadAndWrite(),
                            A->getStrideSize(1),
                            IPIV.get());
      checkLapackInfo(INFO);
      basics::Matrix<float>::const_random_access_iterator it(A.get());
      april_utils::log_float det = april_utils::log_float::from_float(it(0,0));
      int row_changes = 0;
#if defined(USE_MKL) || defined(USE_XCODE)
      // in MKL and XCODE the permutation IPIV is one-based
      if (IPIV[0] != 1) ++row_changes;
#else
      // in atlas_lapack IPIV is zero-based
      if (IPIV[0] != 0) ++row_changes;
#endif
      for (int i=1; i<A->getDimSize(0); ++i) {
        const float &v = it(i,i);
        if (v < 0.0f) {
          ERROR_EXIT(128, "Impossible to compute logDeterminant over "
                     "non-positive matrix\n");
        }
        det *= april_utils::log_float::from_float(v);
#if defined(USE_MKL) || defined(USE_XCODE)
        // in MKL and XCODE the permutation IPIV is one-based
        if (IPIV[i] != (i+1)) ++row_changes;
#else
        // in atlas_lapack IPIV is zero-based
        if (IPIV[i] != i) ++row_changes;
#endif
      }
      if ( (row_changes & 1) == 0 ) sign = 1.0f;
      else sign = -1.0f;
      return det;
    }

    // FROM: http://www.r-bloggers.com/matrix-determinant-with-the-lapack-routine-dspsv/    
    double matDeterminant(const basics::Matrix<float> *obj) {
      if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
      ERROR_EXIT(128, "Only squared bi-dimensional matrices are allowed\n");
      }
      april_utils::SharedPtr< basics::Matrix<float> > A( obj->clone(CblasColMajor) );
      april_utils::UniquePtr<int []> IPIV( new int[A->getDimSize(0)] );
      int INFO;
      INFO = clapack_sgetrf(CblasColMajor,
                            A->getDimSize(0), A->getDimSize(1),
                            A->getRawDataAccess()->getPPALForReadAndWrite(),
                            A->getStrideSize(1),
                            IPIV.get());
      checkLapackInfo(INFO);
      basics::Matrix<float>::const_random_access_iterator it(A.get());
      double det = 1.0f;
      int row_changes = 0;
      for (int i=0; i<obj->getDimSize(0); ++i) {
        const float &v = it(i,i);
        det *= v;
#if defined(USE_MKL) || defined(USE_XCODE)
        // in MKL and XCODE the permutation IPIV is one-based
        if (IPIV[i] != (i+1)) ++row_changes;
#else
        // in atlas_lapack IPIV is zero-based
        if (IPIV[i] != i) ++row_changes;
#endif
      }
      double sign;
      if ( (row_changes & 1) == 0 ) sign = 1.0f;
      else sign = -1.0f;
      return sign*det;
    }
    
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
    basics::Matrix<float> *matCholesky(const basics::Matrix<float> *obj,
                                     char uplo) {
      if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
        ERROR_EXIT(128, "Only squared bi-dimensional matrices are allowed\n");
      }
      basics::Matrix<float> *A = obj->clone(CblasColMajor);
      int INFO = clapack_spotrf(CblasColMajor,
                                (uplo == 'U') ? CblasUpper : CblasLower,
                                A->getDimSize(0),
                                A->getRawDataAccess()->getPPALForReadAndWrite(),
                                A->getStrideSize(1));
      checkLapackInfo(INFO);
      switch(uplo) {
      case 'U':
        {
          basics::Matrix<float>::random_access_iterator it(A);
          for (int i=0; i<A->getDimSize(0); ++i) {
            for (int j=0; j<i; ++j) {
              it(i,j) = 0.0f;
            }
          }
        }
        break;
      case 'L':
      default:
        {
          basics::Matrix<float>::random_access_iterator it(A);
          for (int i=0; i<A->getDimSize(0); ++i) {
            for (int j=i+1; j<A->getDimSize(0); ++j) {
              it(i,j) = 0.0f;
            }
          }
        }
      }
      return A;
    }
    
  } // namespace MatrixExt
} // namespace april_math

#include "matrix-conv.impl.h"

#endif // MATRIX_OPERATIONS_H
