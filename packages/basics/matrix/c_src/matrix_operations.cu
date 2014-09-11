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
#include "cmath_overloads.h"
#include "mathcore.h"
#include "matrix.h"
#include "maxmin.h"
#include "smart_ptr.h"
#include "sparse_matrix.h"

// Must be defined in this order.
#include "matrix_operations.h"

// Must to be defined here.
#include "map_matrix.h"
#include "map_sparse_matrix.h"

// Must to be defined here.
#include "reduce_matrix.h"
#include "reduce_sparse_matrix.h"

using Basics::Matrix;
using Basics::SparseMatrix;

namespace AprilMath {
  namespace MatrixExt {

    /// Useful functors for Matrix operations.
    namespace Functors {
      
      template <typename T>
      struct MatrixNorm2Reductor {
        void operator()(float &acc, const T &b) const {
          float b_abs = AprilMath::m_abs(b);
          acc = AprilMath::m_sqrt(acc*acc + b_abs*b_abs);
        }
      };
      
    } // namespace Functors
    
    namespace Operations {
      
      template <typename T>
      Matrix<T> *matPlogp(Matrix<T> *obj,
                          Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T> (obj, AprilMath::Functors::m_plogp<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matLog(Matrix<T> *obj,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_log<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matLog1p(Matrix<T> *obj,
                          Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_log1p<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matExp(Matrix<T> *obj,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_exp<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matSqrt(Matrix<T> *obj,
                         Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sqrt<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matSqrt(SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sqrt<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matPow(Matrix<T> *obj, const T &value,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_pow<T>(value), dest);
      }

      template <typename T>
      SparseMatrix<T> *matPow(SparseMatrix<T> *obj, const T &value,
                              SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, m_curried_pow<T>(value), dest);
      }
    
      template <typename T>
      Matrix<T> *matTan(Matrix<T> *obj,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_tan<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matTan(SparseMatrix<T> *obj,
                              SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_tan<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matTanh(Matrix<T> *obj,
                         Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_tanh<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matTanh(SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_tanh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAtan(Matrix<T> *obj,
                         Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_atan<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matAtan(SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_atan<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAtanh(Matrix<T> *obj,
                          Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_atanh<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matAtanh(SparseMatrix<T> *obj,
                                SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_atanh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matSin(Matrix<T> *obj,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sin<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matSin(SparseMatrix<T> *obj,
                              SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sin<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matSinh(Matrix<T> *obj,
                         Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sinh<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matSinh(SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sinh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAsin(Matrix<T> *obj,
                         Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_asin<T>(), dest);
      }
    
      template <typename T>
      SparseMatrix<T> *matAsin(SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_asin<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAsinh(Matrix<T> *obj,
                          Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_asinh<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matAsinh(SparseMatrix<T> *obj,
                                SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_asinh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matCos(Matrix<T> *obj,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_cos<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matCosh(Matrix<T> *obj,
                         Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_cosh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAcos(Matrix<T> *obj,
                         Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_acos<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAcosh(Matrix<T> *obj,
                          Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_acosh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAbs(Matrix<T> *obj,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_abs<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matAbs(SparseMatrix<T> *obj,
                              SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_abs<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matComplement(Matrix<T> *obj,
                               Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_complement<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matSign(Matrix<T> *obj,
                         Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sign<T>(), dest);
      }
    
      template <typename T>
      SparseMatrix<T> *matSign(SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sign<T>(), dest);
      }
    
      ////////////////// OTHER MAP FUNCTIONS //////////////////
    
      /// Performs a clamp operation, in-place operation if @c dest=0, otherwise,
      /// the result will be computed at the given @c dest Matrix.
      template<typename T>
      Matrix<T> *matClamp(Matrix<T> *obj,
                          const T lower, const T upper,
                          Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_clamp<T>(lower,upper), dest);
      }
    
      template<typename T>
      Matrix<T> *matFill(Matrix<T> *obj, const T value) {
        return MatrixScalarMap1<T,T>(obj, m_curried_fill<T>(value), obj);
      }

      template<typename T>
      SparseMatrix<T> *matFill(SparseMatrix<T> *obj, const T value) {
        return SparseMatrixScalarMap1<T,T>(obj,  m_curried_fill<T>(value), obj);
      }

      template<typename T>
      Matrix<T> *matZeros(Matrix<T> *obj) {
        return matFill(obj, T());
      }

      template<typename T>
      SparseMatrix<T> *matZeros(SparseMatrix<T> *obj) {
        return matFill(obj, T());
      }
    
      template<typename T>
      Matrix<T> *matOnes(Matrix<T> *obj) {
        return matFill(obj, T(1.0f));
      }

      template<typename T>
      SparseMatrix<T> *matOnes(SparseMatrix<T> *obj) {
        return matFill(obj, T(1.0f));
      }
    
      template <typename T>
      Matrix<T> *matDiag(Matrix<T> *obj, const T value) {
        if (obj->getCudaFlag()) {
          ERROR_PRINT("WARNING! DIAG OPERATION NOT IMPLENTED FOR CUDA\n");
        }
        for (int i=1; i<obj->getNumDim(); ++i) {
          if (obj->getDimSize(i) != obj->getDimSize(i-1)) {
            ERROR_EXIT(128, "Only allowed for squared matrices\n");
          }
        }
        typename Matrix<T>::random_access_iterator it(obj);
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
      Matrix<T> *matCopy(Matrix<T> *obj,
                         const Matrix<T> *other) {
        return MatrixSpanMap1<T,T>(other, doCopy<T>, obj);
      }
      
      // SCOPY BLAS operation this = other
      template <typename T>
      SparseMatrix<T> *matCopy(SparseMatrix<T> *obj,
                               const SparseMatrix<T> *other) {
        return SparseMatrixScalarMap1<T,T>(other,
                                           AprilMath::Functors::m_identity<T>(),
                                           obj);
      }

      // AXPY BLAS operation this = this + alpha * other
      template <typename T>
      Matrix<T> *matAxpy(Matrix<T> *obj, const T alpha,
                         const Matrix<T> *other) {
        if (obj->size() != other->size()) {
          ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d\n",
                      obj->size(), other->size());
        }
#ifdef USE_MKL
        // Limit<int>::max() avoids OMP parallel for
        return MatrixSpanMap1<T,T>(other, CurriedAxpy<T>(alpha), obj,
                                   AprilMath::Limits<int>::max());
#else
        return MatrixSpanMap1<T,T>(other, CurriedAxpy<T>(alpha), obj);
#endif
      }

      template <typename T>
      Matrix<T> *matAxpy(Matrix<T> *obj, T alpha,
                         const SparseMatrix<T> *other) {
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
      Matrix<T> *matGemm(Matrix<T> *C,
                         CBLAS_TRANSPOSE trans_A,
                         CBLAS_TRANSPOSE trans_B,
                         const T alpha,
                         const Matrix<T> *otherA,
                         const Matrix<T> *otherB,
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
      Matrix<T> *matSparseMM(Matrix<T> *C,
                             CBLAS_TRANSPOSE trans_A,
                             CBLAS_TRANSPOSE trans_B,
                             CBLAS_TRANSPOSE trans_C,
                             const T alpha,
                             const SparseMatrix<T> *otherA,
                             const Matrix<T> *otherB,
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
      Matrix<T> *matGemv(Matrix<T> *Y,
                         CBLAS_TRANSPOSE trans_A,
                         const T alpha,
                         const Matrix<T> *otherA,
                         const Matrix<T> *otherX,
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
      Matrix<T> *matGemv(Matrix<T> *Y, CBLAS_TRANSPOSE trans_A,
                         const T alpha,
                         const SparseMatrix<T> *otherA,
                         const Matrix<T> *otherX,
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
      Matrix<T> *matGer(Matrix<T> *A,
                        const T alpha,
                        const Matrix<T> *otherX,
                        const Matrix<T> *otherY) {
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
      T matDot(const Matrix<T> *X, const Matrix<T> *Y) {
        if (X->size() != Y->size()) {
          ERROR_EXIT2(128, "Incorrect dimensions: %d dot %d\n",
                      X->size(), Y->size());
        }
        if (X->getMajorOrder() != Y->getMajorOrder()) {
          ERROR_EXIT(128, "Matrices with different major orders\n");
        }
        return MatrixSpanReduce2(X, Y, doDot< T, AprilMath::Functors::r_add<T,T> >,
                                 AprilMath::Functors::r_add<T,T>(),
                                 T(0.0f));
      }

      // DOT Sparse BLAS operation value = dot(this, other)
      template <typename T>
      T matDot(const Matrix<T> *X, const SparseMatrix<T> *Y) {
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
      Matrix<T> *matScal(Matrix<T> *obj, const T value) {
#ifdef USE_MKL
        // Limit<int>::max() avoids OMP parallel for
        return MatrixSpanMap1<T,T>(obj, CurriedScal<T>(value), obj,
                                   AprilMath::Limits<int>::max());
#else
        return MatrixSpanMap1<T,T>(obj, CurriedScal<T>(value), obj);
#endif
      }

      template <typename T>
      SparseMatrix<T> *matScal(SparseMatrix<T> *obj,
                               const T value) {
        return SparseMatrixScalarMap1<T,T>(obj, m_curried_mul<T>(value), obj);
      }

      template <typename T>
      float matNorm2(Matrix<T> *obj) {
        return MatrixSpanReduce1(obj,
                                 doNrm2< T, Functors::MatrixNorm2Reductor<T> >,
                                 Functors::MatrixNorm2Reductor<T>(),
                                 float(0.0f));
      }
      
      // FIXME: implement using a wrapper
      template <typename T>
      float matNorm2(SparseMatrix<T> *obj) {
        float result = 0.0f;
        for (typename SparseMatrix<T>::const_iterator it(obj->begin());
             it != obj->end(); ++it) {
          result += (*it) * (*it);
        }
        return AprilMath::m_sqrt(result);
      }
    
      /////////////////// MAX MIN REDUCTIONS ///////////////////

      // Min and max over given dimension, be careful, argmin and argmax matrices
      // contains the min/max index at the given dimension, but starting in 1 (not
      // in 0)

      template <typename T>
      Matrix<T> *matMin(Matrix<T> *obj,
                        int dim,
                        Matrix<T> *dest,
                        Matrix<int32_t> *argmin) {
        if (argmin == 0) {
          return MatrixScalarReduce1OverDimension(obj, dim,
                                                  AprilMath::Functors::r_min<T>(),
                                                  AprilMath::Functors::r_min<T>(),
                                                  Limits<T>::max(), dest);
        }
        else {
          return MatrixScalarReduceMinMaxOverDimension(obj, dim,
                                                       AprilMath::Functors::r_min2<T>(),
                                                       Limits<T>::max(), argmin, dest);
        }
      }

      // TODO: use a wrapper for GPU/CPU
      template <typename T>
      Matrix<T> *matMin(SparseMatrix<T> *obj, int dim,
                        Matrix<T> *dest,
                        Matrix<int32_t> *argmin) {
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
          dest = new Matrix<T>(1, result_dims);
        }
        if (argmin) {
          if (argmin->getDimSize(dim) != 1 ||
              argmin->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
          matZeros(argmin);
        }
        matZeros(dest);
        typename Matrix<T>::random_access_iterator dest_it(dest);
        int aux_dims[2] = { 0, 0 };
        if (argmin == 0) {
          for (typename SparseMatrix<T>::const_iterator it(obj->begin());
               it!=obj->end(); ++it) {
            int coords[2];
            it.getCoords(coords[0],coords[1]);
            aux_dims[ndim] = coords[ndim];
            dest_it(aux_dims[0],aux_dims[1]) =
              AprilUtils::min(dest_it(aux_dims[0],aux_dims[1]),(*it));
          }
        }
        else {
          typename Matrix<int32_t>::random_access_iterator argmin_it(argmin);
          for (typename SparseMatrix<T>::const_iterator it(obj->begin());
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
      Matrix<T> *matMax(Matrix<T> *obj,
                        int dim,
                        Matrix<T> *dest,
                        Matrix<int32_t> *argmax) {
        if (argmax == 0) {
          return MatrixScalarReduce1OverDimension(obj, dim,
                                                  AprilMath::Functors::r_max<T>(),
                                                  AprilMath::Functors::r_max<T>(),
                                                  Limits<T>::min(), dest);
        }
        else {
          return MatrixScalarReduceMinMaxOverDimension(obj, dim,
                                                       AprilMath::Functors::r_max2<T>(),
                                                       Limits<T>::min(), argmax, dest);
        }
      }

      // TODO: use a wrapper for GPU/CPU
      template <typename T>
      Matrix<T> *matMax(SparseMatrix<T> *obj,
                        int dim, Matrix<T> *dest,
                        Matrix<int32_t> *argmax) {
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
          dest = new Matrix<T>(1, result_dims);
        }
        if (argmax) {
          if (argmax->getDimSize(dim) != 1 ||
              argmax->getDimSize(ndim) != obj->getDimSize(ndim)) {
            ERROR_EXIT(128, "Incorrect matrix sizes\n");
          }
          matZeros(argmax);
        }
        matZeros(dest);
        typename Matrix<T>::random_access_iterator dest_it(dest);
        int aux_dims[2] = { 0, 0 };
        if (argmax == 0) {
          for (typename SparseMatrix<T>::const_iterator it(obj->begin());
               it!=obj->end(); ++it) {
            int coords[2];
            it.getCoords(coords[0],coords[1]);
            aux_dims[ndim] = coords[ndim];
            dest_it(aux_dims[0],aux_dims[1]) =
              AprilUtils::max(dest_it(aux_dims[0],aux_dims[1]),(*it));
          }
        }
        else {
          typename Matrix<int32_t>::random_access_iterator argmax_it(argmax);
          for (typename SparseMatrix<T>::const_iterator it(obj->begin());
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
      T matMin(const Matrix<T> *obj, int &arg_min, int &arg_min_raw_pos) {
        typename Matrix<T>::const_iterator it(obj->begin());
        typename Matrix<T>::const_iterator result =
          AprilUtils::argmin(it, typename Matrix<T>::const_iterator(obj->end()));
        arg_min = result.getIdx();
        arg_min_raw_pos = result.getRawPos();
        return *result;
      }
    
      // FIXME: using WRAPPER
      template <typename T>
      T matMin(const SparseMatrix<T> *obj, int &c0, int &c1) {
        typename SparseMatrix<T>::const_iterator it =
          AprilUtils::argmin(obj->begin(),obj->end());
        it.getCoords(c0,c1);
        return *it;
      }

      // FIXME: using WRAPPER
      template<typename T>
      T matMax(const Matrix<T> *obj, int &arg_max, int &arg_max_raw_pos) {
        typename Matrix<T>::const_iterator it(obj->begin());
        typename Matrix<T>::const_iterator result =
          AprilUtils::argmax(it, typename Matrix<T>::const_iterator(obj->end()));
        arg_max = result.getIdx();
        arg_max_raw_pos = result.getRawPos();
        return *result;
      }
    
      // FIXME: using WRAPPER
      template<typename T>
      T matMax(const SparseMatrix<T> *obj, int &c0, int &c1) {
        typename SparseMatrix<T>::const_iterator it =
          AprilUtils::argmax(obj->begin(),obj->end());
        it.getCoords(c0,c1);
        return *it;
      }

      // FIXME: using WRAPPER
      template<typename T>
      void matMinAndMax(const Matrix<T> *obj, T &min, T &max) {
        if (obj->getMajorOrder() == CblasRowMajor) {
          typename Matrix<T>::const_iterator it(obj->begin());
          min = *it;
          max = *it;
          for (; it!=obj->end(); ++it) {
            if (*it < min) min = *it;
            if (*it > max) max = *it;
          }
        }
        else {
          typename Matrix<T>::const_col_major_iterator it(obj->begin());
          min = *it;
          max = *it;
          for (; it!=obj->end(); ++it) {
            if (*it < min) min = *it;
            if (*it > max) max = *it;
          }
        }
      }
    
      template<typename T>
      void matMinAndMax(const SparseMatrix<T> *obj, T &min, T &max) {
        typename SparseMatrix<T>::const_iterator it(obj->begin());
        min = max = *it;
        ++it;
        for (; it != obj->end(); ++it) {
          if ( max < (*it) ) max = *it;
          else if ( (*it) < min ) min = *it;
        }
      }
    
      template <typename T>
      Matrix<T> *matMaxSelDim(const Matrix<T> *obj,
                              const int dim,
                              Int32GPUMirroredMemoryBlock *raw_positions,
                              const int shift,
                              Matrix<T> *result) {
        if (dim < 0 || dim > obj->getNumDim()) {
          ERROR_EXIT2(128, "Incorrect dimension %d, numDim=%d\n",
                      dim, obj->getNumDim());
        }
        if (result == 0) {
          result = new Matrix<T>(1, obj->getDimSize(dim),
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
              AprilUtils::SharedPtr< Matrix<T> >
                current( const_cast<Matrix<T>*>(obj)->select(dim, i) );
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
      Matrix<T> *matLT(Matrix<T> *obj, const T &value,
                       Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_lt<T>(value), dest);
      }

      template <typename T>
      Matrix<T> *matLT(Matrix<T> *obj,
                       const Matrix<T> *other,
                       Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap2<T,T,T>(obj, other,
                                       AprilMath::Functors::m_lt<T>(), dest);
      }

      template <typename T>
      Matrix<T> *matGT(Matrix<T> *obj, const T &value,
                       Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_gt<T>(value), dest);
      }

      template <typename T>
      Matrix<T> *matGT(Matrix<T> *obj,
                       const Matrix<T> *other,
                       Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap2<T,T,T>(obj, other,
                                       AprilMath::Functors::m_gt<T>(), dest);
      }

      template <typename T>
      Matrix<T> *matEQ(Matrix<T> *obj, const T &value,
                       Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        if (m_isnan(value)) {
          return MatrixScalarMap1<T,T>(obj, m_curried_eq_nan<T>(), dest);
        }
        else {
          return MatrixScalarMap1<T,T>(obj, m_curried_eq<T>(value), dest);
        }
      }
    
      template <typename T>
      Matrix<T> *matEQ(Matrix<T> *obj,
                       const Matrix<T> *other,
                       Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap2<T,T>(obj, other,
                                     AprilMath::Functors::m_eq<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matNEQ(Matrix<T> *obj, const T &value,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        if (m_isnan(value)) {
          return MatrixScalarMap1<T,T>(obj, m_curried_neq_nan<T>(), dest);
        }
        else {
          return MatrixScalarMap1<T,T>(obj, m_curried_neq<T>(value), dest);
        }
      }
    
      template <typename T>
      Matrix<T> *matNEQ(Matrix<T> *obj,
                        const Matrix<T> *other,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap2<T,T>(obj, other,
                                     AprilMath::Functors::m_neq<T>(), dest);
      }
    
      //////////////////// OTHER MATH OPERATIONS ////////////////////
    
      template <typename T>
      Matrix<T> *matAddition(const Matrix<T> *a,
                             const Matrix<T> *b,
                             Matrix<T> *c) {
        if (c == 0) c = a->clone();
        return matAxpy(c, T(1.0f), b);
      }

      template <typename T>
      Matrix<T> *matSubstraction(const Matrix<T> *a,
                                 const Matrix<T> *b,
                                 Matrix<T> *c) {
        if (c == 0) c = a->clone();
        return matAxpy(c, T(-1.0f), b);
      }
    
      template <typename T>
      Matrix<T> *matMultiply(const Matrix<T> *a,
                             const Matrix<T> *b,
                             Matrix<T> *c) {
        if (b->isVector()) {
          if (a->isColVector()) {
            // OUTER product
            int dim[2] = {a->size(),b->size()};
            if (c == 0) {
              c = new Matrix<T>(2, dim, a->getMajorOrder());
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
              c = new Matrix<T>(b->getNumDim(), dim, a->getMajorOrder());
            }
            else if (!c->sameDim(dim, b->getNumDim())) {
              ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                          dim[0], dim[1]);
            }
#ifdef USE_CUDA
            c->setUseCuda(a->getCudaFlag());
#endif
            matGemv(matZeros(c), CblasNoTrans, T(1.0f), a, b, T());
          }
          else {
            // DOT product
            int dim[2] = {1,1};
            if (c == 0) {
              c = new Matrix<T>(a->getNumDim(), dim, a->getMajorOrder());
            }
            else if (!c->sameDim(dim, a->getNumDim())) {
              ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                          dim[0], dim[1]);
            }
#ifdef USE_CUDA
            c->setUseCuda(a->getCudaFlag());
#endif
            c->getRawDataAccess()->putValue( c->getOffset(), matDot(a, b) );
          }
        }
        else if (a->getNumDim() == 2 && b->getNumDim() == 2 &&
                 a->getDimSize(1) == b->getDimSize(0)) {
          // Matrix-Matrix product
          int dim[2] = {a->getDimSize(0), b->getDimSize(1)};
          if (c == 0) {
            c = new Matrix<T>(2,dim,a->getMajorOrder());
          }
          else if (!c->sameDim(dim,2)) {
            ERROR_EXIT2(128, "Incorrect matrix sizes, expected %dx%d\n",
                        dim[0], dim[1]);
          }
#ifdef USE_CUDA
          c->setUseCuda(a->getCudaFlag());
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
      T matSum(const Matrix<T> *obj) {
        return MatrixSpanSumReduce1(obj,
                                    ScalarToSpanReduce1< T, T, AprilMath::Functors::r_add<T,T> >
                                    (AprilMath::Functors::r_add<T,T>()));
      }
      
      template <>
      ComplexF matSum(const Matrix<ComplexF> *obj) {
        return MatrixScalarReduce1(obj,
                                   AprilMath::Functors::r_add<ComplexF,ComplexF>(),
                                   AprilMath::Functors::r_add<ComplexF,ComplexF>(),
                                   ComplexF(0.0f,0.0f));
      }
      
      template <typename T>
      T matSum(const SparseMatrix<T> *obj) {
        return SparseMatrixScalarReduce1<T>(obj,
                                            AprilMath::Functors::r_add<T,T>(),
                                            T(0.0f));
      }
    
      template <typename T>
      Matrix<T> *matSum(Matrix<T> *obj,
                        int dim,
                        Matrix<T> *dest) {
        return MatrixScalarReduce1OverDimension(obj, dim,
                                                AprilMath::Functors::r_add<T,T>(),
                                                AprilMath::Functors::r_add<T,T>(),
                                                T(0.0f), dest);
      }

      // TODO: Implement using a wrapper for GPU/CPU computation.
      template <typename T>
      Matrix<T> *matSum(const SparseMatrix<T> *obj, int dim,
                        Matrix<T> *dest) {
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
          dest = new Matrix<T>(1, result_dims);
        }
        matZeros(dest);
        typename Matrix<T>::random_access_iterator dest_it(dest);
        int aux_dims[2] = { 0, 0 };
        for (typename SparseMatrix<T>::const_iterator it(obj->begin());
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
      bool matEquals(const Matrix<T> *a, const Matrix<T> *b,
                     float epsilon) {
        if (!a->sameDim(b)) return false;
        typename Matrix<T>::const_iterator a_it(a->begin());
        typename Matrix<T>::const_iterator b_it(b->begin());
        while(a_it != a->end() && b_it != b->end()) {
          if (!m_relative_equals(*a_it, *b_it, epsilon)) {
            return false;
          }
          ++a_it;
          ++b_it;
        }
        if (a_it == a->end() && b_it == b->end()) {
          return true;
        }
        else {
          return false;
        }
      }

      template <typename T>
      bool matEquals(const SparseMatrix<T> *a,
                     const SparseMatrix<T> *b,
                     float epsilon) {
        if (!a->sameDim(b)) return false;
        typename SparseMatrix<T>::const_iterator a_it(a->begin());
        typename SparseMatrix<T>::const_iterator b_it(b->begin());
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
      Matrix<T> *matCmul(Matrix<T> *obj,
                         const Matrix<T> *other,
                         Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap2<T,T,T>(obj, other,
                                       AprilMath::Functors::m_mul<T>(),
                                       dest);
      }
    
      template <typename T>
      Matrix<T> *matScalarAdd(Matrix<T> *obj, const T &v,
                              Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_add<T>(v), dest);
      }
    
      template <typename T>
      Matrix<T> *matAdjustRange(Matrix<T> *obj,
                                const T &rmin, const T &rmax,
                                Matrix<T> *dest) {
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
      Matrix<T> *matDiv(Matrix<T> *obj, const T &value,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_div<T>(value), dest);
      }

      template <typename T>
      Basics::Matrix<T> *matDiv(Basics::Matrix<T> *obj,
                                const Basics::Matrix<T> *other,
                                Basics::Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap2<T,T,T>(obj, other,
                                       AprilMath::Functors::m_div<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matDiv(SparseMatrix<T> *obj, const T &value,
                              SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, m_curried_div<T>(value), dest);
      }
      
      Matrix<float> *matInv(const Matrix<float> *obj) {
        if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
          ERROR_EXIT(128, "Only bi-dimensional matrices are allowed\n");
        }
        Matrix<float> *A = obj->clone(CblasColMajor);
        AprilUtils::UniquePtr<int []> IPIV( new int[obj->getDimSize(0)] );
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

      void matSVD(const Matrix<float> *obj,
                  Matrix<float> **U, SparseMatrix<float> **S,
                  Matrix<float> **VT) {
        if (obj->getNumDim() != 2) {
          ERROR_EXIT(128, "Only bi-dimensional matrices are allowed\n");
        }
        AprilUtils::SharedPtr< Matrix<float> > A( obj->clone(CblasColMajor) );
        int INFO;
        const int m = A->getDimSize(0); // cols
        const int n = A->getDimSize(1); // rows
        const int lda = A->getStrideSize(1);
        const int numSV = (m<n) ? m : n;
        const int dimsU[2]  = {m, m};
        const int dimsVT[2] = {n, n};
        *U  = new Matrix<float>(2, dimsU,  CblasColMajor);
        *S  = SparseMatrix<float>::diag(numSV, 0.0f, CSR_FORMAT);
        *VT = new Matrix<float>(2, dimsVT, CblasColMajor);
        INFO = clapack_sgesdd(CblasColMajor, m, n, lda,
                              A->getRawDataAccess()->getPPALForReadAndWrite(),
                              (*U)->getRawDataAccess()->getPPALForWrite(),
                              (*S)->getRawValuesAccess()->getPPALForWrite(),
                              (*VT)->getRawDataAccess()->getPPALForWrite());
        checkLapackInfo(INFO);
      }

      // FROM: http://www.r-bloggers.com/matrix-determinant-with-the-lapack-routine-dspsv/    
      AprilUtils::log_float matLogDeterminant(const Matrix<float> *obj,
                                              float &sign) {
        if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
          ERROR_EXIT(128, "Only squared bi-dimensional matrices are allowed\n");
        }
        AprilUtils::SharedPtr< Matrix<float> > A( obj->clone(CblasColMajor) );
        AprilUtils::UniquePtr<int []> IPIV( new int[A->getDimSize(0)] );
        int INFO;
        INFO = clapack_sgetrf(CblasColMajor,
                              A->getDimSize(0), A->getDimSize(1),
                              A->getRawDataAccess()->getPPALForReadAndWrite(),
                              A->getStrideSize(1),
                              IPIV.get());
        checkLapackInfo(INFO);
        Matrix<float>::const_random_access_iterator it(A.get());
        AprilUtils::log_float det = AprilUtils::log_float::from_float(it(0,0));
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
          det *= AprilUtils::log_float::from_float(v);
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
      double matDeterminant(const Matrix<float> *obj) {
        if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
          ERROR_EXIT(128, "Only squared bi-dimensional matrices are allowed\n");
        }
        AprilUtils::SharedPtr< Matrix<float> > A( obj->clone(CblasColMajor) );
        AprilUtils::UniquePtr<int []> IPIV( new int[A->getDimSize(0)] );
        int INFO;
        INFO = clapack_sgetrf(CblasColMajor,
                              A->getDimSize(0), A->getDimSize(1),
                              A->getRawDataAccess()->getPPALForReadAndWrite(),
                              A->getStrideSize(1),
                              IPIV.get());
        checkLapackInfo(INFO);
        Matrix<float>::const_random_access_iterator it(A.get());
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
      Matrix<float> *matCholesky(const Matrix<float> *obj,
                                 char uplo) {
        if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
          ERROR_EXIT(128, "Only squared bi-dimensional matrices are allowed\n");
        }
        Matrix<float> *A = obj->clone(CblasColMajor);
        int INFO = clapack_spotrf(CblasColMajor,
                                  (uplo == 'U') ? CblasUpper : CblasLower,
                                  A->getDimSize(0),
                                  A->getRawDataAccess()->getPPALForReadAndWrite(),
                                  A->getStrideSize(1));
        checkLapackInfo(INFO);
        switch(uplo) {
        case 'U':
          {
            Matrix<float>::random_access_iterator it(A);
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
            Matrix<float>::random_access_iterator it(A);
            for (int i=0; i<A->getDimSize(0); ++i) {
              for (int j=i+1; j<A->getDimSize(0); ++j) {
                it(i,j) = 0.0f;
              }
            }
          }
        }
        return A;
      }
      
      // INSTANTIATIONS (float type, dense matrix)
      template Matrix<float> *matPlogp(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matLog(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matLog1p(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matExp(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matSqrt(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matPow(Matrix<float> *, const float &, Matrix<float> *);
      template Matrix<float> *matTan(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matTanh(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAtan(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAtanh(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matSin(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matSinh(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAsin(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAsinh(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matCos(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matCosh(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAcos(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAcosh(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAbs(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matComplement(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matSign(Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matClamp(Matrix<float> *, const float,
                                       const float, Matrix<float> *);
      template Matrix<float> *matFill(Matrix<float> *, const float);
      template Matrix<float> *matZeros(Matrix<float> *);
      template Matrix<float> *matOnes(Matrix<float> *);
      template Matrix<float> *matDiag(Matrix<float> *, const float);
      template Matrix<float> *matCopy(Matrix<float> *, const Matrix<float> *);
      template Matrix<float> *matAxpy(Matrix<float> *, const float,
                                      const Matrix<float> *);
      template Matrix<float> *matAxpy(Matrix<float> *, const float,
                                      const SparseMatrix<float> *);
      template Matrix<float> *matGemm(Matrix<float> *,
                                      CBLAS_TRANSPOSE,
                                      CBLAS_TRANSPOSE,
                                      const float,
                                      const Matrix<float> *otherA,
                                      const Matrix<float> *otherB,
                                      float beta);
      template Matrix<float> *matSparseMM(Matrix<float> *,
                                          CBLAS_TRANSPOSE,
                                          CBLAS_TRANSPOSE,
                                          CBLAS_TRANSPOSE,
                                          const float,
                                          const SparseMatrix<float> *,
                                          const Matrix<float> *,
                                          float);
      template Matrix<float> *matGemv(Matrix<float> *Y,
                                      CBLAS_TRANSPOSE,
                                      const float,
                                      const Matrix<float> *,
                                      const Matrix<float> *,
                                      const float);
      template Matrix<float> *matGemv(Matrix<float> *Y,
                                      CBLAS_TRANSPOSE,
                                      const float,
                                      const SparseMatrix<float> *,
                                      const Matrix<float> *,
                                      const float);
      template Matrix<float> *matGer(Matrix<float> *,
                                     const float,
                                     const Matrix<float> *,
                                     const Matrix<float> *);
      template float matDot(const Matrix<float> *, const Matrix<float> *);
      template float matDot(const Matrix<float> *, const SparseMatrix<float> *);
      template Matrix<float> *matScal(Matrix<float> *, const float);
      template float matNorm2(Matrix<float> *);
      template Matrix<float> *matMin(Matrix<float> *,
                                     int,
                                     Matrix<float> *,
                                     Matrix<int32_t> *);
      template Matrix<float> *matMax(Matrix<float> *,
                                     int,
                                     Matrix<float> *,
                                     Matrix<int32_t> *);
      template float matMin(const Matrix<float> *, int &, int &);
      template float matMax(const Matrix<float> *, int &, int &);
      template void matMinAndMax(const Matrix<float> *, float &, float &);
      template Matrix<float> *matMaxSelDim(const Matrix<float> *,
                                           const int,
                                           Int32GPUMirroredMemoryBlock *,
                                           const int,
                                           Basics::Matrix<float> *);
      template Matrix<float> *matLT(Matrix<float> *, const float &,
                                    Matrix<float> *);

      template Matrix<float> *matLT(Matrix<float> *,
                                    const Matrix<float> *,
                                    Matrix<float> *);
      template Matrix<float> *matGT(Matrix<float> *, const float &, Matrix<float> *);
      template Matrix<float> *matGT(Matrix<float> *,
                                    const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matEQ(Matrix<float> *, const float &, Matrix<float> *);
      template Matrix<float> *matEQ(Matrix<float> *,
                                    const Matrix<float> *,
                                    Matrix<float> *);
      template Matrix<float> *matNEQ(Matrix<float> *, const float &,
                                     Matrix<float> *);
      template Matrix<float> *matNEQ(Matrix<float> *,
                                     const Matrix<float> *,
                                     Matrix<float> *);
      template Matrix<float> *matAddition(const Matrix<float> *,
                                          const Matrix<float> *,
                                          Matrix<float> *);

      template Matrix<float> *matSubstraction(const Matrix<float> *,
                                              const Matrix<float> *,
                                              Matrix<float> *);
      template Matrix<float> *matMultiply(const Matrix<float> *,
                                          const Matrix<float> *,
                                          Matrix<float> *);    
      template float matSum(const Matrix<float> *);
      template Matrix<float> *matSum(Matrix<float> *,
                                     int,
                                     Matrix<float> *);
      template bool matEquals(const Matrix<float> *, const Matrix<float> *,
                              float);
      template Matrix<float> *matCmul(Matrix<float> *,
                                      const Matrix<float> *,
                                      Matrix<float> *);
      template Matrix<float> *matScalarAdd(Matrix<float> *, const float &,
                                           Matrix<float> *);
      template Matrix<float> *matAdjustRange(Matrix<float> *,
                                             const float &, const float &,
                                             Matrix<float> *);        
      template Matrix<float> *matDiv(Matrix<float> *, const float &,
                                     Matrix<float> *);
      template Matrix<float> *matDiv(Matrix<float> *,
                                     const Matrix<float> *,
                                     Matrix<float> *);

      // INSTANTIATIONS (double type, dense matrix)
      template Matrix<double> *matPlogp(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matLog(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matLog1p(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matExp(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matSqrt(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matPow(Matrix<double> *, const double &, Matrix<double> *);
      template Matrix<double> *matTan(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matTanh(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAtan(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAtanh(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matSin(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matSinh(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAsin(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAsinh(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matCos(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matCosh(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAcos(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAcosh(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAbs(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matComplement(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matSign(Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matClamp(Matrix<double> *, const double,
                                       const double, Matrix<double> *);
      template Matrix<double> *matFill(Matrix<double> *, const double);
      template Matrix<double> *matZeros(Matrix<double> *);
      template Matrix<double> *matOnes(Matrix<double> *);
      template Matrix<double> *matDiag(Matrix<double> *, const double);
      template Matrix<double> *matCopy(Matrix<double> *, const Matrix<double> *);
      template Matrix<double> *matAxpy(Matrix<double> *, const double,
                                      const Matrix<double> *);
      template Matrix<double> *matAxpy(Matrix<double> *, const double,
                                      const SparseMatrix<double> *);
      template Matrix<double> *matGemm(Matrix<double> *,
                                      CBLAS_TRANSPOSE,
                                      CBLAS_TRANSPOSE,
                                      const double,
                                      const Matrix<double> *otherA,
                                      const Matrix<double> *otherB,
                                      double beta);
      template Matrix<double> *matSparseMM(Matrix<double> *,
                                          CBLAS_TRANSPOSE,
                                          CBLAS_TRANSPOSE,
                                          CBLAS_TRANSPOSE,
                                          const double,
                                          const SparseMatrix<double> *,
                                          const Matrix<double> *,
                                          double);
      template Matrix<double> *matGemv(Matrix<double> *Y,
                                      CBLAS_TRANSPOSE,
                                      const double,
                                      const Matrix<double> *,
                                      const Matrix<double> *,
                                      const double);
      template Matrix<double> *matGemv(Matrix<double> *Y,
                                      CBLAS_TRANSPOSE,
                                      const double,
                                      const SparseMatrix<double> *,
                                      const Matrix<double> *,
                                      const double);
      template Matrix<double> *matGer(Matrix<double> *,
                                     const double,
                                     const Matrix<double> *,
                                     const Matrix<double> *);
      template double matDot(const Matrix<double> *, const Matrix<double> *);
      template double matDot(const Matrix<double> *, const SparseMatrix<double> *);
      template Matrix<double> *matScal(Matrix<double> *, const double);
      template float matNorm2(Matrix<double> *);
      template Matrix<double> *matMin(Matrix<double> *,
                                     int,
                                     Matrix<double> *,
                                     Matrix<int32_t> *);
      template Matrix<double> *matMax(Matrix<double> *,
                                     int,
                                     Matrix<double> *,
                                     Matrix<int32_t> *);
      template double matMin(const Matrix<double> *, int &, int &);
      template double matMax(const Matrix<double> *, int &, int &);
      template void matMinAndMax(const Matrix<double> *, double &, double &);
      template Matrix<double> *matMaxSelDim(const Matrix<double> *,
                                           const int,
                                           Int32GPUMirroredMemoryBlock *,
                                           const int,
                                           Basics::Matrix<double> *);
      template Matrix<double> *matLT(Matrix<double> *, const double &,
                                    Matrix<double> *);

      template Matrix<double> *matLT(Matrix<double> *,
                                    const Matrix<double> *,
                                    Matrix<double> *);
      template Matrix<double> *matGT(Matrix<double> *, const double &, Matrix<double> *);
      template Matrix<double> *matGT(Matrix<double> *,
                                    const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matEQ(Matrix<double> *, const double &, Matrix<double> *);
      template Matrix<double> *matEQ(Matrix<double> *,
                                    const Matrix<double> *,
                                    Matrix<double> *);
      template Matrix<double> *matNEQ(Matrix<double> *, const double &,
                                     Matrix<double> *);
      template Matrix<double> *matNEQ(Matrix<double> *,
                                     const Matrix<double> *,
                                     Matrix<double> *);
      template Matrix<double> *matAddition(const Matrix<double> *,
                                          const Matrix<double> *,
                                          Matrix<double> *);

      template Matrix<double> *matSubstraction(const Matrix<double> *,
                                              const Matrix<double> *,
                                              Matrix<double> *);
      template Matrix<double> *matMultiply(const Matrix<double> *,
                                          const Matrix<double> *,
                                          Matrix<double> *);    
      template double matSum(const Matrix<double> *);
      template Matrix<double> *matSum(Matrix<double> *,
                                     int,
                                     Matrix<double> *);
      template bool matEquals(const Matrix<double> *, const Matrix<double> *,
                              float);
      template Matrix<double> *matCmul(Matrix<double> *,
                                      const Matrix<double> *,
                                      Matrix<double> *);
      template Matrix<double> *matScalarAdd(Matrix<double> *, const double &,
                                           Matrix<double> *);
      template Matrix<double> *matAdjustRange(Matrix<double> *,
                                             const double &, const double &,
                                             Matrix<double> *);        
      template Matrix<double> *matDiv(Matrix<double> *, const double &,
                                     Matrix<double> *);

      // INSTANTIATIONS (ComplexF type, dense matrix)
      template Matrix<ComplexF> *matFill(Matrix<ComplexF> *, const ComplexF);
      template Matrix<ComplexF> *matZeros(Matrix<ComplexF> *);
      template Matrix<ComplexF> *matOnes(Matrix<ComplexF> *);
      template Matrix<ComplexF> *matDiag(Matrix<ComplexF> *, const ComplexF);
      template Matrix<ComplexF> *matCopy(Matrix<ComplexF> *, const Matrix<ComplexF> *);
      template Matrix<ComplexF> *matAxpy(Matrix<ComplexF> *, const ComplexF,
                                      const Matrix<ComplexF> *);
      template Matrix<ComplexF> *matAxpy(Matrix<ComplexF> *, const ComplexF,
                                      const SparseMatrix<ComplexF> *);
      template Matrix<ComplexF> *matGemm(Matrix<ComplexF> *,
                                      CBLAS_TRANSPOSE,
                                      CBLAS_TRANSPOSE,
                                      const ComplexF,
                                      const Matrix<ComplexF> *otherA,
                                      const Matrix<ComplexF> *otherB,
                                      ComplexF beta);
      template Matrix<ComplexF> *matSparseMM(Matrix<ComplexF> *,
                                          CBLAS_TRANSPOSE,
                                          CBLAS_TRANSPOSE,
                                          CBLAS_TRANSPOSE,
                                          const ComplexF,
                                          const SparseMatrix<ComplexF> *,
                                          const Matrix<ComplexF> *,
                                          ComplexF);
      template Matrix<ComplexF> *matGemv(Matrix<ComplexF> *Y,
                                      CBLAS_TRANSPOSE,
                                      const ComplexF,
                                      const Matrix<ComplexF> *,
                                      const Matrix<ComplexF> *,
                                      const ComplexF);
      template Matrix<ComplexF> *matGemv(Matrix<ComplexF> *Y,
                                      CBLAS_TRANSPOSE,
                                      const ComplexF,
                                      const SparseMatrix<ComplexF> *,
                                      const Matrix<ComplexF> *,
                                      const ComplexF);
      template Matrix<ComplexF> *matGer(Matrix<ComplexF> *,
                                     const ComplexF,
                                     const Matrix<ComplexF> *,
                                     const Matrix<ComplexF> *);
      template ComplexF matDot(const Matrix<ComplexF> *, const Matrix<ComplexF> *);
      template ComplexF matDot(const Matrix<ComplexF> *, const SparseMatrix<ComplexF> *);
      template Matrix<ComplexF> *matScal(Matrix<ComplexF> *, const ComplexF);
      template float matNorm2(Matrix<ComplexF> *);
      template Matrix<ComplexF> *matAddition(const Matrix<ComplexF> *,
                                          const Matrix<ComplexF> *,
                                          Matrix<ComplexF> *);

      template Matrix<ComplexF> *matSubstraction(const Matrix<ComplexF> *,
                                                 const Matrix<ComplexF> *,
                                                 Matrix<ComplexF> *);
      template Matrix<ComplexF> *matMultiply(const Matrix<ComplexF> *,
                                             const Matrix<ComplexF> *,
                                             Matrix<ComplexF> *);    
      template ComplexF matSum(const Matrix<ComplexF> *);
      template Matrix<ComplexF> *matSum(Matrix<ComplexF> *,
                                        int,
                                        Matrix<ComplexF> *);
      template bool matEquals(const Matrix<ComplexF> *, const Matrix<ComplexF> *,
                              float);
      template Matrix<ComplexF> *matCmul(Matrix<ComplexF> *,
                                         const Matrix<ComplexF> *,
                                         Matrix<ComplexF> *);
      template Matrix<ComplexF> *matScalarAdd(Matrix<ComplexF> *, const ComplexF &,
                                              Matrix<ComplexF> *);
      template Matrix<ComplexF> *matDiv(Matrix<ComplexF> *, const ComplexF &,
                                        Matrix<ComplexF> *);

      // INSTANTIATIONS (char type, dense matrix)
      template Matrix<char> *matCopy(Matrix<char> *, const Matrix<char> *);
      template Matrix<char> *matFill(Matrix<char> *, const char);
      template Matrix<char> *matZeros(Matrix<char> *);
      template Matrix<char> *matOnes(Matrix<char> *);
      template Matrix<char> *matDiag(Matrix<char> *, const char);
      
      // INSTANTIATIONS (int32_t type, dense matrix)
      template Matrix<int32_t> *matCopy(Matrix<int32_t> *, const Matrix<int32_t> *);
      template Matrix<int32_t> *matFill(Matrix<int32_t> *, const int32_t);
      template Matrix<int32_t> *matZeros(Matrix<int32_t> *);
      template Matrix<int32_t> *matOnes(Matrix<int32_t> *);
      template Matrix<int32_t> *matDiag(Matrix<int32_t> *, const int32_t);
      
      // INSTANTIATIONS (float type, sparse matrix)
      template SparseMatrix<float> *matSqrt(SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matPow(SparseMatrix<float> *,
                                           const float &,
                                           SparseMatrix<float> *);
      template SparseMatrix<float> *matTan(SparseMatrix<float> *,
                                           SparseMatrix<float> *);
      template SparseMatrix<float> *matAtan(SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matTanh(SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matAtanh(SparseMatrix<float> *,
                                             SparseMatrix<float> *);
      template SparseMatrix<float> *matSin(SparseMatrix<float> *,
                                           SparseMatrix<float> *);
      template SparseMatrix<float> *matAsin(SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matSinh(SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matAsinh(SparseMatrix<float> *,
                                             SparseMatrix<float> *);
      template SparseMatrix<float> *matAbs(SparseMatrix<float> *,
                                           SparseMatrix<float> *);
      template SparseMatrix<float> *matSign(SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matFill(SparseMatrix<float> *, const float );
      template SparseMatrix<float> *matZeros(Basics::SparseMatrix<float> *);
      template SparseMatrix<float> *matOnes(SparseMatrix<float> *);
      template SparseMatrix<float> *matCopy(SparseMatrix<float> *,
                                            const SparseMatrix<float> *);
      template SparseMatrix<float> *matScal(SparseMatrix<float> *,
                                            const float );
      template float matNorm2(SparseMatrix<float> *);
      template Matrix<float> *matMin(SparseMatrix<float> *, int ,
                                     Matrix<float> *,
                                     Matrix<int32_t> *);
      template Matrix<float> *matMax(SparseMatrix<float> *,
                                     int, Matrix<float> *,
                                     Matrix<int32_t> *);
      template float matMin(const SparseMatrix<float> *, int &, int &);
      template float matMax(const SparseMatrix<float> *, int &, int &);
      template void matMinAndMax(const SparseMatrix<float> *, float &, float &);
      template float matSum(const SparseMatrix<float> *);
      template Matrix<float> *matSum(const SparseMatrix<float> *, int,
                                     Matrix<float> *);
      template bool matEquals(const SparseMatrix<float> *,
                              const SparseMatrix<float> *,
                              float);
      template SparseMatrix<float> *matDiv(SparseMatrix<float> *, const float &,
                                           SparseMatrix<float> *);

    } // namespace Operations
  } // namespace MatrixExt
} // namespace AprilMath
