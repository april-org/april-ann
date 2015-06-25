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
#include "realfftwithhamming.h"
#include "smart_ptr.h"
#include "sparse_matrix.h"

// Must be defined in this order.
#include "matrix_ext_blas.h"

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
    
    namespace BLAS {
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
        return MatrixSpanMap1<T,T>(other, CurriedAxpy<T>(alpha), obj);
      }

      template <typename T>
      Matrix<T> *matAxpy(Matrix<T> *obj, T alpha,
                         const SparseMatrix<T> *other) {
        if (obj->size() != other->size()) {
          ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d\n",
                      obj->size(), other->size());
        }
        if ( (other->getSparseFormat() == CSR_FORMAT &&
              other->getDimSize(0) != 1) ||
             (other->getSparseFormat() == CSC_FORMAT &&
              other->getDimSize(1) != 1) ) {
          // bi-dimensional case
          if (other->getDimSize(0) != obj->getDimSize(0) ||
              other->getDimSize(1) != obj->getDimSize(1)) {
            ERROR_EXIT4(128, "Incompatible matrix sizes, sparse is %dx%d, "
                        "dense is %dx%d\n",
                        other->getDimSize(0), other->getDimSize(1),
                        obj->getDimSize(0), obj->getDimSize(1));
          }
          bool cuda_flag = obj->getCudaFlag() || other->getCudaFlag();
          const GPUMirroredMemoryBlock<T> *values = other->getRawValuesAccess();
          const Int32GPUMirroredMemoryBlock *indices = other->getRawIndicesAccess();
          const int32_t *first_index = other->getRawFirstIndexAccess()->getPPALForRead();;
          AprilUtils::SharedPtr< Matrix<T> > slice;
          unsigned int dim = (other->getSparseFormat() == CSR_FORMAT) ? 0 : 1;
          for (int i=0; i<other->getDimSize(dim); ++i) {
            slice = obj->select(dim, i, slice.get());
            doSparseAxpy(first_index[i+1] - first_index[i], alpha,
                         values,
                         indices,
                         slice->getRawDataAccess(),
                         first_index[i],
                         static_cast<unsigned int>(slice->getOffset()),
                         static_cast<unsigned int>(slice->getStrideSize(0)),
                         cuda_flag);
          }
        }
        else {
          // vector case
          if (!obj->isVector()) {
            ERROR_EXIT(128, "sparse AXPY only works with vectors\n");
          }
          doSparseAxpy(other->nonZeroSize(), alpha,
                       other->getRawValuesAccess(),
                       other->getRawIndicesAccess(),
                       obj->getRawDataAccess(), 0,
                       static_cast<unsigned int>(obj->getOffset()),
                       static_cast<unsigned int>(obj->getVectorStride()),
                       obj->getCudaFlag() || other->getCudaFlag());
        }
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
        int aux_A_stride[2], aux_B_stride[2], aux_C_stride[2];
        //
        CBLAS_ORDER order = CblasRowMajor;
        if (C == otherA || C == otherB) {
          ERROR_EXIT(128, "GEMM method cannot receive as A or B argument "
                     "the C argument\n");
        }
        if (C->getNumDim() != 2 ||
            otherA->getNumDim() != 2 ||
            otherB->getNumDim() != 2) {
          ERROR_EXIT(128,"Incorrect number of dimensions, only allowed for numDim=2\n");
        }
        int row_idx_A = 0, col_idx_A = 1;
        int row_idx_B = 0, col_idx_B = 1;
        //
        const int *A_stride = otherA->getStridePtr();
        const int *B_stride = otherB->getStridePtr();
        const int *C_stride = C->getStridePtr();
        const int *A_dim = otherA->getDimPtr();
        const int *B_dim = otherB->getDimPtr();
        const int *C_dim = C->getDimPtr();
        // case of transposed matrix
        if (C_stride[1] != 1) {
          order = CblasColMajor;
        }
        else {
          // case of transposed col vector
          if (C_stride[0] == C_stride[1] && C_dim[1] != 1) {
            C_stride = aux_C_stride;
            aux_C_stride[0] = C_dim[1];
            aux_C_stride[1] = 1;
          }
        }
        if (A_stride[0] == A_stride[1] && A_stride[0] == 1 && A_dim[1] != 1) {
          A_stride = aux_A_stride;
          aux_A_stride[0] = A_dim[1];
          aux_A_stride[1] = 1;
        }
        if (B_stride[0] == B_stride[1] && B_stride[0] == 1 && B_dim[1] != 1) {
          B_stride = aux_B_stride;
          aux_B_stride[0] = B_dim[1];
          aux_B_stride[1] = 1;
        }
        //
        int lda = AprilUtils::max(A_stride[0], A_stride[1]);
        int ldb = AprilUtils::max(B_stride[0], B_stride[1]);
        int ldc = AprilUtils::max(C_stride[0], C_stride[1]);
        if (A_stride[0] + A_stride[1] != lda+1 ||
            B_stride[0] + B_stride[1] != ldb+1 ||
            C_stride[0] + C_stride[1] != ldc+1) {
          ERROR_EXIT(128, "Only allowed with contiguous matrices in leading dimension\n");
        }
        //
        if (trans_A == CblasTrans) AprilUtils::swap(row_idx_A, col_idx_A);
        if (trans_B == CblasTrans) AprilUtils::swap(row_idx_B, col_idx_B);
        if (C_dim[0] != A_dim[row_idx_A] ||
            C_dim[1] != B_dim[col_idx_B] ||
            A_dim[col_idx_A] != B_dim[row_idx_B]) {
          ERROR_EXIT6(128, "Incorrect matrixes dimensions: %dx%d + %dx%d * %dx%d\n",
                      C_dim[0], C_dim[1],
                      A_dim[row_idx_A], A_dim[col_idx_A],
                      B_dim[row_idx_B], B_dim[col_idx_B]);
        }
        int M = C_dim[0], N=C_dim[1];
        int K = A_dim[col_idx_A];
        if (order == CblasRowMajor) {
          if (A_stride[1] != 1) {
            trans_A = NEGATE_CBLAS_TRANSPOSE(trans_A);
          }
          if (B_stride[1] != 1) {
            trans_B = NEGATE_CBLAS_TRANSPOSE(trans_B);
          }
        }
        else {
          if (A_stride[0] != 1) {
            trans_A = NEGATE_CBLAS_TRANSPOSE(trans_A);
          }
          if (B_stride[0] != 1) {
            trans_B = NEGATE_CBLAS_TRANSPOSE(trans_B);
          }
        }
        //
        doGemm(order, trans_A, trans_B,
               M, N, K,
               alpha, otherA->getRawDataAccess(), lda,
               otherB->getRawDataAccess(), ldb,
               beta, C->getRawDataAccess(), ldc,
               otherA->getOffset(), otherB->getOffset(), C->getOffset(),
               C->getCudaFlag() || otherA->getCudaFlag() || otherB->getCudaFlag());
        return C;
      }

      // MM Sparse BLAS operation C = alpha * op(A)*op(B) + beta*op(C)
      template <typename T>
      Matrix<T> *matSparseMM(Matrix<T> *C,
                             CBLAS_TRANSPOSE trans_A,
                             CBLAS_TRANSPOSE trans_B,
                             const T alpha,
                             const SparseMatrix<T> *otherA,
                             const Matrix<T> *otherB,
                             T beta) {
        int aux_B_stride[2], aux_C_stride[2];
        //
        CBLAS_ORDER order = CblasRowMajor;
        if (C == otherB) {
          ERROR_EXIT(128, "Sparse GEMM method couldn't receive as A or B argument "
                     "the C argument\n");
        }
        if (C->getNumDim() != 2 ||
            otherA->getNumDim() != 2 ||
            otherB->getNumDim() != 2) {
          ERROR_EXIT(128,"Incorrect number of dimensions, only allowed for numDim=2\n");
        }
        int row_idx_A = 0, col_idx_A = 1;
        int row_idx_B = 0, col_idx_B = 1;
        //
        const int *B_stride = otherB->getStridePtr();
        const int *C_stride = C->getStridePtr();
        const int *A_dim = otherA->getDimPtr();
        const int *B_dim = otherB->getDimPtr();
        const int *C_dim = C->getDimPtr();
        // case of transposed matrix
        if (C_stride[1] != 1) {
          order = CblasColMajor;
        }
        else {
          // case of transposed col vector
          if (C_stride[0] == C_stride[1] && C_dim[1] != 1) {
            C_stride = aux_C_stride;
            aux_C_stride[0] = C_dim[1];
            aux_C_stride[1] = 1;
          }
        }
        if (B_stride[0] == B_stride[1] && B_stride[0] == 1 && B_dim[1] != 1) {
          B_stride = aux_B_stride;
          aux_B_stride[0] = B_dim[1];
          aux_B_stride[1] = 1;
        }
        //
        int ldb = AprilUtils::max(B_stride[0], B_stride[1]);
        int ldc = AprilUtils::max(C_stride[0], C_stride[1]);
        if (B_stride[0] + B_stride[1] != ldb+1 ||
            C_stride[0] + C_stride[1] != ldc+1) {
          ERROR_EXIT(128, "Only allowed with contiguous matrices in leading dimension\n");
        }
        //
        if (trans_A == CblasTrans) AprilUtils::swap(row_idx_A, col_idx_A);
        if (trans_B == CblasTrans) AprilUtils::swap(row_idx_B, col_idx_B);
        //
        if (C_dim[0] != A_dim[row_idx_A] ||
            C_dim[1] != B_dim[col_idx_B] ||
            A_dim[col_idx_A] != B_dim[row_idx_B]) {
          ERROR_EXIT6(128, "Incorrect matrixes dimensions: %dx%d + %dx%d * %dx%d\n",
                      C_dim[0], C_dim[1],
                      A_dim[row_idx_A], A_dim[col_idx_A],
                      B_dim[row_idx_B], B_dim[col_idx_B]);
        }
        int M=C_dim[0], N=C_dim[1];
        int K=B_dim[row_idx_B];
        if (order == CblasRowMajor) {
          if (B_stride[1] != 1) trans_B = NEGATE_CBLAS_TRANSPOSE(trans_B);
        }
        else {
          if (B_stride[0] != 1) trans_B = NEGATE_CBLAS_TRANSPOSE(trans_B);
        }
        //
        doSparseMM<T>(order,
                      otherA->getSparseFormat(),
                      trans_A,
                      trans_B,
                      M, N, K,
                      alpha,
                      otherA->getRawValuesAccess(),
                      otherA->getRawIndicesAccess(),
                      otherA->getRawFirstIndexAccess(),
                      otherB->getRawDataAccess(), ldb,
                      beta, C->getRawDataAccess(), ldc,
                      otherB->getOffset(), C->getOffset(),
                      C->getCudaFlag() || otherA->getCudaFlag() || otherB->getCudaFlag());
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
        CBLAS_ORDER order = CblasRowMajor;
        if (!Y->isVector() || !otherX->isVector() || otherA->getNumDim() != 2) {
          ERROR_EXIT(128,"Incorrect number of dimensions\n");
        }
        int M, N;
        const int *A_stride = otherA->getStridePtr();
        int lda = AprilUtils::max(A_stride[0],A_stride[1]);
        if (A_stride[0]+A_stride[1] != lda+1) {
          ERROR_EXIT(128, "Only allowed with contiguous matrices in leading dimension\n");
        }
        if (A_stride[1] != 1) order = CblasColMajor;
        M = otherA->getDimSize(0);
        N = otherA->getDimSize(1);
        // SANITY CHECK
        if (trans_A == CblasTrans) {
          if (N != Y->size() || M != otherX->size()) {
            ERROR_EXIT4(128, "Incorrect matrixes dimensions: %dx1 + %dx%d * %dx1\n",
                        Y->size(), N, M, otherX->size());
          }
        }
        else {
          if (M != Y->size() || N != otherX->size()) {
            ERROR_EXIT4(128, "Incorrect matrixes dimensions: %dx1 + %dx%d * %dx1\n",
                        Y->size(), M, N, otherX->size());
          }
        }
        //
        int ldx=otherX->getVectorStride();
        int ldy=Y->getVectorStride();
        doGemv(order, trans_A,
               M, N,
               alpha, otherA->getRawDataAccess(), lda,
               otherX->getRawDataAccess(), ldx,
               beta, Y->getRawDataAccess(), ldy,
               otherA->getOffset(), otherX->getOffset(),
               Y->getOffset(),
               Y->getCudaFlag() || otherX->getCudaFlag() || otherA->getCudaFlag());
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
                     Y->getCudaFlag() || otherX->getCudaFlag() || otherA->getCudaFlag());
        return Y;
      }
  
      // GER BLAS operation A = alpha * X*Y' + A
      template <typename T>
      Matrix<T> *matGer(Matrix<T> *A,
                        const T alpha,
                        const Matrix<T> *otherX,
                        const Matrix<T> *otherY) {
        CBLAS_ORDER order = CblasRowMajor;
        if (!otherX->isVector() || !otherY->isVector() || A->getNumDim()!=2) {
          ERROR_EXIT(128,"Incorrect number of dimensions\n");
        }
        const int *A_stride = A->getStridePtr();
        int lda = AprilUtils::max(A_stride[0],A_stride[1]);
        if (A->getStrideSize(1) != 1) order = CblasColMajor;
        int M=otherX->size(), N=otherY->size();
        if (A->getDimSize(0) != M || A->getDimSize(1) != N) {
          ERROR_EXIT4(128, "Incorrect matrixes dimensions: %dx%d + %dx1 * 1x%d\n",
                      A->getDimSize(0), A->getDimSize(1), M, N);
        }
        int ldx=otherX->getVectorStride();
        int ldy=otherY->getVectorStride();
        doGer(order,
              M, N,
              alpha, otherX->getRawDataAccess(), otherX->getOffset(), ldx,
              otherY->getRawDataAccess(), otherY->getOffset(), ldy,
              A->getRawDataAccess(), A->getOffset(), lda,
              A->getCudaFlag() || otherX->getCudaFlag() || otherY->getCudaFlag());
        return A;
      }

      // DOT BLAS operation value = dot(X, Y)
      template <typename T>
      T matDot(const Matrix<T> *X, const Matrix<T> *Y) {
        if (X->size() != Y->size()) {
          ERROR_EXIT2(128, "Incorrect dimensions: %d dot %d\n",
                      X->size(), Y->size());
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
                            X->getCudaFlag() || Y->getCudaFlag());
        return ret;
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

      ///// INTEGER VERSIONS /////
      
      template<typename T>
      struct GenericAxpyFunctor {
        T alpha;
        GenericAxpyFunctor(const T &alpha) : alpha(alpha) {}
        APRIL_CUDA_EXPORT T operator()(const T &x, const T &y) {
          return alpha*x + y;
        }
      };
      
      // AXPY integer operation this = this + alpha * other
      template <>
      Matrix<int32_t> *matAxpy(Matrix<int32_t> *obj, const int32_t alpha,
                               const Matrix<int32_t> *other) {
        if (obj->size() != other->size()) {
          ERROR_EXIT2(128, "Incorrect matrices sizes: %d != %d\n",
                      obj->size(), other->size());
        }
        if (alpha != 1) {
          return MatrixScalarMap2<int32_t,
                                  int32_t>(other, obj,
                                           GenericAxpyFunctor<int32_t>(alpha),
                                           obj);
        }
        else {
          return MatrixScalarMap2<int32_t,
                                  int32_t>(other, obj,
                                           AprilMath::Functors::m_add<int32_t>(),
                                           obj);
        }
      }

      ///////////////////////////

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
      template float matNorm2(Matrix<float> *);
      
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
      template float matNorm2(Matrix<double> *);
      
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
      template float matNorm2(Matrix<ComplexF> *);
      
      template Matrix<char> *matCopy(Matrix<char> *, const Matrix<char> *);
      
      template Matrix<bool> *matCopy(Matrix<bool> *, const Matrix<bool> *);
      
      template Matrix<int32_t> *matCopy(Matrix<int32_t> *, const Matrix<int32_t> *);

      template SparseMatrix<float> *matCopy(SparseMatrix<float> *,
                                            const SparseMatrix<float> *);
      template float matNorm2(SparseMatrix<float> *);

      template float matNorm2(SparseMatrix<double> *);

      
      template SparseMatrix<double> *matCopy(SparseMatrix<double> *,
                                             const SparseMatrix<double> *);
      
      
      template SparseMatrix<ComplexF> *matCopy(SparseMatrix<ComplexF> *,
                                               const SparseMatrix<ComplexF> *);
      
      

    } // namespace BLAS
    
  } // namespace MatrixExt
} // namespace AprilMath
