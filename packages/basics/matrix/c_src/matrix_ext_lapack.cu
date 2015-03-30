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
#include "matrix_ext_lapack.h"

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
    
    namespace LAPACK {
      
      Matrix<float> *matInv(const Matrix<float> *obj) {
        if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
          ERROR_EXIT(128, "Only squared bi-dimensional matrices are allowed\n");
        }
        Matrix<float> *A = obj->clone();
        AprilUtils::UniquePtr<int []> IPIV( new int[obj->getDimSize(0)] );
        int INFO;
        INFO = clapack_sgetrf(CblasRowMajor,
                              A->getDimSize(0), A->getDimSize(1),
                              A->getRawDataAccess()->getPPALForReadAndWrite(),
                              A->getStrideSize(0),
                              IPIV.get());
        checkLapackInfo(INFO);
        INFO = clapack_sgetri(CblasRowMajor,
                              A->getDimSize(0),
                              A->getRawDataAccess()->getPPALForReadAndWrite(),
                              A->getStrideSize(0),
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
        AprilUtils::SharedPtr< Matrix<float> > A( obj->clone() );
        AprilUtils::SharedPtr< Matrix<float> > AT( A->transpose() );
        int INFO;
        const int m = A->getDimSize(0); // cols
        const int n = A->getDimSize(1); // rows
        const int numSV = (m<n) ? m : n;
        const int dimsU[2]  = {m, m};
        const int dimsVT[2] = {n, n};
        GPUMirroredMemoryBlock<float> *S_values =
          new GPUMirroredMemoryBlock<float>(numSV);
        GPUMirroredMemoryBlock<int32_t> *S_indices =
          new GPUMirroredMemoryBlock<int32_t>(numSV);
        GPUMirroredMemoryBlock<int32_t> *S_first =
          new GPUMirroredMemoryBlock<int32_t>(m+1);
        for (int i=0; i<numSV; ++i) {
          (*S_indices)[i]=i;
          (*S_first)[i]=i;
        }
        for (int i=numSV; i<=m; ++i) {
          (*S_first)[i]=numSV;
        }
        *U  = new Matrix<float>(2, dimsU);
        *S  = new SparseMatrix<float>(m,n,S_values,S_indices,S_first);
        *VT = new Matrix<float>(2, dimsVT);
        AprilUtils::SharedPtr< Matrix<float> > UT( (*VT)->transpose() );
        AprilUtils::SharedPtr< Matrix<float> > V( (*U)->transpose() );
        // m,n are changed by n,m because the tranposition of the matrices
        INFO = clapack_sgesdd(CblasColMajor, n, m, AT->getStrideSize(1),
                              AT->getRawDataAccess()->getPPALForReadAndWrite(),
                              UT->getRawDataAccess()->getPPALForWrite(),
                              S_values->getPPALForWrite(),
                              V->getRawDataAccess()->getPPALForWrite());
        checkLapackInfo(INFO);
      }

      // FROM: http://www.r-bloggers.com/matrix-determinant-with-the-lapack-routine-dspsv/    
      AprilUtils::log_float matLogDeterminant(const Matrix<float> *obj,
                                              float &sign) {
        if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
          ERROR_EXIT(128, "Only squared bi-dimensional matrices are allowed\n");
        }
        AprilUtils::SharedPtr< Matrix<float> > A( obj->clone() );
        AprilUtils::SharedPtr< Matrix<float> > AT( A->transpose() ); // in col major
        AprilUtils::UniquePtr<int []> IPIV( new int[A->getDimSize(0)] );
        int INFO;
        INFO = clapack_sgetrf(CblasColMajor,
                              AT->getDimSize(0), AT->getDimSize(1),
                              AT->getRawDataAccess()->getPPALForReadAndWrite(),
                              AT->getStrideSize(1),
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
        AprilUtils::SharedPtr< Matrix<float> > A( obj->clone() );
        AprilUtils::SharedPtr< Matrix<float> > AT( A->transpose() ); // in col major
        AprilUtils::UniquePtr<int []> IPIV( new int[A->getDimSize(0)] );
        int INFO;
        INFO = clapack_sgetrf(CblasColMajor,
                              AT->getDimSize(0), AT->getDimSize(1),
                              AT->getRawDataAccess()->getPPALForReadAndWrite(),
                              AT->getStrideSize(1),
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
        Matrix<float> *A = obj->clone();
        int INFO = clapack_spotrf(CblasRowMajor,
                                  (uplo == 'U') ? CblasUpper : CblasLower,
                                  A->getDimSize(0),
                                  A->getRawDataAccess()->getPPALForReadAndWrite(),
                                  A->getStrideSize(0));
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

    } // namespace LAPACK
    
  } // namespace MatrixExt
} // namespace AprilMath
