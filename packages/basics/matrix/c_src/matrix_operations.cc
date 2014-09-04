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
#include "matrix.h"
#include "sparse_matrix.h"

// Must be defined in this order.
#include "matrix_operations.h"

namespace AprilMath {
  namespace MatrixExt {
    namespace Operations {
      
      namespace Generic {
        template <typename T>
        Basics::Matrix<T> *matCopy(Basics::Matrix<T> *obj,
                                   const Basics::Matrix<T> *other) {
          if (obj->size() != other->size()) {
            ERROR_EXIT(128, "Sizes don't match\n");
          }
          typename Basics::Matrix<T>::iterator obj_it(obj->begin());
          typename Basics::Matrix<T>::const_iterator other_it(obj->begin());
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
        
      } // namespace Generic

      template <>
      ComplexF matSum(const Basics::Matrix<ComplexF> *obj) {
        return MatrixScalarReduce1<ComplexF>(obj,
                                             (m_complexf_binary_complexf_map_t)r_add<ComplexF>,
                                             ComplexF(0.0f,0.0f));
      }
      
      // Specialization for char
      template <>
      Basics::Matrix<char> *matCopy(Basics::Matrix<char> *obj,
                                    const Basics::Matrix<char> *other) {
        return AprilMath::MatrixExt::Operations::Generic::matCopy(obj,other);
      }

      // Specialization for int32_t
      template <>
      Basics::Matrix<int32_t> *matCopy(Basics::Matrix<int32_t> *obj,
                                       const Basics::Matrix<int32_t> *other) {
        return AprilMath::MatrixExt::Operations::Generic::matCopy(obj,other);
      }
      
      Basics::Matrix<float> *matInv(const Basics::Matrix<float> *obj) {
        if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
          ERROR_EXIT(128, "Only bi-dimensional matrices are allowed\n");
        }
        Basics::Matrix<float> *A = obj->clone(CblasColMajor);
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

      void matSVD(const Basics::Matrix<float> *obj,
                  Basics::Matrix<float> **U, Basics::SparseMatrix<float> **S,
                  Basics::Matrix<float> **VT) {
        if (obj->getNumDim() != 2) {
          ERROR_EXIT(128, "Only bi-dimensional matrices are allowed\n");
        }
        AprilUtils::SharedPtr< Basics::Matrix<float> > A( obj->clone(CblasColMajor) );
        int INFO;
        const int m = A->getDimSize(0); // cols
        const int n = A->getDimSize(1); // rows
        const int lda = A->getStrideSize(1);
        const int numSV = (m<n) ? m : n;
        const int dimsU[2]  = {m, m};
        const int dimsVT[2] = {n, n};
        *U  = new Basics::Matrix<float>(2, dimsU,  CblasColMajor);
        *S  = Basics::SparseMatrix<float>::diag(numSV, 0.0f, CSR_FORMAT);
        *VT = new Basics::Matrix<float>(2, dimsVT, CblasColMajor);
        INFO = clapack_sgesdd(CblasColMajor, m, n, lda,
                              A->getRawDataAccess()->getPPALForReadAndWrite(),
                              (*U)->getRawDataAccess()->getPPALForWrite(),
                              (*S)->getRawValuesAccess()->getPPALForWrite(),
                              (*VT)->getRawDataAccess()->getPPALForWrite());
        checkLapackInfo(INFO);
      }

      // FROM: http://www.r-bloggers.com/matrix-determinant-with-the-lapack-routine-dspsv/    
      AprilUtils::log_float matLogDeterminant(const Basics::Matrix<float> *obj,
                                               float &sign) {
        if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
          ERROR_EXIT(128, "Only squared bi-dimensional matrices are allowed\n");
        }
        AprilUtils::SharedPtr< Basics::Matrix<float> > A( obj->clone(CblasColMajor) );
        AprilUtils::UniquePtr<int []> IPIV( new int[A->getDimSize(0)] );
        int INFO;
        INFO = clapack_sgetrf(CblasColMajor,
                              A->getDimSize(0), A->getDimSize(1),
                              A->getRawDataAccess()->getPPALForReadAndWrite(),
                              A->getStrideSize(1),
                              IPIV.get());
        checkLapackInfo(INFO);
        Basics::Matrix<float>::const_random_access_iterator it(A.get());
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
      double matDeterminant(const Basics::Matrix<float> *obj) {
        if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
          ERROR_EXIT(128, "Only squared bi-dimensional matrices are allowed\n");
        }
        AprilUtils::SharedPtr< Basics::Matrix<float> > A( obj->clone(CblasColMajor) );
        AprilUtils::UniquePtr<int []> IPIV( new int[A->getDimSize(0)] );
        int INFO;
        INFO = clapack_sgetrf(CblasColMajor,
                              A->getDimSize(0), A->getDimSize(1),
                              A->getRawDataAccess()->getPPALForReadAndWrite(),
                              A->getStrideSize(1),
                              IPIV.get());
        checkLapackInfo(INFO);
        Basics::Matrix<float>::const_random_access_iterator it(A.get());
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
      Basics::Matrix<float> *matCholesky(const Basics::Matrix<float> *obj,
                                         char uplo) {
        if (obj->getNumDim() != 2 || obj->getDimSize(0) != obj->getDimSize(1)) {
          ERROR_EXIT(128, "Only squared bi-dimensional matrices are allowed\n");
        }
        Basics::Matrix<float> *A = obj->clone(CblasColMajor);
        int INFO = clapack_spotrf(CblasColMajor,
                                  (uplo == 'U') ? CblasUpper : CblasLower,
                                  A->getDimSize(0),
                                  A->getRawDataAccess()->getPPALForReadAndWrite(),
                                  A->getStrideSize(1));
        checkLapackInfo(INFO);
        switch(uplo) {
        case 'U':
          {
            Basics::Matrix<float>::random_access_iterator it(A);
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
            Basics::Matrix<float>::random_access_iterator it(A);
            for (int i=0; i<A->getDimSize(0); ++i) {
              for (int j=i+1; j<A->getDimSize(0); ++j) {
                it(i,j) = 0.0f;
              }
            }
          }
        }
        return A;
      }

    } // namespace Operations
  } // namespace MatrixExt
} // namespace AprilMath
