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
#include "matrix_ext_operations.h"

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
    
    namespace Operations {
      
      template <typename T>
      Matrix<T> *matPlogp(const Matrix<T> *obj,
                          Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T> (obj, AprilMath::Functors::m_plogp<T>(), dest);
      }
      
      template <typename T>
      Matrix<T> *matLog(const Matrix<T> *obj,
                        Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_log<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matLog1p(const Matrix<T> *obj,
                          Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_log1p<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matExp(const Matrix<T> *obj,
                        Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_exp<T>(), dest);
      }

      template <typename T>
      Matrix<T> *matExpm1(const Matrix<T> *obj,
                          Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_expm1<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matSqrt(const Matrix<T> *obj,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sqrt<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matSqrt(const SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sqrt<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matPow(const Matrix<T> *obj, const T &value,
                        Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, m_curried_pow<T>(value), dest);
      }

      template <typename T>
      SparseMatrix<T> *matPow(const SparseMatrix<T> *obj, const T &value,
                              SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, m_curried_pow<T>(value), dest);
      }
    
      template <typename T>
      Matrix<T> *matTan(const Matrix<T> *obj,
                        Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_tan<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matTan(const SparseMatrix<T> *obj,
                              SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_tan<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matTanh(const Matrix<T> *obj,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_tanh<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matTanh(const SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_tanh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAtan(const Matrix<T> *obj,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_atan<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matAtan(const SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_atan<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAtanh(const Matrix<T> *obj,
                          Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_atanh<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matAtanh(const SparseMatrix<T> *obj,
                                SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_atanh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matSin(const Matrix<T> *obj,
                        Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sin<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matSin(const SparseMatrix<T> *obj,
                              SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sin<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matSinh(const Matrix<T> *obj,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sinh<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matSinh(const SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sinh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAsin(const Matrix<T> *obj,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_asin<T>(), dest);
      }
    
      template <typename T>
      SparseMatrix<T> *matAsin(const SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_asin<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAsinh(const Matrix<T> *obj,
                          Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_asinh<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matAsinh(const SparseMatrix<T> *obj,
                                SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_asinh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matCos(const Matrix<T> *obj,
                        Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_cos<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matCosh(const Matrix<T> *obj,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_cosh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAcos(const Matrix<T> *obj,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_acos<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAcosh(const Matrix<T> *obj,
                          Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_acosh<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matAbs(const Matrix<T> *obj,
                        Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_abs<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matAbs(const SparseMatrix<T> *obj,
                              SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_abs<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matComplement(const Matrix<T> *obj,
                               Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_complement<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matSign(const Matrix<T> *obj,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sign<T>(), dest);
      }
    
      template <typename T>
      SparseMatrix<T> *matSign(const SparseMatrix<T> *obj,
                               SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_sign<T>(), dest);
      }
    
      ////////////////// OTHER MAP FUNCTIONS //////////////////
    
      /// Performs a clamp operation, in-place operation if @c dest=0, otherwise,
      /// the result will be computed at the given @c dest Matrix.
      template<typename T>
      Matrix<T> *matClamp(const Matrix<T> *obj,
                          const T lower, const T upper,
                          Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, m_curried_clamp<T>(lower,upper), dest);
      }

      /// Performs a floor operation, in-place operation if @c dest=0, otherwise,
      /// the result will be computed at the given @c dest Matrix.
      template<typename T>
      Matrix<T> *matFloor(const Matrix<T> *obj,
                          Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_floor<T>(), dest);
      }

      /// Performs a ceil operation, in-place operation if @c dest=0, otherwise,
      /// the result will be computed at the given @c dest Matrix.
      template<typename T>
      Matrix<T> *matCeil(const Matrix<T> *obj,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_ceil<T>(), dest);
      }

      /// Performs a round operation, in-place operation if @c dest=0, otherwise,
      /// the result will be computed at the given @c dest Matrix.
      template<typename T>
      Matrix<T> *matRound(const Matrix<T> *obj,
                          Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::Functors::m_round<T>(), dest);
      }
    
      template <typename T>
      Matrix<T> *matScal(const Matrix<T> *obj, const T value,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixSpanMap1<T,T>(obj, CurriedScal<T>(value), dest);
      }

      template <typename T>
      SparseMatrix<T> *matScal(const SparseMatrix<T> *obj,
                               const T value,
                               SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, m_curried_mul<T>(value), dest);
      }

      template <typename T>
      Matrix<T> *matCmul(const Matrix<T> *obj,
                         const Matrix<T> *other,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap2<T,T,T>(obj, other,
                                       AprilMath::Functors::m_mul<T>(),
                                       dest);
      }
    
      template <typename T>
      Matrix<T> *matScalarAdd(const Matrix<T> *obj, const T &v,
                              Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, m_curried_add<T>(v), dest);
      }
    
      template <typename T>
      Matrix<T> *matAdjustRange(const Matrix<T> *obj,
                                const T &rmin, const T &rmax,
                                Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        T mmin, mmax;
        AprilMath::MatrixExt::Reductions::matMinAndMax(obj, mmin, mmax);
        // especial case, set all values to rmin
        if (mmax - mmin == T()) {
          AprilMath::MatrixExt::Initializers::matFill(dest, rmin);
        }
        else {
          const T ratio = (rmax-rmin)/(mmax-mmin);
          if (mmin > AprilMath::Limits<T>::zero() ||
              mmin < AprilMath::Limits<T>::zero()) {
            matScalarAdd(obj, -mmin, dest);
          }
          AprilMath::MatrixExt::Operations::matScal(dest, ratio);
          if (rmin > AprilMath::Limits<T>::zero() ||
              rmin < AprilMath::Limits<T>::zero()) {
            matScalarAdd(dest, rmin);
          }
        }
        return dest;
      }
        
      template <typename T>
      Matrix<T> *matDiv(const Matrix<T> *obj, const T &value,
                        Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, m_curried_div<T>(value), dest);
      }

      template <typename T>
      Matrix<T> *matIDiv(const Matrix<T> *obj, const T &value,
                         Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, m_curried_idiv<T>(value), dest);
      }

      template <typename T>
      Basics::Matrix<T> *matDiv(Basics::Matrix<T> *obj,
                                const Basics::Matrix<T> *other,
                                Basics::Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap2<T,T,T>(obj, other,
                                       AprilMath::Functors::m_div<T>(), dest);
      }

      template <typename T>
      SparseMatrix<T> *matDiv(const SparseMatrix<T> *obj, const T &value,
                              SparseMatrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return SparseMatrixScalarMap1<T,T>(obj, m_curried_div<T>(value), dest);
      }

      template <typename T>
      Matrix<T> *matMod(const Matrix<T> *obj, const T &value,
                        Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap1<T,T>(obj, AprilMath::m_curried_mod<T>(value),
                                     dest);
      }

      template <typename T>
      Matrix<T> *matMod(const Matrix<T> *obj, const Basics::Matrix<T> *other,
                        Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap2<T,T>(obj, other,
                                     AprilMath::Functors::m_mod<T>(), dest);
      }

      /// Curried masked fill operator.
      template<typename T> struct maskedFillOp {
        const T value; ///< fill value.
        /// The constructor stores inf and sup values.
        maskedFillOp(const T &value) : value(value) { }
        /**
         * @brief Returns <tt> b ? value : a </tt>
         */
        APRIL_CUDA_EXPORT T operator()(const T &a, const bool &b) const {
          return (b) ? (value) : (a);
        }
      };
      
      template <typename T>
      Matrix<T> *matMaskedFill(const Matrix<T> *obj, const Matrix<bool> *mask,
                               const T &value, Matrix<T> *dest) {
        april_assert( dest != 0 );
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap2<T,bool,T>(obj, mask,
                                          maskedFillOp<T>(value),
                                          dest);
      }

      /// Curried masked copy operator.
      template<typename T> struct maskedCopyOp {
        maskedCopyOp() { }
        /**
         * @brief Returns <tt> b ? c : a </tt>
         */
        APRIL_CUDA_EXPORT T operator()(const T &a, const bool &b,
                                       const T &c) const {
          return (b) ? (c) : (a);
        }
      };
      
      template <typename T>
      Matrix<T> *matMaskedCopy(const Matrix<T> *obj1, const Matrix<bool> *mask,
                               const Matrix<T> *obj2, Matrix<T> *dest) {
        if (dest == 0) dest = obj1;
        // TODO: sanity check, to be removed after checking all the code has
        // been updated with commit performed 2015/10/26 around 13:00
        if (dest == 0) ERROR_EXIT(128, "Given a NULL destination matrix");
        return MatrixScalarMap3<T,bool,T,T>(obj1, mask, obj2,
                                            maskedCopyOp<T>(), dest);
      }
      
      //////////////// integer versions ///////////////////
      template <>
      Matrix<int32_t> *matScal(const Matrix<int32_t> *obj, const int32_t value,
                               Matrix<int32_t> *dest) {
        april_assert( dest != 0 );
        return MatrixScalarMap1<int32_t,int32_t>(obj,
                                                 AprilMath::
                                                 m_curried_mul<int32_t>(value),
                                                 dest);
      }
      ////////////////////////////////////////////////////
      
      template Matrix<float> *matPlogp(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matLog(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matLog1p(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matExp(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matExpm1(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matSqrt(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matPow(const Matrix<float> *, const float &, Matrix<float> *);
      template Matrix<float> *matTan(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matTanh(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAtan(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAtanh(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matSin(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matSinh(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAsin(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAsinh(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matCos(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matCosh(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAcos(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAcosh(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matAbs(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matComplement(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matSign(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matClamp(const Matrix<float> *, const float,
                                       const float, Matrix<float> *);
      template Matrix<float> *matFloor(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matCeil(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matRound(const Matrix<float> *, Matrix<float> *);
      template Matrix<float> *matScal(const Matrix<float> *, const float,
                                      Matrix<float> *);
      template Matrix<float> *matCmul(const Matrix<float> *,
                                      const Matrix<float> *,
                                      Matrix<float> *);
      template Matrix<float> *matScalarAdd(const Matrix<float> *, const float &,
                                           Matrix<float> *);
      template Matrix<float> *matAdjustRange(const Matrix<float> *,
                                             const float &, const float &,
                                             Matrix<float> *);        
      template Matrix<float> *matDiv(const Matrix<float> *,
                                     const float &,
                                     Matrix<float> *);
      template Matrix<float> *matDiv(const Matrix<float> *,
                                     const Matrix<float> *,
                                     Matrix<float> *);
      template Matrix<float> *matMaskedFill(const Matrix<float> *,
                                            const Matrix<bool> *,
                                            const float &,
                                            Matrix<float> *);
      template Matrix<float> *matMaskedCopy(const Matrix<float> *,
                                            const Matrix<bool> *,
                                            const Matrix<float> *,
                                            Matrix<float> *);

      template Matrix<double> *matPlogp(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matLog(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matLog1p(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matExp(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matExpm1(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matSqrt(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matPow(const Matrix<double> *, const double &, Matrix<double> *);
      template Matrix<double> *matTan(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matTanh(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAtan(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAtanh(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matSin(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matSinh(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAsin(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAsinh(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matCos(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matCosh(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAcos(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAcosh(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matAbs(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matComplement(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matSign(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matClamp(const Matrix<double> *, const double,
                                        const double, Matrix<double> *);
      template Matrix<double> *matFloor(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matCeil(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matRound(const Matrix<double> *, Matrix<double> *);
      template Matrix<double> *matScal(const Matrix<double> *, const double,
                                       Matrix<double> *);
      template Matrix<double> *matCmul(const Matrix<double> *,
                                       const Matrix<double> *,
                                       Matrix<double> *);
      template Matrix<double> *matScalarAdd(const Matrix<double> *, const double &,
                                            Matrix<double> *);
      template Matrix<double> *matAdjustRange(const Matrix<double> *,
                                              const double &, const double &,
                                              Matrix<double> *);        
      template Matrix<double> *matDiv(const Matrix<double> *, const double &,
                                      Matrix<double> *);
      template Matrix<double> *matDiv(const Matrix<double> *,
                                      const Matrix<double> *,
                                      Matrix<double> *);
      template Matrix<double> *matMaskedFill(const Matrix<double> *,
                                             const Matrix<bool> *,
                                             const double &,
                                             Matrix<double> *);
      template Matrix<double> *matMaskedCopy(const Matrix<double> *,
                                             const Matrix<bool> *,
                                             const Matrix<double> *,
                                             Matrix<double> *);



      template Matrix<int32_t> *matScal(const Matrix<int32_t> *, const int32_t,
                                        Matrix<int32_t> *);
      template Matrix<int32_t> *matCmul(const Matrix<int32_t> *,
                                        const Matrix<int32_t> *,
                                        Matrix<int32_t> *);
      template Matrix<int32_t> *matClamp(const Matrix<int32_t> *, const int32_t,
                                         const int32_t, Matrix<int32_t> *);
      template Matrix<int32_t> *matScalarAdd(const Matrix<int32_t> *, const int32_t &,
                                             Matrix<int32_t> *);
      template Matrix<int32_t> *matIDiv(const Matrix<int32_t> *, const int32_t &,
                                        Matrix<int32_t> *);
      template Matrix<int32_t> *matDiv(const Matrix<int32_t> *,
                                       const Matrix<int32_t> *,
                                       Matrix<int32_t> *);
      template Matrix<int32_t> *matMod(const Matrix<int32_t> *,
                                       const Matrix<int32_t> *,
                                       Matrix<int32_t> *);
      template Matrix<int32_t> *matMod(const Matrix<int32_t> *,
                                       const int32_t &,
                                       Matrix<int32_t> *);
      template Matrix<int32_t> *matMaskedFill(const Matrix<int32_t> *,
                                              const Matrix<bool> *,
                                              const int32_t &,
                                              Matrix<int32_t> *);
      template Matrix<int32_t> *matMaskedCopy(const Matrix<int32_t> *,
                                              const Matrix<bool> *,
                                              const Matrix<int32_t> *,
                                              Matrix<int32_t> *);

      
      template Matrix<char> *matMaskedFill(const Matrix<char> *,
                                           const Matrix<bool> *,
                                           const char &,
                                           Matrix<char> *);
      template Matrix<char> *matMaskedCopy(const Matrix<char> *,
                                           const Matrix<bool> *,
                                           const Matrix<char> *,
                                           Matrix<char> *);


      
      template Matrix<bool> *matMaskedFill(const Matrix<bool> *,
                                           const Matrix<bool> *,
                                           const bool &,
                                           Matrix<bool> *);
      template Matrix<bool> *matMaskedCopy(const Matrix<bool> *,
                                           const Matrix<bool> *,
                                           const Matrix<bool> *,
                                           Matrix<bool> *);
      

      template Matrix<ComplexF> *matFloor(const Matrix<ComplexF> *, Matrix<ComplexF> *);
      template Matrix<ComplexF> *matCeil(const Matrix<ComplexF> *, Matrix<ComplexF> *);
      template Matrix<ComplexF> *matRound(const Matrix<ComplexF> *, Matrix<ComplexF> *);      
      template Matrix<ComplexF> *matScal(const Matrix<ComplexF> *, const ComplexF,
                                         Matrix<ComplexF> *);
      template Matrix<ComplexF> *matCmul(const Matrix<ComplexF> *,
                                         const Matrix<ComplexF> *,
                                         Matrix<ComplexF> *);
      template Matrix<ComplexF> *matScalarAdd(const Matrix<ComplexF> *, const ComplexF &,
                                              Matrix<ComplexF> *);
      template Matrix<ComplexF> *matDiv(const Matrix<ComplexF> *, const ComplexF &,
                                        Matrix<ComplexF> *);
      template Matrix<ComplexF> *matDiv(const Matrix<ComplexF> *,
                                        const Matrix<ComplexF> *,
                                        Matrix<ComplexF> *);
      template Matrix<ComplexF> *matMaskedFill(const Matrix<ComplexF> *,
                                               const Matrix<bool> *,
                                               const ComplexF &,
                                               Matrix<ComplexF> *);
      template Matrix<ComplexF> *matMaskedCopy(const Matrix<ComplexF> *,
                                               const Matrix<bool> *,
                                               const Matrix<ComplexF> *,
                                               Matrix<ComplexF> *);

      
      template SparseMatrix<float> *matSqrt(const SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matPow(const SparseMatrix<float> *,
                                           const float &,
                                           SparseMatrix<float> *);
      template SparseMatrix<float> *matTan(const SparseMatrix<float> *,
                                           SparseMatrix<float> *);
      template SparseMatrix<float> *matAtan(const SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matTanh(const SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matAtanh(const SparseMatrix<float> *,
                                             SparseMatrix<float> *);
      template SparseMatrix<float> *matSin(const SparseMatrix<float> *,
                                           SparseMatrix<float> *);
      template SparseMatrix<float> *matAsin(const SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matSinh(const SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matAsinh(const SparseMatrix<float> *,
                                             SparseMatrix<float> *);
      template SparseMatrix<float> *matAbs(const SparseMatrix<float> *,
                                           SparseMatrix<float> *);
      template SparseMatrix<float> *matSign(const SparseMatrix<float> *,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matScal(const SparseMatrix<float> *, const float,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matDiv(const SparseMatrix<float> *, const float &,
                                           SparseMatrix<float> *);


      
      template SparseMatrix<ComplexF> *matScal(const SparseMatrix<ComplexF> *,
                                               const ComplexF,
                                               SparseMatrix<ComplexF> *);

      

      template SparseMatrix<double> *matSqrt(const SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matPow(const SparseMatrix<double> *,
                                            const double &,
                                            SparseMatrix<double> *);
      template SparseMatrix<double> *matTan(const SparseMatrix<double> *,
                                            SparseMatrix<double> *);
      template SparseMatrix<double> *matAtan(const SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matTanh(const SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matAtanh(const SparseMatrix<double> *,
                                              SparseMatrix<double> *);
      template SparseMatrix<double> *matSin(const SparseMatrix<double> *,
                                            SparseMatrix<double> *);
      template SparseMatrix<double> *matAsin(const SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matSinh(const SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matAsinh(const SparseMatrix<double> *,
                                              SparseMatrix<double> *);
      template SparseMatrix<double> *matAbs(const SparseMatrix<double> *,
                                            SparseMatrix<double> *);
      template SparseMatrix<double> *matSign(const SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matScal(const SparseMatrix<double> *, const double,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matDiv(const SparseMatrix<double> *, const double &,
                                            SparseMatrix<double> *);
      
    } // namespace Operations
    
  } // namespace MatrixExt
} // namespace AprilMath
