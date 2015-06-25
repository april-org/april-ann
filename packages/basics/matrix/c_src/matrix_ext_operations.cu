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
    
      template <typename T>
      Matrix<T> *matScal(Matrix<T> *obj, const T value,
                         Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixSpanMap1<T,T>(obj, CurriedScal<T>(value), dest);
      }

      template <typename T>
      SparseMatrix<T> *matScal(SparseMatrix<T> *obj,
                               const T value,
                               SparseMatrix<T> *dest) {
        if (dest == 0) dest = obj;
        return SparseMatrixScalarMap1<T,T>(obj, m_curried_mul<T>(value), dest);
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
      Matrix<T> *matDiv(Matrix<T> *obj, const T &value,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_div<T>(value), dest);
      }

      template <typename T>
      Matrix<T> *matIDiv(Matrix<T> *obj, const T &value,
                         Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, m_curried_idiv<T>(value), dest);
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

      template <typename T>
      Matrix<T> *matMod(Matrix<T> *obj, const T &value,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<T,T>(obj, AprilMath::m_curried_mod<T>(value),
                                     dest);
      }

      template <typename T>
      Matrix<T> *matMod(Matrix<T> *obj, const Basics::Matrix<T> *other,
                        Matrix<T> *dest) {
        if (dest == 0) dest = obj;
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
      Matrix<T> *matMaskedFill(Matrix<T> *obj, const Matrix<bool> *mask,
                               const T &value, Matrix<T> *dest) {
        if (dest == 0) dest = obj;
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
      Matrix<T> *matMaskedCopy(Matrix<T> *obj1, const Matrix<bool> *mask,
                               const Matrix<T> *obj2, Matrix<T> *dest) {
        if (dest == 0) dest = obj1;
        return MatrixScalarMap3<T,bool,T,T>(obj1, mask, obj2,
                                            maskedCopyOp<T>(), dest);
      }
      
      //////////////// integer versions ///////////////////
      template <>
      Matrix<int32_t> *matScal(Matrix<int32_t> *obj, const int32_t value,
                         Matrix<int32_t> *dest) {
        if (dest == 0) dest = obj;
        return MatrixScalarMap1<int32_t,int32_t>(obj,
                                                 AprilMath::
                                                 m_curried_mul<int32_t>(value),
                                                 dest);
      }
      ////////////////////////////////////////////////////
      
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
      template Matrix<float> *matScal(Matrix<float> *, const float,
                                      Matrix<float> *);
      template Matrix<float> *matCmul(Matrix<float> *,
                                      const Matrix<float> *,
                                      Matrix<float> *);
      template Matrix<float> *matScalarAdd(Matrix<float> *, const float &,
                                           Matrix<float> *);
      template Matrix<float> *matAdjustRange(Matrix<float> *,
                                             const float &, const float &,
                                             Matrix<float> *);        
      template Matrix<float> *matDiv(Matrix<float> *,
                                     const float &,
                                     Matrix<float> *);
      template Matrix<float> *matDiv(Matrix<float> *,
                                     const Matrix<float> *,
                                     Matrix<float> *);
      template Matrix<float> *matMaskedFill(Matrix<float> *,
                                            const Matrix<bool> *,
                                            const float &,
                                            Matrix<float> *);
      template Matrix<float> *matMaskedCopy(Matrix<float> *,
                                            const Matrix<bool> *,
                                            const Matrix<float> *,
                                            Matrix<float> *);

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
      template Matrix<double> *matScal(Matrix<double> *, const double,
                                       Matrix<double> *);
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
      template Matrix<double> *matDiv(Matrix<double> *,
                                      const Matrix<double> *,
                                      Matrix<double> *);
      template Matrix<double> *matMaskedFill(Matrix<double> *,
                                             const Matrix<bool> *,
                                             const double &,
                                             Matrix<double> *);
      template Matrix<double> *matMaskedCopy(Matrix<double> *,
                                             const Matrix<bool> *,
                                             const Matrix<double> *,
                                             Matrix<double> *);



      template Matrix<int32_t> *matScal(Matrix<int32_t> *, const int32_t,
                                        Matrix<int32_t> *);
      template Matrix<int32_t> *matCmul(Matrix<int32_t> *,
                                        const Matrix<int32_t> *,
                                        Matrix<int32_t> *);
      template Matrix<int32_t> *matClamp(Matrix<int32_t> *, const int32_t,
                                         const int32_t, Matrix<int32_t> *);
      template Matrix<int32_t> *matScalarAdd(Matrix<int32_t> *, const int32_t &,
                                             Matrix<int32_t> *);
      template Matrix<int32_t> *matIDiv(Matrix<int32_t> *, const int32_t &,
                                        Matrix<int32_t> *);
      template Matrix<int32_t> *matDiv(Matrix<int32_t> *,
                                       const Matrix<int32_t> *,
                                       Matrix<int32_t> *);
      template Matrix<int32_t> *matMod(Matrix<int32_t> *,
                                       const Matrix<int32_t> *,
                                       Matrix<int32_t> *);
      template Matrix<int32_t> *matMod(Matrix<int32_t> *,
                                       const int32_t &,
                                       Matrix<int32_t> *);
      template Matrix<int32_t> *matMaskedFill(Matrix<int32_t> *,
                                              const Matrix<bool> *,
                                              const int32_t &,
                                              Matrix<int32_t> *);
      template Matrix<int32_t> *matMaskedCopy(Matrix<int32_t> *,
                                              const Matrix<bool> *,
                                              const Matrix<int32_t> *,
                                              Matrix<int32_t> *);
      
      
      template Matrix<ComplexF> *matScal(Matrix<ComplexF> *, const ComplexF,
                                         Matrix<ComplexF> *);
      template Matrix<ComplexF> *matCmul(Matrix<ComplexF> *,
                                         const Matrix<ComplexF> *,
                                         Matrix<ComplexF> *);
      template Matrix<ComplexF> *matScalarAdd(Matrix<ComplexF> *, const ComplexF &,
                                              Matrix<ComplexF> *);
      template Matrix<ComplexF> *matDiv(Matrix<ComplexF> *, const ComplexF &,
                                        Matrix<ComplexF> *);
      template Matrix<ComplexF> *matDiv(Matrix<ComplexF> *,
                                        const Matrix<ComplexF> *,
                                        Matrix<ComplexF> *);
      template Matrix<ComplexF> *matMaskedFill(Matrix<ComplexF> *,
                                               const Matrix<bool> *,
                                               const ComplexF &,
                                               Matrix<ComplexF> *);
      template Matrix<ComplexF> *matMaskedCopy(Matrix<ComplexF> *,
                                               const Matrix<bool> *,
                                               const Matrix<ComplexF> *,
                                               Matrix<ComplexF> *);

      
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
      template SparseMatrix<float> *matScal(SparseMatrix<float> *, const float,
                                            SparseMatrix<float> *);
      template SparseMatrix<float> *matDiv(SparseMatrix<float> *, const float &,
                                           SparseMatrix<float> *);


      
      template SparseMatrix<ComplexF> *matScal(SparseMatrix<ComplexF> *,
                                               const ComplexF,
                                               SparseMatrix<ComplexF> *);

      

      template SparseMatrix<double> *matSqrt(SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matPow(SparseMatrix<double> *,
                                            const double &,
                                            SparseMatrix<double> *);
      template SparseMatrix<double> *matTan(SparseMatrix<double> *,
                                            SparseMatrix<double> *);
      template SparseMatrix<double> *matAtan(SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matTanh(SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matAtanh(SparseMatrix<double> *,
                                              SparseMatrix<double> *);
      template SparseMatrix<double> *matSin(SparseMatrix<double> *,
                                            SparseMatrix<double> *);
      template SparseMatrix<double> *matAsin(SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matSinh(SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matAsinh(SparseMatrix<double> *,
                                              SparseMatrix<double> *);
      template SparseMatrix<double> *matAbs(SparseMatrix<double> *,
                                            SparseMatrix<double> *);
      template SparseMatrix<double> *matSign(SparseMatrix<double> *,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matScal(SparseMatrix<double> *, const double,
                                             SparseMatrix<double> *);
      template SparseMatrix<double> *matDiv(SparseMatrix<double> *, const double &,
                                            SparseMatrix<double> *);
      
    } // namespace Operations
    
  } // namespace MatrixExt
} // namespace AprilMath
