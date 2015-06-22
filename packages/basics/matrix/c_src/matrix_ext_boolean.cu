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
#include "matrix_ext_boolean.h"

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
    
    namespace Boolean {
      
      //////////////////// BOOLEAN CONDITIONS ////////////////////

      /* BOOLEAN CONDITIONS: this methods transforms the given matrix in a
         ZERO/ONE matrix, depending in the truth of the given condition */
    
      template <typename T>
      Matrix<bool> *matLT(const Matrix<T> *obj, const T &value,
                          Matrix<bool> *dest) {
        if (dest == 0) {
          dest = new Matrix<bool>(obj->getNumDim(),
                                  obj->getDimPtr());
        }
        return MatrixScalarMap1<T,bool>(obj, m_curried_lt<T>(value), dest);
      }

      template <typename T>
      Matrix<bool> *matLT(const Matrix<T> *obj,
                          const Matrix<T> *other,
                          Matrix<bool> *dest) {
        if (dest == 0) {
          dest = new Matrix<bool>(obj->getNumDim(),
                                  obj->getDimPtr());
        }
        return MatrixScalarMap2<T,T,bool>(obj, other,
                                          AprilMath::Functors::m_lt<T>(), dest);
      }

      template <typename T>
      Matrix<bool> *matGT(const Matrix<T> *obj, const T &value,
                          Matrix<bool> *dest) {
        if (dest == 0) {
          dest = new Matrix<bool>(obj->getNumDim(),
                                  obj->getDimPtr());
        }
        return MatrixScalarMap1<T,bool>(obj, m_curried_gt<T>(value), dest);
      }

      template <typename T>
      Matrix<bool> *matGT(const Matrix<T> *obj,
                          const Matrix<T> *other,
                          Matrix<bool> *dest) {
        if (dest == 0) {
          dest = new Matrix<bool>(obj->getNumDim(),
                                  obj->getDimPtr());
        }
        return MatrixScalarMap2<T,T,bool>(obj, other,
                                          AprilMath::Functors::m_gt<T>(), dest);
      }

      template <typename T>
      Matrix<bool> *matEQ(const Matrix<T> *obj, const T &value,
                          Matrix<bool> *dest) {
        if (dest == 0) {
          dest = new Matrix<bool>(obj->getNumDim(),
                                  obj->getDimPtr());
        }
        if (m_isnan(value)) {
          return MatrixScalarMap1<T,bool>(obj, m_curried_eq_nan<T>(), dest);
        }
        else {
          return MatrixScalarMap1<T,bool>(obj, m_curried_eq<T>(value), dest);
        }
      }
    
      template <typename T>
      Matrix<bool> *matEQ(const Matrix<T> *obj,
                          const Matrix<T> *other,
                          Matrix<bool> *dest) {
        if (dest == 0) {
          dest = new Matrix<bool>(obj->getNumDim(),
                                  obj->getDimPtr());
        }
        return MatrixScalarMap2<T,T,bool>(obj, other,
                                          AprilMath::Functors::m_eq<T>(), dest);
      }
    
      template <typename T>
      Matrix<bool> *matNEQ(const Matrix<T> *obj, const T &value,
                           Matrix<bool> *dest) {
        if (dest == 0) {
          dest = new Matrix<bool>(obj->getNumDim(),
                                  obj->getDimPtr());
        }
        if (m_isnan(value)) {
          return MatrixScalarMap1<T,bool>(obj, m_curried_neq_nan<T>(), dest);
        }
        else {
          return MatrixScalarMap1<T,bool>(obj, m_curried_neq<T>(value), dest);
        }
      }
    
      template <typename T>
      Matrix<bool> *matNEQ(const Matrix<T> *obj,
                           const Matrix<T> *other,
                           Matrix<bool> *dest) {
        if (dest == 0) {
          dest = new Matrix<bool>(obj->getNumDim(),
                                  obj->getDimPtr());
        }
        return MatrixScalarMap2<T,T,bool>(obj, other,
                                          AprilMath::Functors::m_neq<T>(), dest);
      }

      template Matrix<bool> *matLT(const Matrix<float> *, const float &,
                                   Matrix<bool> *);

      template Matrix<bool> *matLT(const Matrix<float> *,
                                   const Matrix<float> *,
                                   Matrix<bool> *);
      template Matrix<bool> *matGT(const Matrix<float> *, const float &, Matrix<bool> *);
      template Matrix<bool> *matGT(const Matrix<float> *,
                                   const Matrix<float> *, Matrix<bool> *);
      template Matrix<bool> *matEQ(const Matrix<float> *, const float &, Matrix<bool> *);
      template Matrix<bool> *matEQ(const Matrix<float> *,
                                   const Matrix<float> *,
                                   Matrix<bool> *);
      template Matrix<bool> *matNEQ(const Matrix<float> *, const float &,
                                    Matrix<bool> *);
      template Matrix<bool> *matNEQ(const Matrix<float> *,
                                    const Matrix<float> *,
                                    Matrix<bool> *);
      
      template Matrix<bool> *matLT(const Matrix<double> *, const double &,
                                   Matrix<bool> *);
      template Matrix<bool> *matLT(const Matrix<double> *,
                                   const Matrix<double> *,
                                   Matrix<bool> *);
      template Matrix<bool> *matGT(const Matrix<double> *, const double &, Matrix<bool> *);
      template Matrix<bool> *matGT(const Matrix<double> *,
                                   const Matrix<double> *, Matrix<bool> *);
      template Matrix<bool> *matEQ(const Matrix<double> *, const double &, Matrix<bool> *);
      template Matrix<bool> *matEQ(const Matrix<double> *,
                                   const Matrix<double> *,
                                   Matrix<bool> *);
      template Matrix<bool> *matNEQ(const Matrix<double> *, const double &,
                                    Matrix<bool> *);
      template Matrix<bool> *matNEQ(const Matrix<double> *,
                                    const Matrix<double> *,
                                    Matrix<bool> *);

      
      template Matrix<bool> *matLT(const Matrix<int32_t> *, const int32_t &,
                                   Matrix<bool> *);
      template Matrix<bool> *matLT(const Matrix<int32_t> *,
                                   const Matrix<int32_t> *,
                                   Matrix<bool> *);
      template Matrix<bool> *matGT(const Matrix<int32_t> *, const int32_t &, Matrix<bool> *);
      template Matrix<bool> *matGT(const Matrix<int32_t> *,
                                   const Matrix<int32_t> *, Matrix<bool> *);
      template Matrix<bool> *matEQ(const Matrix<int32_t> *, const int32_t &, Matrix<bool> *);
      template Matrix<bool> *matEQ(const Matrix<int32_t> *,
                                   const Matrix<int32_t> *,
                                   Matrix<bool> *);
      template Matrix<bool> *matNEQ(const Matrix<int32_t> *, const int32_t &,
                                    Matrix<bool> *);
      template Matrix<bool> *matNEQ(const Matrix<int32_t> *,
                                    const Matrix<int32_t> *,
                                    Matrix<bool> *);

    } // namespace Boolean
        
  } // namespace MatrixExt
} // namespace AprilMath
