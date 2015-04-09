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
#include "matrix_ext_initializers.h"

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
    
    namespace Initializers {

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

      template Matrix<float> *matFill(Matrix<float> *, const float);
      template Matrix<float> *matZeros(Matrix<float> *);
      template Matrix<float> *matOnes(Matrix<float> *);
      template Matrix<float> *matDiag(Matrix<float> *, const float);
      
      template Matrix<double> *matFill(Matrix<double> *, const double);
      template Matrix<double> *matZeros(Matrix<double> *);
      template Matrix<double> *matOnes(Matrix<double> *);
      template Matrix<double> *matDiag(Matrix<double> *, const double);

      template Matrix<ComplexF> *matFill(Matrix<ComplexF> *, const ComplexF);
      template Matrix<ComplexF> *matZeros(Matrix<ComplexF> *);
      template Matrix<ComplexF> *matOnes(Matrix<ComplexF> *);
      template Matrix<ComplexF> *matDiag(Matrix<ComplexF> *, const ComplexF);

      template Matrix<char> *matFill(Matrix<char> *, const char);
      template Matrix<char> *matZeros(Matrix<char> *);
      template Matrix<char> *matOnes(Matrix<char> *);
      template Matrix<char> *matDiag(Matrix<char> *, const char);

      template Matrix<bool> *matFill(Matrix<bool> *, const bool);
      template Matrix<bool> *matZeros(Matrix<bool> *);
      template Matrix<bool> *matOnes(Matrix<bool> *);
      template Matrix<bool> *matDiag(Matrix<bool> *, const bool);

      template Matrix<int32_t> *matFill(Matrix<int32_t> *, const int32_t);
      template Matrix<int32_t> *matZeros(Matrix<int32_t> *);
      template Matrix<int32_t> *matOnes(Matrix<int32_t> *);
      template Matrix<int32_t> *matDiag(Matrix<int32_t> *, const int32_t);

      template SparseMatrix<float> *matFill(SparseMatrix<float> *, const float );
      template SparseMatrix<float> *matZeros(Basics::SparseMatrix<float> *);
      template SparseMatrix<float> *matOnes(SparseMatrix<float> *);
    
    } // namespace Initializers
        
  } // namespace MatrixExt
} // namespace AprilMath
