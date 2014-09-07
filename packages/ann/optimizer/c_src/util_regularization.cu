/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#include "cuda_utils.h"
#include "map_matrix.h"
#include "util_regularization.h"

using Basics::MatrixFloat;

namespace ANN {
  namespace Optimizer {
    
    namespace Kernels {

      struct L1NormKernel {
        float value;
        L1NormKernel(float value) : value(value) { }
        APRIL_CUDA_EXPORT float operator()(const float &x,
                                           const float &y) {
          float result = x;
          if (y > 0.0f) result = AprilMath::m_max(-y, x-value);
          else if (y < 0.0f) result = AprilMath::m_min(-y, x+value);
          else if (AprilMath::m_abs(x) < value) result = 0.0f;          
          return result;
        }
      };

    }
    
    void UtilRegularization::L1NormMap(MatrixFloat *dest,
				       float value,
				       MatrixFloat *w) {
      april_assert(dest->sameDim(w));
      april_assert(dest->getNumDim() == 2);
      april_assert(dest->getMajorOrder() == CblasColMajor);
      //
      AprilMath::MatrixExt::MatrixScalarMap2(dest, w,
                                             Kernels::L1NormKernel(value),
                                             dest);
    }
  }
}
