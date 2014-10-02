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
        APRIL_CUDA_EXPORT float operator()(const float &w) {
          float result = w;
          if (w > 0.0f) result = AprilMath::m_max(0.0f, w-value);
          else if (w < 0.0f) result = AprilMath::m_min(0.0f, w+value);
          return result;
        }
      };

    }
    
    void UtilRegularization::L1NormMap(MatrixFloat *w,
				       float value) {
      april_assert(w->getNumDim() == 2);
      april_assert(w->getMajorOrder() == CblasColMajor);
      //
      AprilMath::MatrixExt::MatrixScalarMap1(w,Kernels::L1NormKernel(value),w);
    }
  }
}
