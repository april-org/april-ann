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
#include "cuda_utils.h"
#include "map_matrix.impl.h"
#include "util_rprop.h"

using Basics::MatrixFloat;


namespace ANN {
  namespace Optimizer {

    namespace Kernels {
      struct RPropKernel {
        float eta_minus;
        float eta_plus;
        RPropKernel(float eta_minus, float eta_plus) : eta_minus(eta_minus),
                                                       eta_plus(eta_plus) {
        }
        APRIL_CUDA_EXPORT float operator()(const float &a, const float &b) {
          return (a != b) ? eta_minus : eta_plus;
        }
      };
    }

    void UtilRProp::step(MatrixFloat *steps,
			 MatrixFloat *old_sign,
			 MatrixFloat *sign,
			 float eta_minus,
			 float eta_plus) {
      april_assert(steps->sameDim(old_sign) && steps->sameDim(sign));
      april_assert(steps->getNumDim() == 2);
      april_assert(steps->getMajorOrder() == CblasColMajor);
      //
      AprilMath::MatrixExt::
        MatrixScalarMap2(old_sign, sign,
                         Kernels::RPropKernel(eta_minus, eta_plus),
                         steps);
    }
  }
}
