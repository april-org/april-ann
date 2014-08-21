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
#include "util_rprop.h"
#include "wrapper.h"

using basics::MatrixFloat;

namespace ANN {
  namespace optimizer {
    // FIXME: MAKE A CUDA IMPLEMENTATION
    void UtilRProp::step(MatrixFloat *steps,
			 MatrixFloat *old_sign,
			 MatrixFloat *sign,
			 float eta_minus,
			 float eta_plus) {
      april_assert(steps->sameDim(old_sign) && steps->sameDim(sign));
      april_assert(steps->getNumDim() == 2);
      april_assert(steps->getMajorOrder() == CblasColMajor);
      //
      MatrixFloat::col_major_iterator steps_it(steps->begin());
      MatrixFloat::const_col_major_iterator old_sign_it(old_sign->begin());
      MatrixFloat::const_col_major_iterator sign_it(sign->begin());
      //
      while(steps_it != steps->end()) {
	(*steps_it) *= (*old_sign_it != *sign_it) ? eta_minus : eta_plus;
	//
	++steps_it;
	++old_sign_it;
	++sign_it;
      }
    }
  }
}
