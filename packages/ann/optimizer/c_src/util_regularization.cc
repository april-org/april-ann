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
#include <cmath>
#include "maxmin.h"
#include "util_regularization.h"
#include "wrapper.h"

using Basics::MatrixFloat;

namespace ANN {
  namespace optimizer {
    // FIXME: MAKE A CUDA IMPLEMENTATION
    void UtilRegularization::L1NormMap(MatrixFloat *dest,
				       float value,
				       MatrixFloat *w) {
      april_assert(dest->sameDim(w));
      april_assert(dest->getNumDim() == 2);
      april_assert(dest->getMajorOrder() == CblasColMajor);
      //
      MatrixFloat::col_major_iterator dest_it(dest->begin());
      MatrixFloat::const_col_major_iterator w_it(w->begin());
      //
      while(dest_it != dest->end()) {
	float x = *dest_it;
	float y = *w_it;
	if (y > 0.0f)      *dest_it = AprilUtils::max(-y, x-value);
	else if (y < 0.0f) *dest_it = AprilUtils::min(-y, x+value);
	else if (fabsf(x) < value) *dest_it = 0.0f;
	//
	++dest_it;
	++w_it;
      }
    }
  }
}
