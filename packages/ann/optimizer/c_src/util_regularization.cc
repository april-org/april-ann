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
namespace ANN {
  namespace optimizer {
    // FIXME: MAKE A CUDA IMPLEMENTATION
    void UtilRegularization::L1NormMap(MatrixFloat *destw,
				       float value,
				       MatrixFloat *w) {
      april_assert(destw->sameDim(destw));
      april_assert(destw->getNumDim() == 2);
      april_assert(destw->getMajorOrder() == CblasColMajor);
      //
      MatrixFloat::col_major_iterator destw_it(destw->begin());
      MatrixFloat::const_col_major_iterator w_it(w->begin());
      //
      while(destw_it != destw->end()) {
	float x = *destw_it;
	float y = *w_it;
	if (y > 0.0f)      *destw_it = april_utils::max(0.0f, x-value);
	else if (y < 0.0f) *destw_it = april_utils::min(0.0f, x+value);
	else if (fabsf(x) < value) *destw_it = 0.0f;
	//
	++destw_it;
	++w_it;
      }
    }
  }
}
