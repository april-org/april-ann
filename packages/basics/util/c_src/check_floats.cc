/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#include "error_print.h"
#include "check_floats.h"

namespace AprilUtils {
  
  bool check_floats(float *v, unsigned int sz) {
    for (unsigned int i=0; i<sz; ++i) {
      if (!std::isfinite(v[i])) {
	ERROR_PRINT2("No finite number at position %d with value %g\n", i, v[i]);
	return false;
      }
      if (!std::isnormal(v[i])) v[i]=0.0f;
    }
    return true;
  }
}
