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
#include "ann_component.h"

using namespace Basics;
using namespace AprilUtils;
using namespace AprilMath;

namespace ANN {
  unsigned int ANNComponent::next_name_id    = 1;
  unsigned int ANNComponent::next_weights_id = 1;

  unsigned int mult(const int *v, int n) {
    int m = 1;
    for (int i=0; i<n; ++i) m *= v[i];
    return m;
  }
}
