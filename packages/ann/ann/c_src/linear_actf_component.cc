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
#include "unused_variable.h"
#include "cblas_headers.h"
#include "linear_actf_component.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  LinearActfANNComponent::LinearActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  LinearActfANNComponent::~LinearActfANNComponent() { }

  void LinearActfANNComponent::applyActivation(Basics::MatrixFloat *input_units,
					       Basics::MatrixFloat *output_units) {
    matCopy(output_units, input_units);
  }
  
  void LinearActfANNComponent::multiplyDerivatives(Basics::MatrixFloat *input_units,
						   Basics::MatrixFloat *output_units,
						   Basics::MatrixFloat *input_errors,
						   Basics::MatrixFloat *output_errors) {
    UNUSED_VARIABLE(input_units);
    UNUSED_VARIABLE(output_units);
    matCopy(output_errors, input_errors);
  }

  ANNComponent *LinearActfANNComponent::clone() {
    LinearActfANNComponent *obj = new LinearActfANNComponent(name.c_str());
    return obj;
  }

  char *LinearActfANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.actf.linear{ name='%s' }", name.c_str());
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
