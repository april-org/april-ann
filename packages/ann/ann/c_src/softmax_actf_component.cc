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
#include "activation_function_kernels.h"
#include "cblas_headers.h"
#include "softmax_actf_component.h"
#include "unused_variable.h"

using namespace AprilMath;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  SoftmaxActfANNComponent::SoftmaxActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name, true) { }
  SoftmaxActfANNComponent::~SoftmaxActfANNComponent() { }

  void SoftmaxActfANNComponent::applyActivation(MatrixFloat *input_units,
						MatrixFloat *output_units) {
    Kernels::applySoftmax(output_units, input_units);
  }

  void SoftmaxActfANNComponent::multiplyDerivatives(MatrixFloat *input_units,
						    MatrixFloat *output_units,
						    MatrixFloat *input_errors,
						    MatrixFloat *output_errors) {
    UNUSED_VARIABLE(input_units);
    Kernels::applySoftmaxDerivative(output_errors, input_errors, output_units);
  }

  ANNComponent *SoftmaxActfANNComponent::clone() {
    SoftmaxActfANNComponent *obj = new SoftmaxActfANNComponent(name.c_str());
    return obj;
  }

  char *SoftmaxActfANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.actf.softmax{ name='%s' }", name.c_str());
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }

}
