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
#include "softsign_actf_component.h"
#include "unused_variable.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  SoftsignActfANNComponent::SoftsignActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  SoftsignActfANNComponent::~SoftsignActfANNComponent() { }

  void SoftsignActfANNComponent::applyActivation(MatrixFloat *input_units,
						 MatrixFloat *output_units) {
    Kernels::applySoftsign(output_units, input_units);
  }

  void SoftsignActfANNComponent::multiplyDerivatives(MatrixFloat *input_units,
						     MatrixFloat *output_units,
						     MatrixFloat *input_errors,
						     MatrixFloat *output_errors) {
    UNUSED_VARIABLE(input_units);
    Kernels::applySoftsignDerivative(output_errors, output_units);
    matCmul(output_errors, input_errors);
  }

  ANNComponent *SoftsignActfANNComponent::clone() {
    SoftsignActfANNComponent *obj = new SoftsignActfANNComponent(name.c_str());
    return obj;
  }

  char *SoftsignActfANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.actf.softsign{ name='%s' }", name.c_str());
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }

  const char *SoftsignActfANNComponent::luaCtorName() const {
    return "ann.components.actf.softsign";
  }
}
