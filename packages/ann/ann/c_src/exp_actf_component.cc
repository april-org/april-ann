/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#include "exp_actf_component.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  ExpActfANNComponent::ExpActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  ExpActfANNComponent::~ExpActfANNComponent() { }

  void ExpActfANNComponent::applyActivation(MatrixFloat *input_units,
					    MatrixFloat *output_units) {
    matExp(input_units, output_units);
  }

  void ExpActfANNComponent::multiplyDerivatives(MatrixFloat *input_units,
                                                MatrixFloat *output_units,
                                                MatrixFloat *input_errors,
                                                MatrixFloat *output_errors) {
    UNUSED_VARIABLE(input_units);
    matCopy(output_errors, output_units);
    matCmul(output_errors, input_errors);
  }

  ANNComponent *ExpActfANNComponent::clone() {
    ExpActfANNComponent *obj = new ExpActfANNComponent(name.c_str());
    return obj;
  }

  const char *ExpActfANNComponent::luaCtorName() const {
    return "ann.components.actf.exp";
  }
}
