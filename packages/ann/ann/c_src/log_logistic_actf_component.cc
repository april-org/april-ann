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
#include "log_logistic_actf_component.h"
#include "unused_variable.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  LogLogisticActfANNComponent::LogLogisticActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  LogLogisticActfANNComponent::~LogLogisticActfANNComponent() { }

  void LogLogisticActfANNComponent::applyActivation(MatrixFloat *input_units,
                                                    MatrixFloat *output_units) {
    Kernels::applyLogLogistic(output_units, input_units);
  }

  void LogLogisticActfANNComponent::multiplyDerivatives(MatrixFloat *input_units,
							MatrixFloat *output_units,
							MatrixFloat *input_errors,
							MatrixFloat *output_errors) {
    UNUSED_VARIABLE(input_units);
    UNUSED_VARIABLE(output_units);
    // This activation function derivative is cancelled by cross-entropy
    // derivative. It must be used with cross entropy loss function.
    matCopy(output_errors, input_errors);
  }

  ANNComponent *LogLogisticActfANNComponent::clone() {
    LogLogisticActfANNComponent *obj = new LogLogisticActfANNComponent(name.c_str());
    return obj;
  }

  const char *LogLogisticActfANNComponent::luaCtorName() const {
    return "ann.components.actf.log_logistic";
  }
}
