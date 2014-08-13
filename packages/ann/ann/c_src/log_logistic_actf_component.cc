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
#include "log_logistic_actf_component.h"
#include "wrapper.h"

using namespace basics;
using namespace april_utils;
using namespace april_math;

namespace ANN {

  LogLogisticActfANNComponent::LogLogisticActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  LogLogisticActfANNComponent::~LogLogisticActfANNComponent() { }

  void LogLogisticActfANNComponent::applyActivation(FloatGPUMirroredMemoryBlock *input_units,
						    FloatGPUMirroredMemoryBlock *output_units,
						    unsigned int size,
						    unsigned int bunch_size) {
    doApplyLogLogisticActivation(input_units,
				 output_units,
				 size,
				 bunch_size,
				 use_cuda);
  }

  void LogLogisticActfANNComponent::multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
							FloatGPUMirroredMemoryBlock *output_units,
							FloatGPUMirroredMemoryBlock *input_errors,
							FloatGPUMirroredMemoryBlock *output_errors,
							unsigned int size,
							unsigned int bunch_size) {
    UNUSED_VARIABLE(input_units);
    UNUSED_VARIABLE(output_units);
    UNUSED_VARIABLE(bunch_size);
    UNUSED_VARIABLE(size);
    // This activation function derivative is cancelled by cross-entropy
    // derivative. It only could be used with cross entropy loss function.
    doCopy(input_errors->getSize(),
	   input_errors, 0, 1,
	   output_errors, 0, 1,
	   use_cuda);
  }

  ANNComponent *LogLogisticActfANNComponent::clone() {
    LogLogisticActfANNComponent *obj = new LogLogisticActfANNComponent(name.c_str());
    return obj;
  }

  char *LogLogisticActfANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.actf.log_logistic{ name='%s' }", name.c_str());
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
