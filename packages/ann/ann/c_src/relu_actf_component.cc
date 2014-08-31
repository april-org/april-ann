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
#include "relu_actf_component.h"
#include "wrapper.h"

using namespace Basics;
using namespace AprilUtils;
using namespace AprilMath;

namespace ANN {

  ReLUActfANNComponent::ReLUActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  ReLUActfANNComponent::~ReLUActfANNComponent() { }

  void ReLUActfANNComponent::applyActivation(FloatGPUMirroredMemoryBlock *input_units,
					     FloatGPUMirroredMemoryBlock *output_units,
					     unsigned int size,
					     unsigned int bunch_size) {
    doApplyReLUActivation(input_units,
			  output_units,
			  size,
			  bunch_size,
			  use_cuda);
  }

  void ReLUActfANNComponent::multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
						 FloatGPUMirroredMemoryBlock *output_units,
						 FloatGPUMirroredMemoryBlock *input_errors,
						 FloatGPUMirroredMemoryBlock *output_errors,
						 unsigned int size,
						 unsigned int bunch_size) {
    UNUSED_VARIABLE(output_units);
    doMultiplyReLUDerivatives(input_units,
			      input_errors,
			      output_errors,
			      size,
			      bunch_size,
			      use_cuda);
  }

  ANNComponent *ReLUActfANNComponent::clone() {
    ReLUActfANNComponent *obj = new ReLUActfANNComponent(name.c_str());
    return obj;
  }

  char *ReLUActfANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.actf.relu{ name='%s' }", name.c_str());
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }

}
