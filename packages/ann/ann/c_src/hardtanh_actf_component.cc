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
#include "hardtanh_actf_component.h"
#include "wrapper.h"

using namespace Basics;
using namespace AprilUtils;
using namespace AprilMath;

namespace ANN {

  HardtanhActfANNComponent::HardtanhActfANNComponent(const char *name,
						     float inf, float sup) :
    ActivationFunctionANNComponent(name), inf(inf), sup(sup) { }
  HardtanhActfANNComponent::~HardtanhActfANNComponent() { }

  void HardtanhActfANNComponent::applyActivation(FloatGPUMirroredMemoryBlock *input_units,
						 FloatGPUMirroredMemoryBlock *output_units,
						 unsigned int size,
						 unsigned int bunch_size) {
    doApplyHardtanhActivation(input_units,
			      output_units,
			      size,
			      bunch_size,
			      inf, sup,
			      use_cuda);
  }

  void HardtanhActfANNComponent::multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
						     FloatGPUMirroredMemoryBlock *output_units,
						     FloatGPUMirroredMemoryBlock *input_errors,
						     FloatGPUMirroredMemoryBlock *output_errors,
						     unsigned int size,
						     unsigned int bunch_size) {
    UNUSED_VARIABLE(output_units);
    doMultiplyHardtanhDerivatives(input_units,
				  input_errors,
				  output_errors,
				  size,
				  bunch_size,
				  inf, sup,
				  use_cuda);
  }

  ANNComponent *HardtanhActfANNComponent::clone() {
    HardtanhActfANNComponent *obj = new HardtanhActfANNComponent(name.c_str());
    return obj;
  }
  
  char *HardtanhActfANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.actf.hardtanh{ name='%s', inf=%g, sup=%g }",
		  name.c_str(), inf, sup);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }

}
