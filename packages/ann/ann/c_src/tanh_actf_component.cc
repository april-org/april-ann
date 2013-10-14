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
#include "tanh_actf_component.h"
#include "wrapper.h"

namespace ANN {

  TanhActfANNComponent::TanhActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  TanhActfANNComponent::~TanhActfANNComponent() { }

  void TanhActfANNComponent::applyActivation(FloatGPUMirroredMemoryBlock *input_units,
					     FloatGPUMirroredMemoryBlock *output_units,
					     unsigned int size,
					     unsigned int bunch_size) {
    doApplyTanhActivation(input_units,
			  output_units,
			  size,
			  bunch_size,
			  use_cuda);
  }

  void TanhActfANNComponent::multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
						 FloatGPUMirroredMemoryBlock *output_units,
						 FloatGPUMirroredMemoryBlock *input_errors,
						 FloatGPUMirroredMemoryBlock *output_errors,
						 unsigned int size,
						 unsigned int bunch_size) {
    UNUSED_VARIABLE(input_units);
    doMultiplyTanhDerivatives(output_units,
			      input_errors,
			      output_errors,
			      size,
			      bunch_size,
			      use_cuda);
  }

  ANNComponent *TanhActfANNComponent::clone() {
    TanhActfANNComponent *obj = new TanhActfANNComponent(name.c_str());
    obj->setOption(DROPOUT_FACTOR_STRING, getOption(DROPOUT_FACTOR_STRING));
    return obj;
  }

  char *TanhActfANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.actf.tanh{ name='%s' }", name.c_str());
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }

}
