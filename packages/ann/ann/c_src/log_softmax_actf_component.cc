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
#include "log_softmax_actf_component.h"
#include "wrapper.h"
#include "ceiling_power_of_two.h"

using april_utils::ceilingPowerOfTwo;

namespace ANN {

  LogSoftmaxActfANNComponent::LogSoftmaxActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  LogSoftmaxActfANNComponent::~LogSoftmaxActfANNComponent() { }

  void LogSoftmaxActfANNComponent::
  applyActivation(FloatGPUMirroredMemoryBlock *input_units,
		  FloatGPUMirroredMemoryBlock *output_units,
		  unsigned int size,
		  unsigned int bunch_size) {
    FloatGPUMirroredMemoryBlock *minimums = 0;
    FloatGPUMirroredMemoryBlock *maximums = 0;
    FloatGPUMirroredMemoryBlock *sums = 0;
    if (use_cuda) {
      unsigned int reduction_size = ceilingPowerOfTwo(size) >> 1;
      unsigned int mem_size = reduction_size * bunch_size;
      minimums = new FloatGPUMirroredMemoryBlock(mem_size);
      maximums = new FloatGPUMirroredMemoryBlock(mem_size);
      sums = new FloatGPUMirroredMemoryBlock(mem_size);
    }
    doApplyLogSoftmaxActivation(input_units,
				output_units,
				minimums,
				maximums,
				sums,
				size,
				bunch_size,
				use_cuda);
    if (use_cuda) {
      delete minimums;
      delete maximums;
      delete sums;
    }
  }

  void LogSoftmaxActfANNComponent::
  multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
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
  
  ANNComponent *LogSoftmaxActfANNComponent::clone() {
    LogSoftmaxActfANNComponent *obj = new LogSoftmaxActfANNComponent(name.c_str());
    return obj;
  }

  char *LogSoftmaxActfANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.actf.log_softmax{ name='%s' }", name.c_str());
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  
}
