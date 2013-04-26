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
    // This activation function derivative is cancelled by cross-entropy
    // derivative. It only could be used with cross entropy loss function.
    doScopy(input_errors->getSize(),
	    input_errors, 0, 1,
	    output_errors, 0, 1,
	    use_cuda);
  }
  
  ANNComponent *LogSoftmaxActfANNComponent::clone() {
    return new LogSoftmaxActfANNComponent(name.c_str());
  }
  
}
