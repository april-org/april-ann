#include "cblas_headers.h"
#include "softmax_actf_component.h"
#include "wrapper.h"
#include "ceiling_power_of_two.h"

using april_utils::ceilingPowerOfTwo;

namespace ANN {

  SoftmaxActfANNComponent::SoftmaxActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  SoftmaxActfANNComponent::~SoftmaxActfANNComponent() { }

  void SoftmaxActfANNComponent::applyActivation(FloatGPUMirroredMemoryBlock *input_units,
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
    doApplySoftmaxActivation(input_units,
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

  void SoftmaxActfANNComponent::multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
						    FloatGPUMirroredMemoryBlock *output_units,
						    FloatGPUMirroredMemoryBlock *input_errors,
						    FloatGPUMirroredMemoryBlock *output_errors,
						    unsigned int size,
						    unsigned int bunch_size) {
    doMultiplyLogisticDerivatives(output_units,
				  input_errors,
				  output_errors,
				  size,
				  bunch_size,
				  use_cuda);
  }

  ANNComponent *SoftmaxActfANNComponent::clone() {
    return new SoftmaxActfANNComponent(name.c_str());
  }

}
