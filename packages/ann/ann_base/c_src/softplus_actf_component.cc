#include "cblas_headers.h"
#include "softplus_actf_component.h"
#include "wrapper.h"

namespace ANN {

  SoftplusActfANNComponent::SoftplusActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  SoftplusActfANNComponent::~SoftplusActfANNComponent() { }

  void SoftplusActfANNComponent::applyActivation(FloatGPUMirroredMemoryBlock *input_units,
						 FloatGPUMirroredMemoryBlock *output_units,
						 unsigned int size,
						 unsigned int bunch_size) {
    doApplySoftplusActivation(input_units,
			      output_units,
			      size,
			      bunch_size,
			      use_cuda);
  }

  void SoftplusActfANNComponent::multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
						     FloatGPUMirroredMemoryBlock *output_units,
						     FloatGPUMirroredMemoryBlock *input_errors,
						     FloatGPUMirroredMemoryBlock *output_errors,
						     unsigned int size,
						     unsigned int bunch_size) {
    doMultiplySoftplusDerivatives(output_units,
				  input_errors,
				  output_errors,
				  size,
				  bunch_size,
				  use_cuda);
  }

  ANNComponent *SoftplusActfANNComponent::clone() {
    return new SoftplusActfANNComponent(name.c_str());
  }

}
