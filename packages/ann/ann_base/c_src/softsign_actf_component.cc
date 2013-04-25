#include "cblas_headers.h"
#include "softsign_actf_component.h"
#include "wrapper.h"

namespace ANN {

  SoftsignActfANNComponent::SoftsignActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  SoftsignActfANNComponent::~SoftsignActfANNComponent() { }

  void SoftsignActfANNComponent::applyActivation(FloatGPUMirroredMemoryBlock *input_units,
						 FloatGPUMirroredMemoryBlock *output_units,
						 unsigned int size,
						 unsigned int bunch_size) {
    doApplySoftsignActivation(input_units,
			      output_units,
			      size,
			      bunch_size,
			      use_cuda);
  }

  void SoftsignActfANNComponent::multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
						     FloatGPUMirroredMemoryBlock *output_units,
						     FloatGPUMirroredMemoryBlock *input_errors,
						     FloatGPUMirroredMemoryBlock *output_errors,
						     unsigned int size,
						     unsigned int bunch_size) {
    doMultiplySoftsignDerivatives(output_units,
				  input_errors,
				  output_errors,
				  size,
				  bunch_size,
				  use_cuda);
  }

  ANNComponent *SoftsignActfANNComponent::clone() {
    return new SoftsignActfANNComponent(name.c_str());
  }

}
