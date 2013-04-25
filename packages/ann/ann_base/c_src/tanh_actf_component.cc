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
    doMultiplyTanhDerivatives(output_units,
			      input_errors,
			      output_errors,
			      size,
			      bunch_size,
			      use_cuda);
  }

  ANNComponent *TanhActfANNComponent::clone() {
    return new TanhActfANNComponent(name.c_str());
  }

}
