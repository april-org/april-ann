#include "cblas_headers.h"
#include "logistic_actf_component.h"
#include "wrapper.h"

namespace ANN {

  LogisticActfANNComponent::LogisticActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  LogisticActfANNComponent::~LogisticActfANNComponent() { }

  void LogisticActfANNComponent::applyActivation(FloatGPUMirroredMemoryBlock *input_units,
						 FloatGPUMirroredMemoryBlock *output_units,
						 unsigned int size,
						 unsigned int bunch_size) {
    doApplyLogisticActivation(input_units,
			      output_units,
			      size,
			      bunch_size,
			      use_cuda);
  }

  void LogisticActfANNComponent::multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
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

  ANNComponent *LogisticActfANNComponent::clone() {
    return new LogisticActfANNComponent(name.c_str());
  }

}
