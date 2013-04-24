#include "cblas_headers.h"
#include "log_logistic_actf_component.h"
#include "wrapper.h"

namespace ANN {

  LogLogisticActfANNComponent::LogLogisticActfANNComponent(const char *name) :
    ActivationFunctionANNComponent(name) { }
  LogLogisticActfANNComponent::~LogLogisticActfANNComponent() { }

  void LogLogisticActfANNComponent::applyActivation(FloatGPUMirroredMemoryBlock *input_units,
						    FloatGPUMirroredMemoryBlock *output_units,
						    unsigned int size,
						    unsigned int bunch_size) {
    doApplyLogLogisticActivation(input_units,
				 output_units,
				 size,
				 bunch_size,
				 use_cuda);
  }

  void LogLogisticActfANNComponent::multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
							FloatGPUMirroredMemoryBlock *output_units,
							FloatGPUMirroredMemoryBlock *input_errors,
							FloatGPUMirroredMemoryBlock *output_errors,
							unsigned int size,
							unsigned int bunch_size,
							bool is_output) {
    // This activation function derivative is cancelled by cross-entropy
    // derivative. It only could be used with cross entropy loss function.
    doScopy(input_errors->getSize(),
	    input_errors, 0, 1,
	    output_errors, 0, 1,
	    use_cuda);
  }

  ANNComponent *LogLogisticActfANNComponent::clone() {
    return new LogLogisticActfANNComponent(name.c_str());
  }

}
