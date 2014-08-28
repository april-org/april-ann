  // LOSS FUNCTIONS
  void doMSELossFunction(FloatGPUMirroredMemoryBlock *input,
                         FloatGPUMirroredMemoryBlock *target,
                         FloatGPUMirroredMemoryBlock *loss_output,
                         float zero_epsilon_distance,
                         unsigned int size,
                         unsigned int bunch_size,
                         bool use_gpu);

  void doComputeMSEGradient(FloatGPUMirroredMemoryBlock *input,
                            FloatGPUMirroredMemoryBlock *target,
                            FloatGPUMirroredMemoryBlock *error_output,
                            float zero_epsilon_distance,
                            unsigned int size,
                            unsigned int bunch_size,
                            bool use_gpu);

  void doMAELossFunction(FloatGPUMirroredMemoryBlock *input,
                         FloatGPUMirroredMemoryBlock *target,
                         FloatGPUMirroredMemoryBlock *loss_output,
                         float zero_epsilon_distance,
                         unsigned int size,
                         unsigned int bunch_size,
                         bool use_gpu);

  void doComputeMAEGradient(FloatGPUMirroredMemoryBlock *input,
                            FloatGPUMirroredMemoryBlock *target,
                            FloatGPUMirroredMemoryBlock *error_output,
                            float zero_epsilon_distance,
                            unsigned int size,
                            unsigned int bunch_size,
                            bool use_gpu);

  void doCrossEntropyLossFunction(FloatGPUMirroredMemoryBlock *input,
                                  FloatGPUMirroredMemoryBlock *target,
                                  FloatGPUMirroredMemoryBlock *loss_output,
                                  float epsilon,
                                  unsigned int size,
                                  unsigned int bunch_size,
                                  bool use_gpu);

  void doMultiClassCrossEntropyLossFunction(FloatGPUMirroredMemoryBlock *input,
                                            FloatGPUMirroredMemoryBlock *target,
                                            FloatGPUMirroredMemoryBlock *loss_output,
                                            float epsilon,
                                            unsigned int size,
                                            unsigned int bunch_size,
                                            bool use_gpu);

  void doComputeCrossEntropyGradient(FloatGPUMirroredMemoryBlock *input,
                                     FloatGPUMirroredMemoryBlock *target,
                                     FloatGPUMirroredMemoryBlock *error_output,
                                     float epsilon,
                                     unsigned int size,
                                     unsigned int bunch_size,
                                     bool use_gpu);

  /*
    void doCalculateTanhErrorFunction(FloatGPUMirroredMemoryBlock *output,
    FloatGPUMirroredMemoryBlock *target_output,
    FloatGPUMirroredMemoryBlock *output_error,
    FloatGPUMirroredMemoryBlock *pattern_errors,
    unsigned int output_size,
    const ANNConfiguration &conf,
    bool use_gpu);
  */
