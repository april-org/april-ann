
  template<typename T>
  void doDiv(unsigned int N,
             GPUMirroredMemoryBlock<T> *v,
             unsigned int stride,
             unsigned int shift,
             T value,
             bool use_gpu);


  template<typename T>
  void doFill(unsigned int N,
              GPUMirroredMemoryBlock<T> *v,
              unsigned int stride,
              unsigned int shift,
              T value,
              bool use_gpu);

  template<typename T>
  void doScalarAdd(unsigned int N,
                   GPUMirroredMemoryBlock<T> *v,
                   unsigned int stride,
                   unsigned int shift,
                   T value,
                   bool use_gpu);

  template<typename T>
  bool doEquals(unsigned int N,
                const GPUMirroredMemoryBlock<T> *v1,
                const GPUMirroredMemoryBlock<T> *v2,
                unsigned int stride1,
                unsigned int stride2,
                unsigned int shift1,
                unsigned int shift2,
                float epsilon,
                bool use_gpu);

  template <typename T>
  void doCmul(int N,
              const GPUMirroredMemoryBlock<T>* x,
              unsigned int x_shift,
              unsigned int x_inc,
              GPUMirroredMemoryBlock<T>* y,
              unsigned int y_shift,
              unsigned int y_inc,
              bool use_gpu);
