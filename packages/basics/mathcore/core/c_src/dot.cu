/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
 *
 * The APRIL-MLP toolkit is free software; you can redistribute it and/or modify it
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
#include "cblas_headers.h"
#include "cuda_utils.h"
#include "unused_variable.h"

namespace AprilMath {

#ifdef USE_CUDA
  /***************************************
   ************** CUDA SECTION ***********
   ***************************************/

  cublasStatus_t wrapperCublasDot(cublasHandle_t &handle,
                                  unsigned int size,
                                  const float *x_mem,
                                  unsigned int x_inc,
                                  const float *y_mem,
                                  unsigned int y_inc,
                                  float *ret) {
    return cublasSdot(handle,
                      size,
                      x_mem, x_inc,
                      y_mem, y_inc,
                      ret);
  }

  cublasStatus_t wrapperCublasDot(cublasHandle_t &handle,
                                  unsigned int size,
                                  const double *x_mem,
                                  unsigned int x_inc,
                                  const double *y_mem,
                                  unsigned int y_inc,
                                  double *ret) {
    return cublasDdot(handle,
                      size,
                      x_mem, x_inc,
                      y_mem, y_inc,
                      ret);
  }

  cublasStatus_t wrapperCublasDot(cublasHandle_t &handle,
                                  unsigned int size,
                                  const ComplexF *x_mem,
                                  unsigned int x_inc,
                                  const ComplexF *y_mem,
                                  unsigned int y_inc,
                                  ComplexF *ret) {
    ERROR_EXIT(256, "Dot product for complex numbers not implemented with CUDA\n");
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }

#endif

  /***************************************
   ************* CBLAS SECTION ***********
   ***************************************/

  float wrapperCblasDot(unsigned int size,
                        const float *x_mem, unsigned int x_inc,
                        const float *y_mem, unsigned int y_inc) {
    return cblas_sdot(size,
                      x_mem, x_inc,
                      y_mem, y_inc);
  }

  double wrapperCblasDot(unsigned int size,
                         const double *x_mem, unsigned int x_inc,
                         const double *y_mem, unsigned int y_inc) {
    return cblas_ddot(size,
                      x_mem, x_inc,
                      y_mem, y_inc);
  }

  ComplexF wrapperCblasDot(unsigned int size,
                           const ComplexF *x_mem, unsigned int x_inc,
                           const ComplexF *y_mem, unsigned int y_inc) {
    ComplexF ret;
    cblas_zdotu_sub(size,
                    x_mem, x_inc,
                    y_mem, y_inc,
                    &ret);
    return ret;
  }

  /***************************************
   *********** TEMPLATE SECTION **********
   ***************************************/

  template <typename T>
  T doDot(unsigned int size,
          const GPUMirroredMemoryBlock<T> *x,
          unsigned int x_inc,
          unsigned int x_shift,
          const GPUMirroredMemoryBlock<T> *y,
          unsigned int y_inc,
          unsigned int y_shift,
          bool use_gpu) {
    const T *x_mem;
    const T *y_mem;
    T ret;
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      cublasStatus_t status;
      cublasHandle_t handle = GPUHelper::getHandler();
      x_mem = x->getGPUForRead() + x_shift;
      y_mem = y->getGPUForRead() + y_shift;
    
      status = cublasSetStream(handle, GPUHelper::getCurrentStream());
      checkCublasError(status);
      status = wrapperCublasDot(handle, size, x_mem, x_inc, y_mem, y_inc, &ret);
      checkCublasError(status);
    }
    else {
#endif
      x_mem = x->getPPALForRead() + x_shift;
      y_mem = y->getPPALForRead() + y_shift;
    
      ret = wrapperCblasDot(size,
                            x_mem, x_inc,
                            y_mem, y_inc);
#ifdef USE_CUDA
    }
#endif
    return ret;
  }

  template<typename T>
  T doSparseDot(int NNZ,
                const GPUMirroredMemoryBlock<T> *x_values,
                const Int32GPUMirroredMemoryBlock *x_indices,
                const GPUMirroredMemoryBlock<T> *y,
                int y_shift,
                int y_inc,
                bool use_gpu) {
    const T *x_values_mem;
    const int *x_indices_mem;
    const T *y_mem;
    T ret;
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu)
      ERROR_PRINT("CUDA sparse DOT not implemented\n");
#endif
    x_values_mem  = x_values->getPPALForRead();
    x_indices_mem = x_indices->getPPALForRead();
    y_mem = y->getPPALForRead() + y_shift;
    ret = cblas_sparse_dot(NNZ,
                           x_values_mem,
                           x_indices_mem,
                           y_mem,
                           y_inc);
    return ret;
  }

  template float doDot<float>(unsigned int,
                              const GPUMirroredMemoryBlock<float> *,
                              unsigned int,
                              unsigned int,
                              const GPUMirroredMemoryBlock<float> *,
                              unsigned int,
                              unsigned int,
                              bool);

  template double doDot<double>(unsigned int,
                                const GPUMirroredMemoryBlock<double> *,
                                unsigned int,
                                unsigned int,
                                const GPUMirroredMemoryBlock<double> *,
                                unsigned int,
                                unsigned int,
                                bool);

  template ComplexF doDot<ComplexF>(unsigned int,
                                    const GPUMirroredMemoryBlock<ComplexF> *,
                                    unsigned int,
                                    unsigned int,
                                    const GPUMirroredMemoryBlock<ComplexF> *,
                                    unsigned int,
                                    unsigned int,
                                    bool);

  template float doSparseDot(int NNZ,
                             const GPUMirroredMemoryBlock<float> *x_values,
                             const Int32GPUMirroredMemoryBlock *x_indices,
                             const GPUMirroredMemoryBlock<float> *y,
                             int y_shift,
                             int y_inc,
                             bool use_gpu);

  template double doSparseDot(int NNZ,
                              const GPUMirroredMemoryBlock<double> *x_values,
                              const Int32GPUMirroredMemoryBlock *x_indices,
                              const GPUMirroredMemoryBlock<double> *y,
                              int y_shift,
                              int y_inc,
                              bool use_gpu);

  template ComplexF doSparseDot(int NNZ,
                                const GPUMirroredMemoryBlock<ComplexF> *x_values,
                                const Int32GPUMirroredMemoryBlock *x_indices,
                                const GPUMirroredMemoryBlock<ComplexF> *y,
                                int y_shift,
                                int y_inc,
                                bool use_gpu);

} // namespace AprilMath
