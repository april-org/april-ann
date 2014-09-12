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
#ifndef COPY_IMPL_H
#define COPY_IMPL_H
#include "cblas_headers.h"
#include "cuda_utils.h"
#include "unused_variable.h"

namespace AprilMath {

#ifdef USE_CUDA
  namespace CUDA {
    
    /***************************************
     ************** CUDA SECTION ***********
     ***************************************/

    template<typename T>
    cublasStatus_t wrapperCublasCopy(cublasHandle_t &handle,
                                     int N,
                                     const T *x_mem,
                                     unsigned int x_inc,
                                     T *y_mem,
                                     unsigned int y_inc) {
      UNUSED_VARIABLE(handle);
      UNUSED_VARIABLE(N);
      UNUSED_VARIABLE(x_mem);
      UNUSED_VARIABLE(x_inc);
      UNUSED_VARIABLE(y_mem);
      UNUSED_VARIABLE(y_inc);
      ERROR_EXIT(128, "CUDA VERSION NOT IMPLEMENTED\n");
      return CUBLAS_STATUS_INTERNAL_ERROR;
    }

    template<>
    cublasStatus_t wrapperCublasCopy<float>(cublasHandle_t &handle,
                                            int N,
                                            const float *x_mem,
                                            unsigned int x_inc,
                                            float *y_mem,
                                            unsigned int y_inc);

    template<>
    cublasStatus_t wrapperCublasCopy<double>(cublasHandle_t &handle,
                                             int N,
                                             const double *x_mem,
                                             unsigned int x_inc,
                                             double *y_mem,
                                             unsigned int y_inc);
    
    template<>
    cublasStatus_t wrapperCublasCopy<ComplexF>(cublasHandle_t &handle,
                                               int N,
                                               const ComplexF *x_mem,
                                               unsigned int x_inc,
                                               ComplexF *y_mem,
                                               unsigned int y_inc);

  } // namespace CUDA
#endif

  /***************************************
   ************* CBLAS SECTION ***********
   ***************************************/

  template<typename T>
  void wrapperCblasCopy(int N, const T *x_mem, unsigned int x_inc,
                        T *y_mem, unsigned int y_inc) {
    for (int i=0; i<N; ++i, x_mem += x_inc, y_mem += y_inc) *y_mem = *x_mem;
  }

  template<>
  void wrapperCblasCopy<float>(int N, const float *x_mem, unsigned int x_inc,
                               float *y_mem, unsigned int y_inc);

  template<>
  void wrapperCblasCopy<double>(int N, const double *x_mem, unsigned int x_inc,
                                double *y_mem, unsigned int y_inc);

  template<>
  void wrapperCblasCopy<ComplexF>(int N, const ComplexF *x_mem, unsigned int x_inc,
                                  ComplexF *y_mem, unsigned int y_inc);

  /***************************************
   *********** TEMPLATE SECTION **********
   ***************************************/

  template<typename T>
  void doCopy(int N,
              const GPUMirroredMemoryBlock<T>* x,
              unsigned int x_inc,
              unsigned int x_shift,
              GPUMirroredMemoryBlock<T>* y,
              unsigned int y_inc,
              unsigned int y_shift,
              bool use_gpu)
  {
    const T *x_mem;
    T *y_mem;
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      cublasStatus_t status;
      cublasHandle_t handle = CUDA::GPUHelper::getHandler();
      //printf("Doing a scopy with comp=1 & cuda=1\n");
      x_mem = x->getGPUForRead() + x_shift;
      y_mem = y->getGPUForWrite() + y_shift;
    
      status = cublasSetStream(handle, CUDA::GPUHelper::getCurrentStream());
      checkCublasError(status);
    
      status = CUDA::wrapperCublasCopy(handle, N, x_mem, x_inc, y_mem, y_inc);
    
      checkCublasError(status);
    }
    else {
      //printf("Doing a scopy with comp=1 & cuda=0\n");
#endif
#ifndef USE_CUDA
      //printf("Doing a scopy with comp=0 & cuda=0\n");
#endif
      x_mem = x->getPPALForRead() + x_shift;
      y_mem = y->getPPALForWrite() + y_shift;

      wrapperCblasCopy(N, x_mem, x_inc, y_mem, y_inc);
#ifdef USE_CUDA
    }
#endif
  }

} // namespace AprilMath
#endif // COPY_IMPL_H

