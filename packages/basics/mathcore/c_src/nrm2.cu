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
#include "mathcore.h"
#include "unused_variable.h"

namespace AprilMath {

#ifdef USE_CUDA
  namespace CUDA {
    /***************************************
     ************** CUDA SECTION ***********
     ***************************************/

    cublasStatus_t wrapperCublasNrm2(cublasHandle_t &handle,
                                     unsigned int n,
                                     const float *x_mem,
                                     unsigned int x_inc,
                                     float *result) {
      return cublasSnrm2(handle, n, x_mem, x_inc, result);
    }

    cublasStatus_t wrapperCublasNrm2(cublasHandle_t &handle,
                                     unsigned int n,
                                     const double *x_mem,
                                     unsigned int x_inc,
                                     float *result) {
      double aux;
      cublasStatus_t status = cublasDnrm2(handle, n, x_mem, x_inc, &aux);
      *result = aux;
      return status;
    }

    cublasStatus_t wrapperCublasNrm2(cublasHandle_t &handle,
                                     unsigned int size,
                                     const ComplexF *x_mem,
                                     unsigned int x_inc,
                                     float *result) {
      UNUSED_VARIABLE(handle);
      UNUSED_VARIABLE(size);
      UNUSED_VARIABLE(x_mem);
      UNUSED_VARIABLE(x_inc);
      UNUSED_VARIABLE(result);
      ERROR_EXIT(256, "Nrm2 for complex numbers not implemented in CUDA\n");
      return CUBLAS_STATUS_INTERNAL_ERROR;
    }
  } // namespace CUDA
#endif

  /***************************************
   ************* CBLAS SECTION ***********
   ***************************************/

  float wrapperCblasNrm2(unsigned int size,
                         const float *x_mem, unsigned int x_inc) {
    return cblas_snrm2(size, x_mem, x_inc);
  }

  float wrapperCblasNrm2(unsigned int size,
                         const double *x_mem, unsigned int x_inc) {
    return cblas_dnrm2(size, x_mem, x_inc);
  }

  float wrapperCblasNrm2(unsigned int size,
                         const ComplexF *x_mem, unsigned int x_inc) {
    return cblas_scnrm2(size, x_mem, x_inc);
  }

  /***************************************
   *********** TEMPLATE SECTION **********
   ***************************************/

  template <typename T>
  float doNrm2(unsigned int n,
               const GPUMirroredMemoryBlock<T> *x,
               unsigned int inc,
               unsigned int shift,
               bool use_gpu,
               float zero,
               GPUMirroredMemoryBlock<float> *dest,
               unsigned int dest_raw_pos) {
    float result = zero;
    const T *x_mem;
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      cublasStatus_t status;
      cublasHandle_t handle = CUDA::GPUHelper::getHandler();
      x_mem  = x->getGPUForRead() + shift;
      status = cublasSetStream(handle, CUDA::GPUHelper::getCurrentStream());
      checkCublasError(status);
      status = CUDA::wrapperCublasNrm2(handle, n, x_mem, inc, &result);
      checkCublasError(status);
    }
    else {
#endif
      x_mem = x->getPPALForRead() + shift;
      result = wrapperCblasNrm2(n, x_mem, inc);
#ifdef USE_CUDA
    }
#endif
    if (dest != 0) dest->putValue(dest_raw_pos, result);
    return result;
  }

  template float doNrm2<float>(unsigned int n,
                               const GPUMirroredMemoryBlock<float> *x,
                               unsigned int inc,
                               unsigned int shift,
                               bool use_gpu,
                               float,
                               GPUMirroredMemoryBlock<float> *,
                               unsigned int);

  template float doNrm2<double>(unsigned int n,
                                const GPUMirroredMemoryBlock<double> *x,
                                unsigned int inc,
                                unsigned int shift,
                                bool use_gpu,
                                float,
                                GPUMirroredMemoryBlock<float> *,
                                unsigned int);

  template float doNrm2<ComplexF>(unsigned int n,
                                  const GPUMirroredMemoryBlock<ComplexF> *x,
                                  unsigned int inc,
                                  unsigned int shift,
                                  bool use_gpu,
                                  float,
                                  GPUMirroredMemoryBlock<float> *,
                                  unsigned int);

} // namespace AprilMath
