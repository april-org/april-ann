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
#ifndef NRM2_H
#define NRM2_H
#include "mathcore.h"
#include "unused_variable.h"

namespace AprilMath {

#ifdef USE_CUDA
  namespace CUDA {
    /***************************************
     ************** CUDA SECTION ***********
     ***************************************/

    template<typename T>
    cublasStatus_t wrapperCublasNrm2(cublasHandle_t &handle,
                                     unsigned int size,
                                     const T *x_mem,
                                     unsigned int x_inc,
                                     float *result) {
      UNUSED_VARIABLE(handle);
      UNUSED_VARIABLE(size);
      UNUSED_VARIABLE(x_mem);
      UNUSED_VARIABLE(x_inc);
      UNUSED_VARIABLE(result);
      ERROR_EXIT(256, "NOT IMPLEMENTED\n");
      return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    template<>
    cublasStatus_t wrapperCublasNrm2<float>(cublasHandle_t &handle,
                                            unsigned int size,
                                            const float *x_mem,
                                            unsigned int x_inc,
                                            float *result);
    
  } // namespace CUDA
#endif

  /***************************************
   ************* CBLAS SECTION ***********
   ***************************************/
  template<typename T>
  float wrapperCblasNrm2(unsigned int size,
                         const T *x_mem, unsigned int x_inc) {
    UNUSED_VARIABLE(size);
    UNUSED_VARIABLE(x_mem);
    UNUSED_VARIABLE(x_inc);
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
    return 0.0f;
  }
  template<> float wrapperCblasNrm2<float>(unsigned int, const float *,
                                           unsigned int);
  template<> float wrapperCblasNrm2<double>(unsigned int, const double *,
                                            unsigned int);
  template<> float wrapperCblasNrm2<ComplexF>(unsigned int, const ComplexF *,
                                              unsigned int);

  /***************************************
   *********** TEMPLATE SECTION **********
   ***************************************/

  template <typename T, typename OP>
  void doNrm2(unsigned int n,
              const GPUMirroredMemoryBlock<T> *x,
              unsigned int inc,
              unsigned int shift,
              bool use_gpu,
              float zero,
              OP functor,
              GPUMirroredMemoryBlock<float> *dest,
              unsigned int dest_raw_pos) {
    UNUSED_VARIABLE(zero);
    GPUMirroredMemoryBlock<float> result(1);
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
      status = CUDA::wrapperCublasNrm2(handle, n, x_mem, inc, result.getGPUForWrite());
      checkCublasError(status);
    }
    else {
#endif
      x_mem = x->getPPALForRead() + shift;
      result.putValue(0u, wrapperCblasNrm2(n, x_mem, inc));
#ifdef USE_CUDA
    }
#endif
    float aux;
    dest->getValue(dest_raw_pos, aux);
    functor(aux, *(result.getPPALForRead()));
    dest->putValue(dest_raw_pos, aux);
  }

} // namespace AprilMath
#endif // NRM2_H
