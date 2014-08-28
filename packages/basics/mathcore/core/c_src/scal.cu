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

namespace april_math {

#ifdef USE_CUDA
  /***************************************
   ************** CUDA SECTION ***********
   ***************************************/

  cublasStatus_t wrapperCublasScal(cublasHandle_t &handle,
                                   int N,
                                   float *alpha,
                                   float *x_mem,
                                   unsigned int x_inc) {
    return cublasSscal(handle, N, alpha, x_mem, x_inc);
  }

  cublasStatus_t wrapperCublasScal(cublasHandle_t &handle,
                                   int N,
                                   ComplexF *alpha,
                                   ComplexF *x_mem,
                                   unsigned int x_inc) {
    return cublasCscal(handle, N, reinterpret_cast<cuComplex*>(alpha),
                       reinterpret_cast<cuComplex*>(x_mem), x_inc);
  }
#endif

  /***************************************
   ************* CBLAS SECTION ***********
   ***************************************/

  void wrapperCblasScal(int N, float alpha, float *x_mem, unsigned int x_inc) {
    cblas_sscal(N, alpha, x_mem, x_inc);
  }

  void wrapperCblasScal(int N, ComplexF alpha,
                        ComplexF *x_mem, unsigned int x_inc) {
    cblas_cscal(N, &alpha, x_mem, x_inc);
  }

  /***************************************
   *********** TEMPLATE SECTION **********
   ***************************************/

  template <typename T>
  void doScal(unsigned int size,
              GPUMirroredMemoryBlock<T> *x,
              unsigned int inc,
              unsigned int shift,
              T alpha,
              bool use_gpu) {
    T *x_mem;
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      cublasStatus_t status;
      cublasHandle_t handle = GPUHelper::getHandler();
      x_mem = x->getGPUForReadAndWrite() + shift;

      status = cublasSetStream(handle, GPUHelper::getCurrentStream());
      checkCublasError(status);

      status = wrapperCublasScal(handle, size, &alpha, x_mem, inc);

      checkCublasError(status);
    }
    else {
#endif
      x_mem = x->getPPALForReadAndWrite() + shift;
      wrapperCblasScal(size, alpha, x_mem, inc);
#ifdef USE_CUDA
    }
#endif
  }

  template void doScal<float>(unsigned int size,
                              GPUMirroredMemoryBlock<float> *x,
                              unsigned int inc,
                              unsigned int shift,
                              float alpha,
                              bool use_gpu);

  template void doScal<ComplexF>(unsigned int size,
                                 GPUMirroredMemoryBlock<ComplexF> *x,
                                 unsigned int inc,
                                 unsigned int shift,
                                 ComplexF alpha,
                                 bool use_gpu);

} // namespace april_math
