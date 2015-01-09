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

  /**
   * @brief doNrm2 reduction, it is applied over a vector span, and follows the
   * CBLAS wrapper API.
   *
   * @tparam T - The type of the input/output data.
   * @tparam OP - A functor typename, needed to reduce the partial result of
   * dest pointer in case @c set_dest_to_zero=false
   *
   * @param n - The size of the vector.
   * @param x - The input vector.
   * @param inc - The stride between consecutive elements.
   * @param shift - The position of the first valid element.
   * @param use_gpu - Indicates if do computation using GPU or not.
   * @param zero - The initial value of the reduction.
   * @param functor - The functor for partial results reduction, used to reduce
   * dest content.
   * @param dest - A pointer to result memory.
   * @param dest_raw_pos - The position in previous memory block where result
   * will be stored.
   * @param set_dest_to_zero - Indicates if @c dest memory needs to be initialized
   * to zero or if its content will be taken as partial value of the reduction.
   *
   * @note functor implements <tt>void operator()(T &acc, const T &other) const;</tt>
   *
   * @see AprilMath::MatrixExt for more information about API of CBLAS and
   * user-defined reductions.
   */
  template <typename T, typename OP>
  void doNrm2(unsigned int n,
              const GPUMirroredMemoryBlock<T> *x,
              unsigned int inc,
              unsigned int shift,
              bool use_gpu,
              float zero,
              OP functor,
              GPUMirroredMemoryBlock<float> *dest,
              unsigned int dest_raw_pos,
              bool set_dest_to_zero) {
    UNUSED_VARIABLE(zero);
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
      if (set_dest_to_zero) {
        status = CUDA::wrapperCublasNrm2(handle, n, x_mem, inc,
                                         dest->getPPALForWrite() + dest_raw_pos);
      }
      else {
        GPUMirroredMemoryBlock<float> result(1);
        status = CUDA::wrapperCublasNrm2(handle, n, x_mem, inc,
                                         result.getPPALForWrite());
        float *dest_ptr = dest->getPPALForReadAndWrite() + dest_raw_pos;
        functor(*dest_ptr, *(result.getPPALForRead()));
      }
      checkCublasError(status);
    }
    else {
#endif
      x_mem = x->getPPALForRead() + shift;
      if (set_dest_to_zero) {
        float *dest_ptr = dest->getPPALForWrite() + dest_raw_pos;
        *dest_ptr = wrapperCblasNrm2(n, x_mem, inc);
      }
      else {
        float *dest_ptr = dest->getPPALForReadAndWrite() + dest_raw_pos;
        functor(*dest_ptr, wrapperCblasNrm2(n, x_mem, inc));
      }
#ifdef USE_CUDA
    }
#endif
  }

} // namespace AprilMath
#endif // NRM2_H
