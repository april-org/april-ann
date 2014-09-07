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
#ifndef DOT_H
#define DOT_H
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
    cublasStatus_t wrapperCublasDot(cublasHandle_t &handle,
                                    unsigned int size,
                                    const T *x_mem,
                                    unsigned int x_inc,
                                    const T *y_mem,
                                    unsigned int y_inc,
                                    T *ret) {
      UNUSED_VARIABLE(handle);
      UNUSED_VARIABLE(size);
      UNUSED_VARIABLE(x_mem);
      UNUSED_VARIABLE(x_inc);
      UNUSED_VARIABLE(y_mem);
      UNUSED_VARIABLE(y_inc);
      UNUSED_VARIABLE(ret);
      ERROR_EXIT(128, "NOT IMPLEMENTED\n");
      return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    template<>
    cublasStatus_t wrapperCublasDot<float>(cublasHandle_t &handle,
                                           unsigned int size,
                                           const float *x_mem,
                                           unsigned int x_inc,
                                           const float *y_mem,
                                           unsigned int y_inc,
                                           float *ret);

    template<>
    cublasStatus_t wrapperCublasDot<double>(cublasHandle_t &handle,
                                            unsigned int size,
                                            const double *x_mem,
                                            unsigned int x_inc,
                                            const double *y_mem,
                                            unsigned int y_inc,
                                            double *ret);

    template<>
    cublasStatus_t wrapperCublasDot<ComplexF>(cublasHandle_t &handle,
                                              unsigned int size,
                                              const ComplexF *x_mem,
                                              unsigned int x_inc,
                                              const ComplexF *y_mem,
                                              unsigned int y_inc,
                                              ComplexF *ret);
    
  } // namespace CUDA
#endif

  /***************************************
   ************* CBLAS SECTION ***********
   ***************************************/

  template<typename T>
  T wrapperCblasDot(unsigned int size,
                    const T *x_mem, unsigned int x_inc,
                    const T *y_mem, unsigned int y_inc) {
    UNUSED_VARIABLE(size);
    UNUSED_VARIABLE(x_mem);
    UNUSED_VARIABLE(x_inc);
    UNUSED_VARIABLE(y_mem);
    UNUSED_VARIABLE(y_inc);
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
    return T(0.0f);
  }

  template<>
  float wrapperCblasDot<float>(unsigned int size,
                               const float *x_mem, unsigned int x_inc,
                               const float *y_mem, unsigned int y_inc);

  template<>
  double wrapperCblasDot<double>(unsigned int size,
                                 const double *x_mem, unsigned int x_inc,
                                 const double *y_mem, unsigned int y_inc);

  template<>
  ComplexF wrapperCblasDot<ComplexF>(unsigned int size,
                                     const ComplexF *x_mem, unsigned int x_inc,
                                     const ComplexF *y_mem, unsigned int y_inc);
  
  /***************************************
   *********** TEMPLATE SECTION **********
   ***************************************/


  /**
   * @brief Template wrapper for dot product operation using CBLAS interface.
   *
   * @tparam T - The data type. In APRIl-ANN it could be float, double,
   * ComplexF.
   *
   * @tparam OP - Functor type.
   *
   * @param size - The number of elements in both vectors.
   * @param x - The GPUMirroredMemoryBlock pointer with x vector.
   * @param x_inc - The stride between consecutive elements in x vector.
   * @param x_shift - The position of the first valid element in x pointer.
   * @param[in,out] y - The GPUMirroredMemoryBlock pointer with y vector.
   * @param y_inc - The stride between consecutive elements in y vector.
   * @param y_shift - The position of the first valid element in y pointer.
   * @param use_gpu - Indicates if use GPU or not for the computation.
   * @param zero - Zero value (unused, needed for API compliant).
   * @param functor - A functor reduction to write the result into dest pointer.
   * @param dest - Destination of the computation.
   * @param dest_raw_pos - Position in dest where to store the result.
   */
  template <typename T, typename OP>
  void doDot(unsigned int size,
             const GPUMirroredMemoryBlock<T> *x,
             unsigned int x_inc,
             unsigned int x_shift,
             const GPUMirroredMemoryBlock<T> *y,
             unsigned int y_inc,
             unsigned int y_shift,
             bool use_gpu,
             T zero,
             OP functor,
             GPUMirroredMemoryBlock<T> *dest,
             unsigned int dest_raw_pos) {
    UNUSED_VARIABLE(zero);
    GPUMirroredMemoryBlock<T> result(1);
    const T *x_mem;
    const T *y_mem;
#ifndef USE_CUDA
    UNUSED_VARIABLE(use_gpu);
#endif
#ifdef USE_CUDA
    if (use_gpu) {
      cublasStatus_t status;
      cublasHandle_t handle = CUDA::GPUHelper::getHandler();
      x_mem = x->getGPUForRead() + x_shift;
      y_mem = y->getGPUForRead() + y_shift;
      
      status = cublasSetStream(handle, CUDA::GPUHelper::getCurrentStream());
      checkCublasError(status);
      status = CUDA::wrapperCublasDot(handle, size, x_mem, x_inc,
                                      y_mem, y_inc,
                                      result.getGPUForWrite());
      checkCublasError(status);
    }
    else {
#endif
      x_mem = x->getPPALForRead() + x_shift;
      y_mem = y->getPPALForRead() + y_shift;
      result.putValue(0u, wrapperCblasDot(size, x_mem, x_inc, y_mem, y_inc));
#ifdef USE_CUDA
    }
#endif
    T aux;
    dest->getValue(dest_raw_pos, aux);
    functor(aux, *(result.getPPALForRead()));
    dest->putValue(dest_raw_pos, aux);
  }

  /**
   * @brief Template wrapper for sparse dot product operation using CBLAS
   * interface.
   *
   * It computes dot(x,y) where x is a sparse vector and y is a dense vector.
   *
   * @tparam T - The data type. In APRIl-ANN it could be float, double,
   * ComplexF.
   *
   * @param NNZ - The number of non-zero elements in x vector.
   * @param x_values - The GPUMirroredMemoryBlock pointer with x values.
   * @param x_indices - The Int32GPUMirroredMemoryBlock pointer with x indices.
   * @param[in,out] y - The GPUMirroredMemoryBlock pointer with y vector.
   * @param y_inc - The stride between consecutive elements in y vector.
   * @param y_shift - The position of the first valid element in y pointer.
   * @param use_gpu - Indicates if use GPU or not for the computation.
   */
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
    if (use_gpu) {
      ERROR_PRINT("CUDA sparse DOT not implemented\n");
    }
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
  
} // namespace AprilMath
#endif // DOT_H
