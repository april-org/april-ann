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
#include "dot.h"
#include "unused_variable.h"

namespace AprilMath {

#ifdef USE_CUDA
  namespace CUDA {
    
    /***************************************
     ************** CUDA SECTION ***********
     ***************************************/

    template<>
    cublasStatus_t wrapperCublasDot<float>(cublasHandle_t &handle,
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
    
    template<>
    cublasStatus_t wrapperCublasDot<double>(cublasHandle_t &handle,
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
    
  } // namespace CUDA
#endif

  /***************************************
   ************* CBLAS SECTION ***********
   ***************************************/

  template<>
  float wrapperCblasDot<float>(unsigned int size,
                               const float *x_mem, unsigned int x_inc,
                               const float *y_mem, unsigned int y_inc) {
    return cblas_sdot(size,
                      x_mem, x_inc,
                      y_mem, y_inc);
  }

  template<>
  double wrapperCblasDot<double>(unsigned int size,
                                 const double *x_mem, unsigned int x_inc,
                                 const double *y_mem, unsigned int y_inc) {
    return cblas_ddot(size,
                      x_mem, x_inc,
                      y_mem, y_inc);
  }

  template<>
  ComplexF wrapperCblasDot<ComplexF>(unsigned int size,
                                     const ComplexF *x_mem, unsigned int x_inc,
                                     const ComplexF *y_mem, unsigned int y_inc) {
    ComplexF ret;
    cblas_zdotu_sub(size,
                    x_mem, x_inc,
                    y_mem, y_inc,
                    &ret);
    return ret;
  }
  
} // namespace AprilMath
