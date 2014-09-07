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
#include "nrm2.h"
#include "unused_variable.h"

namespace AprilMath {

#ifdef USE_CUDA
  namespace CUDA {
    /***************************************
     ************** CUDA SECTION ***********
     ***************************************/

    template<>
    cublasStatus_t wrapperCublasNrm2<float>(cublasHandle_t &handle,
                                            unsigned int n,
                                            const float *x_mem,
                                            unsigned int x_inc,
                                            float *result) {
      return cublasSnrm2(handle, n, x_mem, x_inc, result);
    }
        
  } // namespace CUDA
#endif

  /***************************************
   ************* CBLAS SECTION ***********
   ***************************************/

  template<> float wrapperCblasNrm2<float>(unsigned int size,
                                           const float *x_mem,
                                           unsigned int x_inc) {  
    return cblas_snrm2(size, x_mem, x_inc);
  }
  
  template<> float wrapperCblasNrm2<double>(unsigned int size,
                                            const double *x_mem,
                                            unsigned int x_inc) {
    return cblas_dnrm2(size, x_mem, x_inc);
  }
  
  template<> float wrapperCblasNrm2<ComplexF>(unsigned int size,
                                              const ComplexF *x_mem,
                                              unsigned int x_inc) {
    return cblas_scnrm2(size, x_mem, x_inc);
  }

} // namespace AprilMath
