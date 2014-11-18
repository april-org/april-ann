/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
 * Copyright 2012, Salvador España-Boquera
 *
 * The APRIL-ANN toolkit is free software; you can redistribute it and/or modify it
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
#ifndef TRANSPOSE_H
#define TRANSPOSE_H
#include "mathcore.h"

#ifdef USE_CUDA
namespace CUDA {
  cublasStatus_t wrapperTranspose(cublasHandle_t &handle,
                                  int m, int n,
                                  const float *orig, int orig_inc,
                                  float *dest, int dest_inc);

  cublasStatus_t wrapperTranspose(cublasHandle_t &handle,
                                  int m, int n,
                                  const double *orig, int orig_inc,
                                  double *dest, int dest_inc);

  cublasStatus_t wrapperTranspose(cublasHandle_t &handle,
                                  int m, int n,
                                  const ComplexF *orig, int orig_inc,
                                  ComplexF *dest, int dest_inc);
}
#endif // USE_CUDA

#endif // TRANSPOSE_H
