/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2015, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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
#ifndef CUDA_HEADERS_H
#define CUDA_HEADERS_H

// AUXILIAR INLINE FUNCTIONS //
#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#define MAX_CUDA_REDUCE_THREAD_SIZE 1024
#define MIN_CUDA_REDUCE_THREAD_SIZE    8
#define MAX_CUDA_REDUCE_NUM_THREADS 1024 // must be multiple of 2 (wrap size=32)

#define APRIL_CUDA_EXPORT __host__ __device__
#define APRIL_CUDA_ERROR_EXIT(code,msg) aprilCudaErrorExit((code),(msg))

// FIXME: implement properly this feature to control errors in CUDA kernels and
// CPU equivalent code.
static APRIL_CUDA_EXPORT void aprilCudaErrorExit(int code, const char *msg) {
  UNUSED_VARIABLE(code);
  UNUSED_VARIABLE(msg);
  // do nothing
}

// static void aprilCudaErrorExit(int code, const char *msg) {
//   ERROR_EXIT(code,msg);
// }

#else

#define APRIL_CUDA_EXPORT
#define APRIL_CUDA_ERROR_EXIT(code,msg) ERROR_EXIT(code,msg)

#endif

#endif // CUDA_UTILS_H
