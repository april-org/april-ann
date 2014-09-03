/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
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
#ifndef MATHCORE_H
#define MATHCORE_H
#include <cstdio>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "cublas_error.h"
#include "cusparse_error.h"
#endif

#include "cblas_headers.h"
#include "lapack_headers.h"
#include "cublas_error.h"
#include "cusparse_error.h"
#include "cuda_utils.h"
#include "gpu_helper.h"
#include "gpu_mirrored_memory_block.h"

namespace AprilMath {
  
  //////////////////////////////////////////////////////////////////////

  int doSearchCSCSparseIndexOf(const Int32GPUMirroredMemoryBlock *indices,
                               const Int32GPUMirroredMemoryBlock *first_index,
                               const int c1, const int c2, bool use_gpu);

  int doSearchCSRSparseIndexOf(const Int32GPUMirroredMemoryBlock *indices,
                               const Int32GPUMirroredMemoryBlock *first_index,
                               const int c1, const int c2, bool use_gpu);

  int doSearchCSCSparseIndexOfFirst(const Int32GPUMirroredMemoryBlock *indices,
                                    const Int32GPUMirroredMemoryBlock *first_index,
                                    const int c1, const int c2, bool use_gpu);

  int doSearchCSRSparseIndexOfFirst(const Int32GPUMirroredMemoryBlock *indices,
                                    const Int32GPUMirroredMemoryBlock *first_index,
                                    const int c1, const int c2, bool use_gpu);

}

#endif // MATHCORE_H
