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
#ifndef _CUBLAS_HANDLER_H_
#define _CUBLAS_HANDLER_H_

#ifdef USE_CUDA
#include "error_print.h"
#include "cblas_headers.h"
#include "vector.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

class GPUHelper {
  static cublasHandle_t handler;
  static bool initialized;
  static cudaDeviceProp properties;
  static CUdevice  device;
  static CUcontext context;
  static april_utils::vector<CUstream> streams;
  static unsigned int current_stream;
 public:
  static void initHelper() {
    if (!initialized) {
      cublasStatus_t state;	
      state = cublasCreate(&handler);
      if (state != CUBLAS_STATUS_SUCCESS)
	ERROR_EXIT(150, "Cublas couldn't be initialized\n");
      else
	initialized = true;
      
      if (cuDeviceGet (&device, 0) != CUDA_SUCCESS)
	ERROR_EXIT(152, "Failed to get device\n");
      if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess)
	ERROR_EXIT(153, "Failed to get properties\n");

      streams.push_back(0);
      current_stream = 0;
      fprintf(stderr,
	      "# Initialized CUDA and CUBLAS for GPU capabilitites of version %d.%d\n",
	      properties.major, properties.minor);
    }
  }

  static void destroyHandler() {
    destroyStreams();
    if (initialized != true)
      ERROR_EXIT(151, "Destroying CUBLAS handler, that it's not initialized\n");
    // Need to free all the allocated memory in GPU
    cublasDestroy(handler);
    initialized = false;
  } 

  static cublasHandle_t &getHandler() {
    initHelper();
    return handler;
  }
  
  static CUdevice &getCUdevice() {
    initHelper();
    return device;
  }
  
  static unsigned int getMaxThreadsPerBlock() {
    initHelper();
    return properties.maxThreadsPerBlock;
  }
  
  static void createNStreams(unsigned int n) {
    initHelper();
    cuStreamSynchronize(0);
    for (unsigned int i=0; i<n; ++i) {
      CUstream stream;
      cuStreamCreate(&stream, 0);
      streams.push_back(stream);
    }
  }
  
  static void destroyStreams() {
    initHelper();
    for (unsigned int i=streams.size(); i>1; --i) {
      cuStreamSynchronize(streams[i-1]);
      cuStreamDestroy(streams[i-1]);
      streams.pop_back();
    }
    current_stream = 0;
    assert(streams.size() == 1);
  }
  
  static void setCurrentStream(unsigned int i) {
    assert(i < streams.size());
    current_stream = i;
  }
  
  static CUstream getCurrentStream() {
    initHelper();
    return streams[current_stream];
  }
};
#endif

#endif // _CUBLAS_HANDLER_H_
