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
#ifndef GPU_MIRRORED_MEMORY_BLOCK_H
#define GPU_MIRRORED_MEMORY_BLOCK_H

// Define NO_POOL to avoid the use of a pool of pointers
// #define NO_POOL

#include <cassert>
#include "referenced.h"
#include <new>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include "gpu_helper.h"
#include "error_print.h"
#include "aligned_memory.h"

#ifndef NO_POOL
#include "hash_table.h"
#include "aux_hash_table.h"
#include "list.h"
#endif

template<typename T>
class GPUMirroredMemoryBlock : public Referenced {
#ifndef NO_POOL
  const static unsigned int MAX_POOL_LIST_SIZE = 100;
  static april_utils::hash<unsigned int,april_utils::list<T*> > pool_lists;
  // FIXME: This static class is not working... therefore the memory allocated
  // by pool pointers is never freed. 
  class PoolFreeBeforeExit {
  public:
    PoolFreeBeforeExit() { }
    ~PoolFreeBeforeExit() {
      for (typename april_utils::hash<unsigned int,april_utils::list<T*> >::iterator it = pool_lists.begin();
	   it != pool_lists.end(); ++it) {
	for (typename april_utils::list<T*>::iterator lit = it->second.begin();
	     lit != it->second.end(); ++lit)
	  delete *lit;
      }
    }
  };
  const static PoolFreeBeforeExit pool_free_before_exit();
#endif
  unsigned int size;
  mutable T      *mem_ppal;
#ifdef USE_CUDA  
  mutable CUdeviceptr mem_gpu;
  mutable char        updated; // bit 0 CPU, bit 1 GPU
  bool    pinned;
#endif

#ifdef USE_CUDA  
  bool getUpdatedPPAL() const {
    return updated & 0x01;    // 0000 0001
  }
  bool getUpdatedGPU() const {
    return updated & 0x02;    // 0000 0010
  }
  void unsetUpdatedPPAL() {
    updated = updated & 0xFE; // 1111 1110
  }
  void unsetUpdatedGPU() {
    updated = updated & 0xFD; // 1111 1101
  }
  void setUpdatedPPAL() {
    updated = updated | 0x01; // 0000 0001
  }
  void setUpdatedGPU() {
    updated = updated | 0x02; // 0000 0010
  }
  
  void updateMemPPAL() {
    if (!getUpdatedPPAL()) {
      CUresult result;
      setUpdatedPPAL();
      assert(mem_gpu != 0);

      if (!pinned) {
	result = cuMemcpyDtoH(mem_ppal, mem_gpu, sizeof(T)*size);
	if (result != CUDA_SUCCESS)
	  ERROR_EXIT1(160, "Could not copy memory from device to host: %s\n",
		      cudaGetErrorString(cudaGetLastError()));
      }
      else {
	if (cudaMemcpyAsync(reinterpret_cast<void*>(mem_ppal),
			    reinterpret_cast<void*>(mem_gpu),
			    sizeof(T)*size,
			    cudaMemcpyDeviceToHost, 0) != cudaSuccess)
	  ERROR_EXIT1(162, "Could not copy memory from device to host: %s\n",
		      cudaGetErrorString(cudaGetLastError()));
	cudaThreadSynchronize();
      }
    }
  }

  void copyPPALtoGPU() {
    CUresult result;

    if (!pinned) {
      result = cuMemcpyHtoD(mem_gpu, mem_ppal, sizeof(T)*size);
      if (result != CUDA_SUCCESS)
	ERROR_EXIT1(162, "Could not copy memory from host to device: %s\n",
		    cudaGetErrorString(cudaGetLastError()));
    }
    else {
      cudaThreadSynchronize();
      if (cudaMemcpyAsync(reinterpret_cast<void*>(mem_gpu),
			  reinterpret_cast<void*>(mem_ppal),
			  sizeof(T)*size,
			  cudaMemcpyHostToDevice, 0) != cudaSuccess)
	ERROR_EXIT1(162, "Could not copy memory from host to device: %s\n",
		    cudaGetErrorString(cudaGetLastError()));
    }
  }

  bool allocMemGPU() {
    if (mem_gpu == 0) {
      CUresult result;
      result = cuMemAlloc(&mem_gpu, sizeof(T)*size);
      if (result != CUDA_SUCCESS)
	ERROR_EXIT(161, "Could not allocate memory in device.\n");
      return true;
    }
    return false;
  }
  
  void updateMemGPU() {
    if (!getUpdatedGPU()) {
      allocMemGPU();
      setUpdatedGPU();
      copyPPALtoGPU();
    }
  }

#endif
  
public:

  // WARNING!!! the memory zone is not initialized by default
  GPUMirroredMemoryBlock(unsigned int sz,
			 bool initialize=false) : Referenced(), size(sz) {
#ifdef USE_CUDA
    updated  = 0;
    unsetUpdatedGPU();
    setUpdatedPPAL();
    mem_gpu  = 0;
    pinned   = false;
#endif
#ifndef NO_POOL
    april_utils::list<T*> &l = pool_lists[size];
    if (l.empty())
      mem_ppal = aligned_malloc<T>(size);
    else {
      mem_ppal = *(l.begin());
      l.pop_front();
    }
#else
    mem_ppal = aligned_malloc<T>(size);
#endif
    if (initialize) for (unsigned int i=0; i<size; ++i) new(mem_ppal+i) T();
  }
  ~GPUMirroredMemoryBlock() {
    // for (unsigned int i=0; i<size; ++i) mem_ppal[i].~T();
#ifdef USE_CUDA
    if (pinned) {
      if (cudaFreeHost(reinterpret_cast<void*>(mem_ppal)) != cudaSuccess)
	ERROR_EXIT1(162, "Could not copy memory from host to device: %s\n",
		    cudaGetErrorString(cudaGetLastError()));
    }
    else {
#ifndef NO_POOL
      april_utils::list<T*> &l = pool_lists[size];
      if (l.size() < MAX_POOL_LIST_SIZE) l.push_front(mem_ppal);
      else aligned_free(mem_ppal);
#else
      aligned_free(mem_ppal);
#endif
    }
    if (mem_gpu != 0) {
      CUresult result;
      result = cuMemFree(mem_gpu);
      if (result != CUDA_SUCCESS)
        ERROR_EXIT(163, "Could not free memory from device.\n");
    }
#else
#ifndef NO_POOL
    april_utils::list<T*> &l = pool_lists[size];
    if (l.size() < MAX_POOL_LIST_SIZE) l.push_front(mem_ppal);
    else aligned_free(mem_ppal);
#else
    aligned_free(mem_ppal);
#endif
#endif
  }

  unsigned int getSize() const { return size; }
  
#ifdef USE_CUDA
  void pinnedMemoryPageLock() {
    if (mem_ppal) aligned_free(mem_ppal);
    void *ptr;
    if (cudaHostAlloc(&ptr, sizeof(T)*size, 0) != cudaSuccess)
      ERROR_EXIT1(162, "Could not copy memory from host to device: %s\n",
		  cudaGetErrorString(cudaGetLastError()));
    mem_ppal = reinterpret_cast<T*>(ptr);
    pinned = true;
  }
#endif
  
  const T *getPPALForRead() const {
#ifdef USE_CUDA
    updateMemPPAL();
#endif
    return mem_ppal;
  }

#ifdef USE_CUDA
  const T *getGPUForRead() const {
    updateMemGPU();
    return reinterpret_cast<T*>(mem_gpu);
  }
#endif

  T *getPPALForWrite() {
#ifdef USE_CUDA
    setUpdatedPPAL();
    unsetUpdatedGPU();
#endif
    return mem_ppal;
  }

#ifdef USE_CUDA
  T *getGPUForWrite() {
    if (allocMemGPU()) copyPPALtoGPU();
    setUpdatedGPU();
    unsetUpdatedPPAL();
    return reinterpret_cast<T*>(mem_gpu);
  }
#endif
  
  T *getPPALForReadAndWrite() {
#ifdef USE_CUDA
    updateMemPPAL();
    unsetUpdatedGPU();
#endif
    return mem_ppal;
  }

#ifdef USE_CUDA
  T *getGPUForReadAndWrite() {
    updateMemGPU();
    unsetUpdatedPPAL();
    return reinterpret_cast<T*>(mem_gpu);
  }
#endif

  bool getCudaFlag() {
#ifdef USE_CUDA
    return getUpdatedGPU();
#else
    return false;
#endif
  }

  T &get(unsigned int pos) {
#ifdef USE_CUDA
    updateMemPPAL();
    unsetUpdatedGPU();
#endif
    return mem_ppal[pos];
  }

  T &operator[](unsigned int pos) {
    return get(pos);
  }

  const T &get(unsigned int pos) const {
#ifdef USE_CUDA
    updateMemPPAL();
#endif
    return mem_ppal[pos];
  }

  const T &operator[](unsigned int pos) const {
    return get(pos);
  }
};

// typedef for referring to float memory blocks
typedef GPUMirroredMemoryBlock<float> FloatGPUMirroredMemoryBlock;

#ifndef NO_POOL
template<typename T>
april_utils::hash<unsigned int,april_utils::list<T*> >
GPUMirroredMemoryBlock<T>::pool_lists(1024);
#endif

#endif // GPU_MIRRORED_MEMORY_BLOCK_H
