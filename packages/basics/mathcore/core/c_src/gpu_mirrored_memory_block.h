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

// Define POOL_DEBUG for verbose debug printf's
// #define POOL_DEBUG

#include <cstring>
#include <cstdio>
extern "C" {
#include <errno.h>
#include <stdint.h>
}

#include "april_assert.h"
#include "referenced.h"
#include "complex_number.h"
#include "unused_variable.h"
#include <new>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include "gpu_helper.h"
#include "error_print.h"
#include "aligned_memory.h"
#include "mmapped_data.h"

#ifndef NO_POOL
#include "hash_table.h"
#include "aux_hash_table.h"
#include "list.h"
#endif

#define PPAL_MASK  0x01 // bit 0 = 1
#define GPU_MASK   0x02 // bit 1 = 2
#define CONST_MASK 0x04 // bit 2 = 4
#define ALLOC_MASK 0x08 // bit 3 = 8
#define MMAP_MASK  0x10 // bit 4 = 16

namespace AprilMath {

  /**
   * @brief Class base for memory blocks mirrored between host (mem ppal) and
   * device (GPU).
   *
   * This base defines a generic which keeps a pool of memory pointers, stores the
   * mem ppal pointer and the device pointer, and updates the status of this
   * pointers. This class does not know pointers type, it works with generic void*
   * or char* pointers. Therefore, the size property stores the number of
   * bytes. Doing this way, it is possible to reinterpret the same memory pointer
   * as different types, allowing a complex number block to be interpreted as a
   * float block with twice number of floats (but the same size).
   */
  class GPUMirroredMemoryBlockBase : public Referenced {
  public:
#ifndef NO_POOL
    typedef AprilUtils::list<char*>                PoolListType;
    typedef AprilUtils::hash<size_t, PoolListType> PoolType;
#endif
    static bool USE_CUDA_DEFAULT;
    
  private:
    static bool use_mmap_allocation;
    
  protected:
#ifndef NO_POOL
    static size_t MAX_POOL_LIST_SIZE;
    static size_t MIN_MEMORY_TH_IN_POOL;
    static size_t pool_size;
    static PoolType *pool_lists;
    /// Auxiliary class for free of memory pool
    class PoolFreeBeforeExit {
    public:
      PoolFreeBeforeExit() { }
      ~PoolFreeBeforeExit() {
        for (PoolType::iterator it = pool_lists->begin();
             it != pool_lists->end(); ++it) {
          for (PoolListType::iterator lit = it->second.begin();
               lit != it->second.end(); ++lit) {
            AprilUtils::aligned_free(*lit);
          }
        }
        delete pool_lists;
      }
    };
    static PoolFreeBeforeExit pool_free_before_exit;
#endif
    size_t size;
    union {
      char *char_mem;
      void *mem_ppal;
      const void *const_mem_ppal;
    };
#ifdef USE_CUDA  
    mutable CUdeviceptr mem_gpu;
    bool    pinned;
#endif
    mutable unsigned char status; // bit 0 CPU, bit 1 GPU, bit 2 CONST, bit 3 ALLOCATED
    AprilUtils::MMappedDataReader *mmapped_data;
    
    void setConst() {
      status = status | CONST_MASK;
    }

    bool isConst() const {
      return status & CONST_MASK;
    }

    void setAllocated() {
      status = status | ALLOC_MASK;
    }

    bool isAllocated() const {
      return status & ALLOC_MASK;
    }

    void setMMapped() {
      status = status | MMAP_MASK;
    }

    bool isMMapped() const {
      return status & MMAP_MASK;
    }

#ifdef USE_CUDA  
    bool getUpdatedPPAL() const {
      return status & PPAL_MASK;
}
    bool getUpdatedGPU() const {
      return status & GPU_MASK;
    }
    void unsetUpdatedPPAL() {
      status = status & (~PPAL_MASK);
    }
    void unsetUpdatedGPU() {
      status = status & (~GPU_MASK);
    }
    void setUpdatedPPAL() {
      status = status | PPAL_MASK;
    }
    void setUpdatedGPU() {
      status = status | GPU_MASK;
    }
    void setUpdatedGPU() const {
      status = status | GPU_MASK;
    }
    
    void updateMemPPAL() const {
      if (!getUpdatedPPAL())
        ERROR_EXIT(128, "You need first to update the "
                   "memory in a non const pointer\n");
    }

    void updateMemPPAL() {
      if (!getUpdatedPPAL()) {
        CUresult result;
        setUpdatedPPAL();
        april_assert(mem_gpu != 0);

        if (!pinned) {
          result = cuMemcpyDtoH(mem_ppal, mem_gpu, size);
          if (result != CUDA_SUCCESS)
            ERROR_EXIT1(160, "Could not copy memory from device to host: %s\n",
                        cudaGetErrorString(cudaGetLastError()));
        }
        else {
          if (cudaMemcpyAsync(mem_ppal,
                              reinterpret_cast<void*>(mem_gpu),
                              size,
                              cudaMemcpyDeviceToHost, 0) != cudaSuccess)
            ERROR_EXIT1(162, "Could not copy memory from device to host: %s\n",
                        cudaGetErrorString(cudaGetLastError()));
          cudaThreadSynchronize();
        }
      }
    }
  
    void copyPPALtoGPU() const {
      /*
	ERROR_EXIT(128, "You need first to update the "
	"memory in a non const pointer\n");
      */
      CUresult result;
      
      if (!pinned) {
	result = cuMemcpyHtoD(mem_gpu, mem_ppal, size);
	if (result != CUDA_SUCCESS)
	  ERROR_EXIT1(162, "Could not copy memory from host to device: %s\n",
		      cudaGetErrorString(cudaGetLastError()));
      }
      else {
	cudaThreadSynchronize();
	if (cudaMemcpyAsync(reinterpret_cast<void*>(mem_gpu),
			    mem_ppal,
			    size,
			    cudaMemcpyHostToDevice, 0) != cudaSuccess)
	  ERROR_EXIT1(162, "Could not copy memory from host to device: %s\n",
		      cudaGetErrorString(cudaGetLastError()));
      }
    }
    
    void copyPPALtoGPU() {
      CUresult result;
      
      if (!pinned) {
	result = cuMemcpyHtoD(mem_gpu, mem_ppal, size);
	if (result != CUDA_SUCCESS)
	  ERROR_EXIT1(162, "Could not copy memory from host to device: %s\n",
		      cudaGetErrorString(cudaGetLastError()));
      }
      else {
	cudaThreadSynchronize();
	if (cudaMemcpyAsync(reinterpret_cast<void*>(mem_gpu),
			    mem_ppal,
			    size,
			    cudaMemcpyHostToDevice, 0) != cudaSuccess)
	  ERROR_EXIT1(162, "Could not copy memory from host to device: %s\n",
		      cudaGetErrorString(cudaGetLastError()));
      }
    }

    bool allocMemGPU() const {
      /*
	ERROR_EXIT(128, "You need first to update the "
	"memory in a non const pointer\n");
	return false;
      */
      if (mem_gpu == 0) {
	GPUHelper::initHelper();
	CUresult result;
	result = cuMemAlloc(&mem_gpu, size);
	if (result != CUDA_SUCCESS)
	  ERROR_EXIT1(161, "Could not allocate memory in device, error %d\n", result);
	return true;
      }
      return false;
    }
    
    bool allocMemGPU() {
      if (mem_gpu == 0) {
	GPUHelper::initHelper();
	CUresult result;
	result = cuMemAlloc(&mem_gpu, size);
	if (result != CUDA_SUCCESS)
	  ERROR_EXIT1(161, "Could not allocate memory in device, error %d\n", result);
      }
      return true;
    }
#endif

  public:

#ifdef USE_CUDA  
  void updateMemGPU() const {
    if (!getUpdatedGPU()) {
      /*
        ERROR_EXIT(128, "You need first to update the "
        "memory in a non const pointer\n");
      */
      allocMemGPU();
      setUpdatedGPU();
      copyPPALtoGPU();
    }
  }
  
    void updateMemGPU() {
      if (!getUpdatedGPU()) {
        allocMemGPU();
        setUpdatedGPU();
        copyPPALtoGPU();
      }
    }
#endif

    void toMMappedDataWriter(AprilUtils::MMappedDataWriter *mmapped_data) const {
#ifdef USE_CUDA
      if (!getUpdatedPPAL())
        ERROR_EXIT(128, "Impossible to update memory from a const pointer\n");
#endif
      mmapped_data->put(&size);
      mmapped_data->put(char_mem, size);
    }
  
    void toMMappedDataWriter(AprilUtils::MMappedDataWriter *mmapped_data) {
#ifdef USE_CUDA
      updateMemPPAL();
#endif
      mmapped_data->put(&size);
      mmapped_data->put(char_mem, size);
    }
  
    GPUMirroredMemoryBlockBase(AprilUtils::MMappedDataReader *mmapped_data) :
      Referenced() {
      this->size     = *(mmapped_data->get<size_t>());
      this->char_mem = mmapped_data->get<char>(this->size);
      this->mmapped_data = mmapped_data;
      IncRef(mmapped_data);
      this->status = 0;
      this->setMMapped();
#ifdef USE_CUDA
      this->unsetUpdatedGPU();
      this->setUpdatedPPAL();
      this->mem_gpu = 0;
      this->pinned  = false;
#endif
    }

    GPUMirroredMemoryBlockBase(size_t sz,
                               void *mem) : Referenced(), size(sz),
                                            mem_ppal(mem) {
      status = 0;
      mmapped_data = 0;
#ifdef USE_CUDA
      unsetUpdatedGPU();
      setUpdatedPPAL();
      mem_gpu  = 0;
      pinned   = false;
#endif
    }

    GPUMirroredMemoryBlockBase(size_t sz,
                               const void *mem) : Referenced(), size(sz),
                                                  const_mem_ppal(mem) {
      status = 0;
      mmapped_data = 0;
      setConst();
#ifdef USE_CUDA
      unsetUpdatedGPU();
      setUpdatedPPAL();
      mem_gpu  = 0;
      pinned   = false;
#endif
    }
  
    // WARNING!!! the memory zone is not initialized by default
    GPUMirroredMemoryBlockBase(size_t sz) : Referenced(),
                                            size(sz) {
      status = 0;
      mmapped_data = 0;
      setAllocated();
#ifdef USE_CUDA
      unsetUpdatedGPU();
      setUpdatedPPAL();
      mem_gpu  = 0;
      pinned   = false;
#endif
#ifndef NO_POOL
      PoolListType &l = (*pool_lists)[size];
      if (l.empty()) {
        if (!use_mmap_allocation) {
          char_mem = AprilUtils::aligned_malloc<char>(size);
#ifdef POOL_DEBUG
          printf("ALLOC %lu :: %p\n", size, char_mem);
#endif
        }
        else {
          setMMapped();
          mem_ppal = (void*)mmap(NULL, sz,
                                 PROT_READ | PROT_WRITE, MAP_ANON | MAP_SHARED,
                                 -1, 0);
          if (mem_ppal == MAP_FAILED)
            ERROR_EXIT1(128, "Impossible to open required mmap memory: %s\n",
                        strerror(errno));
        }
      }
      else {
        mem_ppal = *(l.begin());
        l.pop_front();
        pool_size -= size;
#ifdef POOL_DEBUG
        printf("POP %lu :: %p\n", size, mem_ppal);
#endif
      }
#else
      if (!use_mmap_allocation) {
        char_mem = AprilUtils::aligned_malloc<char>(size);
      }
      else {
        setMMapped();
        mem_ppal = (void*)mmap(NULL, sz,
                               PROT_READ | PROT_WRITE, MAP_ANON | MAP_SHARED,
                               -1, 0);
        if (mem_ppal == MAP_FAILED)
          ERROR_EXIT1(128, "Impossible to open required mmap memory: %s\n",
                      strerror(errno));
      }
#endif
    }
  
    virtual ~GPUMirroredMemoryBlockBase() {
      // for (unsigned int i=0; i<size; ++i) mem_ppal[i].~T();
#ifdef USE_CUDA
      if (pinned) {
        if (cudaFreeHost(mem_ppal) != cudaSuccess)
          ERROR_EXIT1(162, "Could not copy memory from host to device: %s\n",
                      cudaGetErrorString(cudaGetLastError()));
      }
      else {
        if (isAllocated()) {
          if (!isMMapped()) {
#ifndef NO_POOL
            PoolListType &l = (*pool_lists)[size];
            if (pool_size + size <= MAX_POOL_LIST_SIZE && size >= MIN_MEMORY_TH_IN_POOL) {
              pool_size += size;
              l.push_front(char_mem);
#ifdef POOL_DEBUG
              printf("PUSH %lu :: %p\n", size, char_mem);
#endif
            }
            else {
#ifdef POOL_DEBUG
              printf("FREE %lu :: %p\n", size, char_mem);
#endif
              AprilUtils::aligned_free(char_mem);
            }
#else
            AprilUtils::aligned_free(char_mem);
#endif
          }
          else munmap(mem_ppal, size);
        }
      }
      if (mem_gpu != 0) {
        CUresult result;
        result = cuMemFree(mem_gpu);
        if (result != CUDA_SUCCESS)
          ERROR_EXIT(163, "Could not free memory from device.\n");
      }
#else
      if (isAllocated()) {
        if (!isMMapped()) {
#ifndef NO_POOL
          PoolListType &l = (*pool_lists)[size];
          if (pool_size+size <= MAX_POOL_LIST_SIZE && size >= MIN_MEMORY_TH_IN_POOL) {
            pool_size += size;
            l.push_front(char_mem);
#ifdef POOL_DEBUG
            printf("PUSH %lu :: %p\n", size, char_mem);
#endif
          }
          else {
#ifdef POOL_DEBUG
            printf("FREE %lu :: %p\n", size, char_mem);
#endif
            AprilUtils::aligned_free(char_mem);
          }
#else
          AprilUtils::aligned_free(char_mem);
#endif
        }
        else munmap(mem_ppal, size);
      }
#endif
      if (isMMapped() && mmapped_data != 0) DecRef(mmapped_data);
    }

#ifdef USE_CUDA
    void pinnedMemoryPageLock() const {
      ERROR_EXIT(128, "Execute it from a non const pointer\n");
    }

    void pinnedMemoryPageLock() {
      if (isConst() || isMMapped()) {
        ERROR_EXIT(128, "Impossible to set as pinned a const or mmapped memory block\n");
      }
      if (mem_ppal) AprilUtils::aligned_free(char_mem);
      void *ptr;
      if (cudaHostAlloc(&ptr, size, 0) != cudaSuccess)
        ERROR_EXIT1(162, "Could not copy memory from host to device: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
      mem_ppal = ptr;
      pinned = true;
    }
#endif
  
    bool getCudaFlag() const {
#ifdef USE_CUDA
      return getUpdatedGPU();
#else
      return false;
#endif
    }
  
    static void setUseMMapAllocation(bool v) { use_mmap_allocation = v; }
    void forceUpdate(bool use_cuda) {
#ifdef USE_CUDA
      if (isConst())
        ERROR_EXIT(128, "Impossible to write in a const memory block\n");
      if (use_cuda)
        updateMemGPU();
      else
        updateMemPPAL();
#else
      UNUSED_VARIABLE(use_cuda);
#endif
    }

#ifndef NO_POOL
    static void changeMaxPoolSize(size_t max_pool_size) {
      MAX_POOL_LIST_SIZE = max_pool_size;
      // TODO: free pool memory if necessary
    }
#endif
  };

  ////////////////////////////////////////////////////////////////////////////

  /**
   * @brief Class template for instantiation of GPUMirroredMemoryBlockBase for different
   * data types.
   */
  template<typename T>
  class GPUMirroredMemoryBlock : public GPUMirroredMemoryBlockBase {
    
  protected:
    
    GPUMirroredMemoryBlock(AprilUtils::MMappedDataReader *mmapped_data) :
      GPUMirroredMemoryBlockBase(mmapped_data) { }
  
    T *getPointer() {
      union {
        void *void_ptr;
        T    *T_ptr;
      };
      void_ptr = mem_ppal;
      return T_ptr;
    }

    const T *getPointer() const {
      union {
        const void *void_ptr;
        const T    *T_ptr;
      };
      void_ptr = const_mem_ppal;
      return T_ptr;
    }
  
  public:
  
    static GPUMirroredMemoryBlock<T> *
    fromMMappedDataReader(AprilUtils::MMappedDataReader *mmapped_data) {
      return new GPUMirroredMemoryBlock<T>(mmapped_data);
    }
  
    /// Constructor from non-allocated memory, does not free mem pointer.
    GPUMirroredMemoryBlock(unsigned int sz, T *mem) :
      GPUMirroredMemoryBlockBase(sz*sizeof(T), mem) { }
  
    /// Constructor from non-allocated const memory, does not free mem pointer,
    /// and writes are not allowed
    GPUMirroredMemoryBlock(unsigned int sz, const T *mem) :
      GPUMirroredMemoryBlockBase(sz*sizeof(T), mem) { }
  
    /// Constructor with allocation of memory, the memory pointer will be properly
    /// allocated and freed at destruction.
    GPUMirroredMemoryBlock(unsigned int sz) :
      GPUMirroredMemoryBlockBase(sz*sizeof(T)) { }
  
    virtual ~GPUMirroredMemoryBlock() { }
  
    GPUMirroredMemoryBlock<T> *clone() const {
      GPUMirroredMemoryBlock<T> *result = new GPUMirroredMemoryBlock(getSize());
      result->copyFromBlock(0, this, 0, getSize());
      return result;
    }
  
    void copyFromBlock(size_t where,
                       const GPUMirroredMemoryBlock<T> *other, size_t from,
                       size_t sz) {
#ifdef USE_CUDA
      if (other->getUpdatedGPU()) {
        const T *other_ptr = other->getGPUForRead() + from;
        T *this_ptr        = this->getGPUForWrite() + where;
        cudaMemcpy(this_ptr, other_ptr, sz * sizeof(T),
                   cudaMemcpyDeviceToDevice);
      }
      else {
#endif
        const T *other_ptr = other->getPPALForRead() + from;
        T *this_ptr        = this->getPPALForWrite() + where;
        memcpy(this_ptr, other_ptr, sz * sizeof(T));
#ifdef USE_CUDA
      }
#endif
    }

    void putValue(size_t where, const T &value) {
#ifdef USE_CUDA
      if (this->getUpdatedGPU()) {
        cudaMemcpy(this->getGPUForWrite() + where,
                   &value,
                   sizeof(T), cudaMemcpyHostToDevice);
      }
      else {
#endif
        T *ptr = this->getPPALForWrite();
        ptr[where] = value;
#ifdef USE_CUDA
      }
#endif
    }

    void getValue(size_t from, T &result) const {
#ifdef USE_CUDA
      if (this->getUpdatedGPU()) {
        cudaMemcpy(&result,
                   this->getGPUForRead() + from,
                   sizeof(T), cudaMemcpyDeviceToHost);
      }
      else {
#endif
        const T *ptr = this->getPPALForRead();
        result = ptr[from];
#ifdef USE_CUDA
      }
#endif
    }

    unsigned int getSize() const {
      return size/sizeof(T);
    }
  
    const T *getPPALForRead() const {
#ifdef USE_CUDA
      if (!getUpdatedPPAL())
        ERROR_EXIT(128, "Update the memory from a non const pointer\n");
#endif
      return getPointer();
    }

    const T *getPPALForRead() {
#ifdef USE_CUDA
      updateMemPPAL();
#endif
      return getPointer();
    }

#ifdef USE_CUDA
    const T *getGPUForRead() const {
      /*
	if (!getUpdatedGPU())
	ERROR_EXIT(128, "Update the memory from a non const pointer\n");
      */
      updateMemGPU();
      return reinterpret_cast<T*>(mem_gpu);
    }
    
    const T *getGPUForRead() {
      updateMemGPU();
      return reinterpret_cast<T*>(mem_gpu);
    }
#endif

    T *getPPALForWrite() {
      if (isConst())
        ERROR_EXIT(128, "Impossible to write in a const memory block\n");
#ifdef USE_CUDA
      setUpdatedPPAL();
      unsetUpdatedGPU();
#endif
      return getPointer();
    }
  
#ifdef USE_CUDA
    T *getGPUForWrite() {
      if (isConst())
        ERROR_EXIT(128, "Impossible to write in a const memory block\n");
      if (allocMemGPU()) copyPPALtoGPU();
      setUpdatedGPU();
      unsetUpdatedPPAL();
      return reinterpret_cast<T*>(mem_gpu);
    }
#endif
  
    T *getPPALForReadAndWrite() {
      if (isConst())
        ERROR_EXIT(128, "Impossible to write in a const memory block\n");
#ifdef USE_CUDA
      updateMemPPAL();
      unsetUpdatedGPU();
#endif
      return getPointer();
    }

#ifdef USE_CUDA
    T *getGPUForReadAndWrite() {
      if (isConst())
        ERROR_EXIT(128, "Impossible to write in a const memory block\n");
      updateMemGPU();
      unsetUpdatedPPAL();
      return reinterpret_cast<T*>(mem_gpu);
    }
#endif

    T &get(unsigned int pos) {
#ifdef USE_CUDA
      updateMemPPAL();
      unsetUpdatedGPU();
#endif
      return getPointer()[pos];
    }

    T &operator[](unsigned int pos) {
      return get(pos);
    }

    const T &get(unsigned int pos) const {
#ifdef USE_CUDA
      updateMemPPAL();
#endif
      return getPointer()[pos];
    }

    const T &operator[](unsigned int pos) const {
      return get(pos);
    }
  
    template<typename O>
    GPUMirroredMemoryBlock<O> *reinterpretAs() {
      union {
        GPUMirroredMemoryBlock<O> *other;
        GPUMirroredMemoryBlock<T> *ptr;
      };
      ptr = this;
      return other;
    }
  };

  // typedef for referring to float memory blocks
  typedef GPUMirroredMemoryBlock<float>    FloatGPUMirroredMemoryBlock;
  typedef GPUMirroredMemoryBlock<double>   DoubleGPUMirroredMemoryBlock;
  typedef GPUMirroredMemoryBlock<int32_t>  Int32GPUMirroredMemoryBlock;
  typedef GPUMirroredMemoryBlock<ComplexF> ComplexFGPUMirroredMemoryBlock;

} // namespace AprilMath

#endif // GPU_MIRRORED_MEMORY_BLOCK_H
