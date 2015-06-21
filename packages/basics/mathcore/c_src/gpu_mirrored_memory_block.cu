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
#include "gpu_mirrored_memory_block.h"

namespace AprilMath {

  bool   GPUMirroredMemoryBlockBase::use_mmap_allocation = false;
  bool   GPUMirroredMemoryBlockBase::USE_CUDA_DEFAULT = false;

#ifndef NO_POOL
  size_t GPUMirroredMemoryBlockBase::MAX_POOL_LIST_SIZE = 200*1024*1024; // 200 Megabytes
  size_t GPUMirroredMemoryBlockBase::MIN_MEMORY_TH_IN_POOL = 20; // 20 bytes
  GPUMirroredMemoryBlockBase::PoolFreeBeforeExit GPUMirroredMemoryBlockBase::pool_free_before_exit;
  size_t GPUMirroredMemoryBlockBase::pool_size = 0;
  GPUMirroredMemoryBlockBase::PoolType *GPUMirroredMemoryBlockBase::pool_lists =
    new GPUMirroredMemoryBlockBase::PoolType(1024);
#endif

  template<>
  const char *GPUMirroredMemoryBlock<char>::luaCtorName() const {
    return "mathcore.block.char.read";
  }
  template<>
  const char *GPUMirroredMemoryBlock<float>::luaCtorName() const {
    return "mathcore.block.float.read";
  }
  template<>
  const char *GPUMirroredMemoryBlock<double>::luaCtorName() const {
    return "mathcore.block.double.read";
  }
  template<>
  const char *GPUMirroredMemoryBlock<int32_t>::luaCtorName() const {
    return "mathcore.block.int32.read";
  }
  template<>
  const char *GPUMirroredMemoryBlock<ComplexF>::luaCtorName() const {
    return "mathcore.block.complex.read";
  }
  template<>
  const char *GPUMirroredMemoryBlock<bool>::luaCtorName() const {
    return "mathcore.block.bool.read";
  }
  
  template class GPUMirroredMemoryBlock<float>;
  template class GPUMirroredMemoryBlock<double>;
  template class GPUMirroredMemoryBlock<int32_t>;
  template class GPUMirroredMemoryBlock<ComplexF>;
  template class GPUMirroredMemoryBlock<bool>;
} // namespace AprilMath
