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
  const char *GPUMirroredMemoryBlock<float>::ctorName() const {
    return "mathcore.block.float.deserialize";
  }
  template<>
  const char *GPUMirroredMemoryBlock<double>::ctorName() const {
    return "mathcore.block.double.deserialize";
  }
  template<>
  const char *GPUMirroredMemoryBlock<ComplexF>::ctorName() const {
    return "mathcore.block.complex.deserialize";
  }
  template<>
  const char *GPUMirroredMemoryBlock<int32_t>::ctorName() const {
    return "mathcore.block.int32.deserialize";
  }

  template<>
  int GPUMirroredMemoryBlock<float>::exportParamsToLua(lua_State *L) {

  }
  template<>
  int GPUMirroredMemoryBlock<double>::exportParamsToLua(lua_State *L) {
    AprilIO::OutputLuaStringStream destination(L);
    const int columns = 16;
    char b[10];
    const double *ptr = getPPALForRead();
    unsigned int i;
    for (i=0; i<getSize(); ++i) {
      AprilUtils::binarizer::code_double(ptr[i], b);
      destination.put(b, 10);
      if ((i+1) % columns == 0) destination.printf("\n");
    }
    if ((i % columns) != 0) destination.printf("\n");
    destination.push(L);
    return 1;
  }
  template<>
  int GPUMirroredMemoryBlock<ComplexF>::exportParamsToLua(lua_State *L) {
    AprilIO::OutputLuaStringStream destination(L);
    const int columns = 16;
    char b[5];
    const ComplexF *ptr = getPPALForRead();
    unsigned int i;
    for (i=0; i<getSize(); ++i) {
      AprilUtils::binarizer::code_float(ptr[i].real(), b);
      destination.put(b, 5);
      AprilUtils::binarizer::code_float(ptr[i].img(), b);
      destination.put(b, 5);
      if ((i+1) % columns == 0) destination.printf("\n");
    }
    if ((i % columns) != 0) destination.printf("\n");
    destination.push(L);
    return 1;
  }
  template<>
  int GPUMirroredMemoryBlock<int32_t>::exportParamsToLua(lua_State *L) {
    AprilIO::OutputLuaStringStream destination(L);
    const int columns = 16;
    char b[5];
    const int32_t *ptr = getPPALForRead();
    unsigned int i;
    for (i=0; i<getSize(); ++i) {
      AprilUtils::binarizer::code_int32(ptr[i], b);
      destination.put(b, 5);
      if ((i+1) % columns == 0) destination.printf("\n");
    }
    if ((i % columns) != 0) destination.printf("\n");
    destination.push(L);
    return 1;
  }
  
  template class GPUMirroredMemoryBlock<float>;
  template class GPUMirroredMemoryBlock<double>;
  template class GPUMirroredMemoryBlock<int32_t>;
  template class GPUMirroredMemoryBlock<ComplexF>;
} // namespace AprilMath
