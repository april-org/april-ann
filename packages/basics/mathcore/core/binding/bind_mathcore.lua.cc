/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
//BIND_HEADER_H
#include "gpu_mirrored_memory_block.h"

using namespace AprilMath;
//BIND_END

//BIND_HEADER_C
#include "luabindutil.h"
#include "luabindmacros.h"
#include "error_print.h"

namespace AprilMath {
  
#define FUNCTION_NAME "Constructor"
  template<typename T>
  void GPUMirroredMemoryBlockConstructor(lua_State *L,
                                         GPUMirroredMemoryBlock<T> *&obj) {
    unsigned int N;
    if (lua_istable(L,1)) N = lua_rawlen(L,1);
    else LUABIND_GET_PARAMETER(1,uint,N);
    obj = new GPUMirroredMemoryBlock<T>(N);
    if (lua_istable(L,1)) {
      T *ptr = obj->getPPALForWrite();
      for (unsigned int i=0; i<N; ++i) {
        lua_pushinteger(L, i+1);
        lua_gettable(L, -2);
        LUABIND_GET_PARAMETER(-1,number,ptr[i]);
        lua_pop(L,1);
      }
    }
  }
#undef FUNCTION_NAME

#define FUNCTION_NAME "set"
  template<typename T>
  void GPUMirroredMemoryBlockSet(lua_State *L,
                                 GPUMirroredMemoryBlock<T> *obj) {
    T value;
    unsigned int i;
    LUABIND_GET_PARAMETER(1,uint,i);
    LUABIND_GET_PARAMETER(2,number,value);
    if (i<=0 || i> obj->getSize()) ERROR_EXIT(128, "Index out of bounds\n");
    T *ptr = obj->getPPALForWrite();
    ptr[i-1] = value;
  }
#undef FUNCTION_NAME

#define FUNCTION_NAME "get"
  template<typename T>
  T GPUMirroredMemoryBlockGet(lua_State *L,
                              GPUMirroredMemoryBlock<T> *obj) {
    unsigned int i;
    LUABIND_GET_PARAMETER(1,uint,i);
    if (i<=0 || i> obj->getSize()) ERROR_EXIT(128, "Index out of bounds\n");
    const T *ptr = obj->getPPALForRead();
    return ptr[i-1];
  }
#undef FUNCTION_NAME

} // namespace AprilMath

//BIND_END

//BIND_FUNCTION mathcore.set_mmap_allocation
{
  bool v;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, bool, v);
  GPUMirroredMemoryBlockBase::setUseMMapAllocation(v);
}
//BIND_END

//BIND_FUNCTION mathcore.set_max_pool_size
{
  int max_pool_size;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, int, max_pool_size);
#ifndef NO_POOL
  GPUMirroredMemoryBlockBase::
    changeMaxPoolSize(static_cast<size_t>(max_pool_size));
#endif
}
//BIND_END

//BIND_FUNCTION mathcore.set_use_cuda_default
{
  bool v;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, bool, v);
  GPUMirroredMemoryBlockBase::USE_CUDA_DEFAULT = v;
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME FloatGPUMirroredMemoryBlock mathcore.block.float
//BIND_CPP_CLASS FloatGPUMirroredMemoryBlock

//BIND_CONSTRUCTOR FloatGPUMirroredMemoryBlock
{
  LUABIND_CHECK_ARGN(==,1);
  GPUMirroredMemoryBlockConstructor(L,obj);
  LUABIND_RETURN(FloatGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD FloatGPUMirroredMemoryBlock size
{
  LUABIND_RETURN(uint,obj->getSize());
}
//BIND_END

//BIND_METHOD FloatGPUMirroredMemoryBlock set
{
  LUABIND_CHECK_ARGN(==,2);
  GPUMirroredMemoryBlockSet(L,obj);
  LUABIND_RETURN(FloatGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD FloatGPUMirroredMemoryBlock get
{
  LUABIND_RETURN(float,GPUMirroredMemoryBlockGet(L,obj));
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME DoubleGPUMirroredMemoryBlock mathcore.block.double
//BIND_CPP_CLASS DoubleGPUMirroredMemoryBlock

//BIND_CONSTRUCTOR DoubleGPUMirroredMemoryBlock
{
  LUABIND_CHECK_ARGN(==,1);
  GPUMirroredMemoryBlockConstructor(L,obj);
  LUABIND_RETURN(DoubleGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD DoubleGPUMirroredMemoryBlock size
{
  LUABIND_RETURN(uint,obj->getSize());
}
//BIND_END

//BIND_METHOD DoubleGPUMirroredMemoryBlock set
{
  LUABIND_CHECK_ARGN(==,2);
  GPUMirroredMemoryBlockSet(L,obj);
  LUABIND_RETURN(DoubleGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD DoubleGPUMirroredMemoryBlock get
{
  LUABIND_RETURN(double,GPUMirroredMemoryBlockGet(L,obj));
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME Int32GPUMirroredMemoryBlock mathcore.block.int32
//BIND_CPP_CLASS Int32GPUMirroredMemoryBlock

//BIND_CONSTRUCTOR Int32GPUMirroredMemoryBlock
{
  LUABIND_CHECK_ARGN(==,1);
  GPUMirroredMemoryBlockConstructor(L,obj);
  LUABIND_RETURN(Int32GPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD Int32GPUMirroredMemoryBlock size
{
  LUABIND_RETURN(uint,obj->getSize());
}
//BIND_END

//BIND_METHOD Int32GPUMirroredMemoryBlock set
{
  LUABIND_CHECK_ARGN(==,2);
  GPUMirroredMemoryBlockSet(L,obj);
  LUABIND_RETURN(Int32GPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD Int32GPUMirroredMemoryBlock get
{
  LUABIND_RETURN(int,GPUMirroredMemoryBlockGet(L,obj));
}
//BIND_END
