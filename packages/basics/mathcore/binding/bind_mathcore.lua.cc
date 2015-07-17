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
#include "bind_april_io.h"
#include "bind_complex.h"
#include "bind_mathcore.h"
#include "cmath_overloads.h"
#include "error_print.h"
#include "luabindutil.h"
#include "luabindmacros.h"
#include "maxmin.h"

using namespace AprilMath;

typedef bool boolean;

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(CharGPUMirroredMemoryBlock);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(FloatGPUMirroredMemoryBlock);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(DoubleGPUMirroredMemoryBlock);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(Int32GPUMirroredMemoryBlock);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(ComplexFGPUMirroredMemoryBlock);
IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(BoolGPUMirroredMemoryBlock);

#define MAKE_READ_BLOCK_LUA_METHOD(BlockType, Type) do {        \
    BlockType *obj = readBlockLuaMethod<Type>(L);               \
    if (obj == 0) {                                             \
      luaL_error(L, "Error happens reading from file stream");  \
    }                                                           \
    else {                                                      \
      lua_push##BlockType(L, obj);                              \
    }                                                           \
  } while(false)

template<typename T>
GPUMirroredMemoryBlock<T> *readBlockLuaMethod(lua_State *L) {
  AprilUtils::SharedPtr<AprilIO::StreamInterface> stream;
  if (lua_isstring(L, 1)) {
    stream = new InputLuaStringStream(L, 1);
  }
  else {
    stream = lua_toAuxStreamInterface<AprilIO::StreamInterface>(L,1);
    if (!stream) luaL_error(L, "Needs a stream as first argument");
  }
  AprilUtils::LuaTable options(L,2);
  return GPUMirroredMemoryBlock<T>::read(stream.get(), options); 
}

namespace MathCoreBinding {
  template<typename T> T luaToFunc(lua_State *L, int n) {
    return AprilUtils::LuaTable::convertTo<T>(L,n);
  }
}

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
        ptr[i] = MathCoreBinding::luaToFunc<T>(L, -1);
        lua_pop(L,1);
      }
    }
  }
#undef FUNCTION_NAME

#define FUNCTION_NAME "raw_set"
  template<typename T>
  void GPUMirroredMemoryBlockSet(lua_State *L,
                                 GPUMirroredMemoryBlock<T> *obj) {
    T value;
    unsigned int i;
    LUABIND_GET_PARAMETER(1,uint,i);
    value = MathCoreBinding::luaToFunc<T>(L, 2);
    if (i >= obj->getSize()) ERROR_EXIT(128, "Index out of bounds\n");
    T *ptr = obj->getPPALForWrite();
    ptr[i] = value;
  }
#undef FUNCTION_NAME

#define FUNCTION_NAME "raw_get"
  template<typename T>
  T GPUMirroredMemoryBlockGet(lua_State *L,
                              GPUMirroredMemoryBlock<T> *obj) {
    unsigned int i;
    LUABIND_GET_PARAMETER(1,uint,i);
    if (i >= obj->getSize()) ERROR_EXIT(128, "Index out of bounds\n");
    const T *ptr = obj->getPPALForRead();
    return ptr[i];
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

//BIND_FUNCTION mathcore.get_mmap_allocation
{
  LUABIND_RETURN(bool, GPUMirroredMemoryBlockBase::getUseMMapAllocation());
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

//BIND_FUNCTION mathcore.get_use_cuda_default
{
  LUABIND_RETURN(bool, GPUMirroredMemoryBlockBase::USE_CUDA_DEFAULT);
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME Serializable aprilio.serializable

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME FloatGPUMirroredMemoryBlock mathcore.block.float
//BIND_CPP_CLASS FloatGPUMirroredMemoryBlock
//BIND_SUBCLASS_OF FloatGPUMirroredMemoryBlock Serializable

//BIND_CONSTRUCTOR FloatGPUMirroredMemoryBlock
{
  LUABIND_CHECK_ARGN(==,1);
  GPUMirroredMemoryBlockConstructor(L,obj);
  LUABIND_RETURN(FloatGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_CLASS_METHOD FloatGPUMirroredMemoryBlock read
{
  MAKE_READ_BLOCK_LUA_METHOD(FloatGPUMirroredMemoryBlock, float);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//BIND_METHOD FloatGPUMirroredMemoryBlock size
{
  LUABIND_RETURN(uint,obj->getSize());
}
//BIND_END

//BIND_METHOD FloatGPUMirroredMemoryBlock raw_set
{
  LUABIND_CHECK_ARGN(==,2);
  GPUMirroredMemoryBlockSet(L,obj);
  LUABIND_RETURN(FloatGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD FloatGPUMirroredMemoryBlock raw_get
{
  LUABIND_RETURN(float,GPUMirroredMemoryBlockGet(L,obj));
}
//BIND_END

//BIND_METHOD FloatGPUMirroredMemoryBlock get_reference_string
{
  char buff[128];
  sprintf(buff,"data= %p", (void*)obj);
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_METHOD FloatGPUMirroredMemoryBlock copy
{
  FloatGPUMirroredMemoryBlock *other;
  LUABIND_GET_PARAMETER(1, FloatGPUMirroredMemoryBlock, other);
  obj->copyFromBlock(0, other, 0, AprilUtils::min(obj->getSize(),other->getSize()));
  LUABIND_RETURN(FloatGPUMirroredMemoryBlock, obj);
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME DoubleGPUMirroredMemoryBlock mathcore.block.double
//BIND_CPP_CLASS DoubleGPUMirroredMemoryBlock
//BIND_SUBCLASS_OF DoubleGPUMirroredMemoryBlock Serializable

//BIND_CONSTRUCTOR DoubleGPUMirroredMemoryBlock
{
  LUABIND_CHECK_ARGN(==,1);
  GPUMirroredMemoryBlockConstructor(L,obj);
  LUABIND_RETURN(DoubleGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_CLASS_METHOD DoubleGPUMirroredMemoryBlock read
{
  MAKE_READ_BLOCK_LUA_METHOD(DoubleGPUMirroredMemoryBlock, double);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//BIND_METHOD DoubleGPUMirroredMemoryBlock size
{
  LUABIND_RETURN(uint,obj->getSize());
}
//BIND_END

//BIND_METHOD DoubleGPUMirroredMemoryBlock raw_set
{
  LUABIND_CHECK_ARGN(==,2);
  GPUMirroredMemoryBlockSet(L,obj);
  LUABIND_RETURN(DoubleGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD DoubleGPUMirroredMemoryBlock raw_get
{
  LUABIND_RETURN(double,GPUMirroredMemoryBlockGet(L,obj));
}
//BIND_END

//BIND_METHOD DoubleGPUMirroredMemoryBlock get_reference_string
{
  char buff[128];
  sprintf(buff,"data= %p", (void*)obj);
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_METHOD DoubleGPUMirroredMemoryBlock copy
{
  DoubleGPUMirroredMemoryBlock *other;
  LUABIND_GET_PARAMETER(1, DoubleGPUMirroredMemoryBlock, other);
  obj->copyFromBlock(0, other, 0, AprilUtils::min(obj->getSize(),other->getSize()));
  LUABIND_RETURN(DoubleGPUMirroredMemoryBlock, obj);
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME Int32GPUMirroredMemoryBlock mathcore.block.int32
//BIND_CPP_CLASS Int32GPUMirroredMemoryBlock
//BIND_SUBCLASS_OF Int32GPUMirroredMemoryBlock Serializable

//BIND_CONSTRUCTOR Int32GPUMirroredMemoryBlock
{
  LUABIND_CHECK_ARGN(==,1);
  GPUMirroredMemoryBlockConstructor(L,obj);
  LUABIND_RETURN(Int32GPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_CLASS_METHOD Int32GPUMirroredMemoryBlock read
{
  MAKE_READ_BLOCK_LUA_METHOD(Int32GPUMirroredMemoryBlock, int);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//BIND_METHOD Int32GPUMirroredMemoryBlock size
{
  LUABIND_RETURN(uint,obj->getSize());
}
//BIND_END

//BIND_METHOD Int32GPUMirroredMemoryBlock raw_set
{
  LUABIND_CHECK_ARGN(==,2);
  GPUMirroredMemoryBlockSet(L,obj);
  LUABIND_RETURN(Int32GPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD Int32GPUMirroredMemoryBlock raw_get
{
  LUABIND_RETURN(int,GPUMirroredMemoryBlockGet(L,obj));
}
//BIND_END

//BIND_METHOD Int32GPUMirroredMemoryBlock get_reference_string
{
  char buff[128];
  sprintf(buff,"data= %p", (void*)obj);
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_METHOD Int32GPUMirroredMemoryBlock copy
{
  Int32GPUMirroredMemoryBlock *other;
  LUABIND_GET_PARAMETER(1, Int32GPUMirroredMemoryBlock, other);
  obj->copyFromBlock(0, other, 0, AprilUtils::min(obj->getSize(),other->getSize()));
  LUABIND_RETURN(Int32GPUMirroredMemoryBlock, obj);
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME BoolGPUMirroredMemoryBlock mathcore.block.bool
//BIND_CPP_CLASS BoolGPUMirroredMemoryBlock
//BIND_SUBCLASS_OF BoolGPUMirroredMemoryBlock Serializable

//BIND_CONSTRUCTOR BoolGPUMirroredMemoryBlock
{
  LUABIND_CHECK_ARGN(==,1);
  GPUMirroredMemoryBlockConstructor(L,obj);
  LUABIND_RETURN(BoolGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_CLASS_METHOD BoolGPUMirroredMemoryBlock read
{
  MAKE_READ_BLOCK_LUA_METHOD(BoolGPUMirroredMemoryBlock, boolean);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//BIND_METHOD BoolGPUMirroredMemoryBlock size
{
  LUABIND_RETURN(uint,obj->getSize());
}
//BIND_END

//BIND_METHOD BoolGPUMirroredMemoryBlock raw_set
{
  LUABIND_CHECK_ARGN(==,2);
  GPUMirroredMemoryBlockSet(L,obj);
  LUABIND_RETURN(BoolGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD BoolGPUMirroredMemoryBlock raw_get
{
  LUABIND_RETURN(boolean,GPUMirroredMemoryBlockGet(L,obj));
}
//BIND_END

//BIND_METHOD BoolGPUMirroredMemoryBlock get_reference_string
{
  char buff[128];
  sprintf(buff,"data= %p", (void*)obj);
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_METHOD BoolGPUMirroredMemoryBlock copy
{
  BoolGPUMirroredMemoryBlock *other;
  LUABIND_GET_PARAMETER(1, BoolGPUMirroredMemoryBlock, other);
  obj->copyFromBlock(0, other, 0, AprilUtils::min(obj->getSize(),other->getSize()));
  LUABIND_RETURN(BoolGPUMirroredMemoryBlock, obj);
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME CharGPUMirroredMemoryBlock mathcore.block.char
//BIND_CPP_CLASS CharGPUMirroredMemoryBlock
//BIND_SUBCLASS_OF CharGPUMirroredMemoryBlock Serializable

//BIND_CONSTRUCTOR CharGPUMirroredMemoryBlock
{
  LUABIND_CHECK_ARGN(==,1);
  GPUMirroredMemoryBlockConstructor(L,obj);
  LUABIND_RETURN(CharGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_CLASS_METHOD CharGPUMirroredMemoryBlock read
{
  MAKE_READ_BLOCK_LUA_METHOD(CharGPUMirroredMemoryBlock, char);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//BIND_METHOD CharGPUMirroredMemoryBlock size
{
  LUABIND_RETURN(uint,obj->getSize());
}
//BIND_END

//BIND_METHOD CharGPUMirroredMemoryBlock raw_set
{
  LUABIND_CHECK_ARGN(==,2);
  GPUMirroredMemoryBlockSet(L,obj);
  LUABIND_RETURN(CharGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD CharGPUMirroredMemoryBlock raw_get
{
  char c = GPUMirroredMemoryBlockGet(L,obj);
  lua_pushlstring(L, &c, 1);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//BIND_METHOD CharGPUMirroredMemoryBlock get_reference_string
{
  char buff[128];
  sprintf(buff,"data= %p", (void*)obj);
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_METHOD CharGPUMirroredMemoryBlock copy
{
  CharGPUMirroredMemoryBlock *other;
  LUABIND_GET_PARAMETER(1, CharGPUMirroredMemoryBlock, other);
  obj->copyFromBlock(0, other, 0, AprilUtils::min(obj->getSize(),other->getSize()));
  LUABIND_RETURN(CharGPUMirroredMemoryBlock, obj);
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME ComplexFGPUMirroredMemoryBlock mathcore.block.complex
//BIND_CPP_CLASS ComplexFGPUMirroredMemoryBlock
//BIND_SUBCLASS_OF ComplexFGPUMirroredMemoryBlock Serializable

//BIND_CONSTRUCTOR ComplexFGPUMirroredMemoryBlock
{
  LUABIND_CHECK_ARGN(==,1);
  GPUMirroredMemoryBlockConstructor(L,obj);
  LUABIND_RETURN(ComplexFGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_CLASS_METHOD ComplexFGPUMirroredMemoryBlock read
{
  MAKE_READ_BLOCK_LUA_METHOD(ComplexFGPUMirroredMemoryBlock, ComplexF);
  LUABIND_INCREASE_NUM_RETURNS(1);
}
//BIND_END

//BIND_METHOD ComplexFGPUMirroredMemoryBlock size
{
  LUABIND_RETURN(uint,obj->getSize());
}
//BIND_END

//BIND_METHOD ComplexFGPUMirroredMemoryBlock raw_set
{
  LUABIND_CHECK_ARGN(==,2);
  GPUMirroredMemoryBlockSet(L,obj);
  LUABIND_RETURN(ComplexFGPUMirroredMemoryBlock,obj);
}
//BIND_END

//BIND_METHOD ComplexFGPUMirroredMemoryBlock raw_get
{
  LUABIND_RETURN(ComplexF,GPUMirroredMemoryBlockGet(L,obj));
}
//BIND_END

//BIND_METHOD ComplexFGPUMirroredMemoryBlock get_reference_string
{
  char buff[128];
  sprintf(buff,"data= %p", (void*)obj);
  LUABIND_RETURN(string, buff);
}
//BIND_END

//BIND_METHOD ComplexFGPUMirroredMemoryBlock copy
{
  ComplexFGPUMirroredMemoryBlock *other;
  LUABIND_GET_PARAMETER(1, ComplexFGPUMirroredMemoryBlock, other);
  obj->copyFromBlock(0, other, 0, AprilUtils::min(obj->getSize(),other->getSize()));
  LUABIND_RETURN(ComplexFGPUMirroredMemoryBlock, obj);
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_FUNCTION mathcore.limits.float.max
{
  LUABIND_RETURN(float,Limits<float>::max());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.float.min
{
  LUABIND_RETURN(float,Limits<float>::min());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.float.lowest
{
  LUABIND_RETURN(float,Limits<float>::lowest());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.float.epsilon
{
  LUABIND_RETURN(float,Limits<float>::epsilon());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.float.infinity
{
  LUABIND_RETURN(float,Limits<float>::infinity());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.float.quiet_NaN
{
  LUABIND_RETURN(float,Limits<float>::quiet_NaN());
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_FUNCTION mathcore.limits.double.max
{
  LUABIND_RETURN(double,Limits<double>::max());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.double.min
{
  LUABIND_RETURN(double,Limits<double>::min());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.double.lowest
{
  LUABIND_RETURN(double,Limits<double>::lowest());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.double.epsilon
{
  LUABIND_RETURN(double,Limits<double>::epsilon());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.double.infinity
{
  LUABIND_RETURN(double,Limits<double>::infinity());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.double.quiet_NaN
{
  LUABIND_RETURN(double,Limits<double>::quiet_NaN());
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_FUNCTION mathcore.limits.int32.max
{
  LUABIND_RETURN(int,Limits<int32_t>::max());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.int32.min
{
  LUABIND_RETURN(int,Limits<int32_t>::min());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.int32.lowest
{
  LUABIND_RETURN(int,Limits<int32_t>::lowest());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.int32.epsilon
{
  LUABIND_RETURN(int,Limits<int32_t>::epsilon());
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_FUNCTION mathcore.limits.char.max
{
  LUABIND_RETURN(char,Limits<char>::max());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.char.min
{
  LUABIND_RETURN(char,Limits<char>::min());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.char.lowest
{
  LUABIND_RETURN(char,Limits<char>::lowest());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.char.epsilon
{
  LUABIND_RETURN(char,Limits<char>::epsilon());
}
//BIND_END

////////////////////////////////////////////////////////////////////////////

//BIND_FUNCTION mathcore.limits.complex.max
{
  LUABIND_RETURN(ComplexF,Limits<ComplexF>::max());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.complex.min
{
  LUABIND_RETURN(ComplexF,Limits<ComplexF>::min());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.complex.lowest
{
  LUABIND_RETURN(ComplexF,Limits<ComplexF>::lowest());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.complex.epsilon
{
  LUABIND_RETURN(ComplexF,Limits<ComplexF>::epsilon());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.complex.infinity
{
  LUABIND_RETURN(ComplexF,Limits<ComplexF>::infinity());
}
//BIND_END

//BIND_FUNCTION mathcore.limits.complex.quiet_NaN
{
  LUABIND_RETURN(ComplexF,Limits<ComplexF>::quiet_NaN());
}
//BIND_END
