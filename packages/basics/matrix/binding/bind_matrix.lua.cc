/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
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
//BIND_HEADER_C
extern "C" {
#include <ctype.h>
}
#include "bind_april_io.h"
#include "bind_mathcore.h"
#include "bind_mtrand.h"
#include "bind_matrix_int32.h"
#include "bind_matrix_bool.h"
#include "bind_sparse_matrix.h"
#include "luabindutil.h"
#include "luabindmacros.h"
#include "lua_string.h"
#include "matrix_ext.h"
#include "mystring.h"
#include "smart_ptr.h"
#include "utilMatrixFloat.h"

using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Boolean;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilMath::MatrixExt::Misc;
using namespace AprilMath::MatrixExt::LAPACK;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;

IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(MatrixFloat);

#define FUNCTION_NAME "read_vector"
static int *read_vector(lua_State *L, const char *key, int num_dim, int add) {
  int *v = 0;
  lua_getfield(L, 1, key);
  if (!lua_isnil(L, -1)) {
    LUABIND_CHECK_PARAMETER(-1, table);
    int table_len;
    LUABIND_TABLE_GETN(-1, table_len);
    if (table_len != num_dim)
      LUABIND_FERROR3("Table '%s' with incorrect size, expected %d, found %d",
		      key, num_dim, table_len);
    v = new int[num_dim];
    for(int i=0; i < num_dim; i++) {
      lua_rawgeti(L, -1, i+1);
      v[i] = static_cast<int>(lua_tonumber(L, -1)) + add;
      lua_pop(L,1);
    }
  }
  lua_pop(L, 1);
  return v;
}
#undef FUNCTION_NAME

int sliding_window_iterator_function(lua_State *L) {
  SlidingWindow *obj = lua_toSlidingWindow(L,1);
  if (obj->isEnd()) {
    lua_pushnil(L);
    return 1;
  }
  // lua_pushSlidingWindow(L, obj);
  MatrixFloat *mat = obj->getMatrix();
  lua_pushMatrixFloat(L, mat);
  obj->next();
  return 1;
}

template<typename T>
static bool check_number(lua_State *L, int i, T &dest) {
  if (lua_isnumber(L,i)) {
    dest = static_cast<T>(lua_tonumber(L,i));
    return true;
  }
  const char *str = lua_tostring(L,i);
  if ( str != 0 &&
       ( AprilUtils::strcmpi(str, "-nan")==0 ||
         AprilUtils::strcmpi(str, "nan") ) ) {
    dest = T(0.0f/0.0f);
    return true;
  }
  return false;
}

//BIND_END

//BIND_HEADER_H
#include "bind_april_io.h"
#include "gpu_mirrored_memory_block.h"
#include "matrixFloat.h"
#include "luabindmacros.h"
#include "luabindutil.h"
#include "utilLua.h"

using namespace Basics;

typedef MatrixFloat::sliding_window SlidingWindow;

namespace Basics {

  /// Implements binding functions reusable in different Matrix flavors.
  template<typename T>
  class MatrixBindings {
  public:
#define BEGIN_METHOD(name)       static int name(lua_State *L, Matrix<T> *obj)
#define BEGIN_CLASS_METHOD(name) static int name(lua_State *L)
    
#define FUNCTION_NAME "constructor"
    BEGIN_CLASS_METHOD(constructor)
    {
      int i,argn;
      argn = lua_gettop(L); // number of arguments
      LUABIND_CHECK_ARGN(>=, 1);
      int ndims = (!lua_isnumber(L,argn)) ? argn-1 : argn;
      AprilUtils::UniquePtr<int []> dim;
      if (ndims == 0) { // caso matrix{valores} o matrix(block)
        ndims = 1;
        dim = new int[ndims];
        dim[0] = -1;
      } else {
        dim = new int[ndims];
        for (i=1; i <= ndims; i++) {
          if (!lua_isnumber(L,i))
            // TODO: Este mensaje de error parece que no es correcto... y no se todavia por que!!!
            LUABIND_FERROR2("incorrect argument to matrix dimension (arg %d must"
                            " be a number and is a %s)",
                            i, lua_typename(L,i));
          dim[i-1] = (int)lua_tonumber(L,i);
          if (dim[i-1] <= 0)
            LUABIND_FERROR1("incorrect argument to matrix dimension (arg %d must be >0)",i);
        }
      }
      Matrix<T>* obj;
      if (AprilUtils::LuaTable::
          checkType<AprilMath::GPUMirroredMemoryBlock<T>*>(L,argn)) {
        AprilMath::GPUMirroredMemoryBlock<T> *block;
        block = AprilUtils::LuaTable::
          convertTo<AprilMath::GPUMirroredMemoryBlock<T>*>(L, argn);
        if (dim[0] == -1) dim[0] = block->getSize();
        obj = new Matrix<T>(ndims, dim.get(), block);
      }
      else {
        if (lua_istable(L,argn)) {
          if (dim[0] == -1) LUABIND_TABLE_GETN(1, dim[0]);
          obj = new Matrix<T>(ndims, dim.get());
          int i=1;
          int len;
          LUABIND_TABLE_GETN(argn, len);
          if (len != obj->size()) {
            LUABIND_FERROR2("Incorrect number of elements at the given table, "
                            "found %d, expected %d", len, obj->size());
          }
          for (typename Matrix<T>::iterator it(obj->begin());
               it != obj->end(); ++it, ++i) {
            lua_rawgeti(L,argn,i);
            /*
              if (!check_number(L,-1,*it))
            */
            if (!AprilUtils::LuaTable::checkType<T>(L, -1)) {
              LUABIND_FERROR1("The given table has an invalid value at position"
                              " %d, check table size and its content", i);
            }
            *it = AprilUtils::LuaTable::convertTo<T>(L, -1);
            lua_remove(L,-1);
          } // for each matrix position
        } // if lua_istable(L,argn)
        else {
          obj = new Matrix<T>(ndims, dim.get());
        }
      } // else { !checkType(L,argn) }
      AprilUtils::LuaTable::pushInto(L, obj);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "convert_to"
    BEGIN_METHOD(convert_to)
    {
      LUABIND_CHECK_ARGN(==,1);
      const char *type;
      LUABIND_GET_PARAMETER(1, string, type);
      if (!strcmp(type,"float")) {
        Matrix<float> *obj2 = AprilMath::MatrixExt::Misc::
          matConvertTo<T,float>(obj);
        AprilUtils::LuaTable::pushInto(L, obj2);
      }
      else if (!strcmp(type,"bool")) {
        Matrix<bool> *obj2 = AprilMath::MatrixExt::Misc::
          matConvertTo<T,bool>(obj);
        AprilUtils::LuaTable::pushInto(L, obj2);
      }
      else if (!strcmp(type,"int32")) {
        Matrix<int32_t> *obj2 = AprilMath::MatrixExt::Misc::
          matConvertTo<T,int32_t>(obj);
        AprilUtils::LuaTable::pushInto(L, obj2);
      }
      else if (!strcmp(type,"double")) {
        Matrix<double> *obj2 = AprilMath::MatrixExt::Misc::
          matConvertTo<T,double>(obj);
        AprilUtils::LuaTable::pushInto(L, obj2);
      }
      else if (!strcmp(type,"char")) {
        Matrix<char> *obj2 = AprilMath::MatrixExt::Misc::
          matConvertTo<T,char>(obj);
        AprilUtils::LuaTable::pushInto(L, obj2);
      }
      else {
        LUABIND_FERROR1("Not implemented casting for type %s", type);
      }
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "rewrap"
    BEGIN_METHOD(rewrap)
    {
      LUABIND_CHECK_ARGN(>=, 1);
      int ndims;
      ndims = lua_gettop(L); // number of dimensions
      bool clone_if_not_contiguous = false;
      if (lua_isboolean(L, ndims)) {
        LUABIND_GET_PARAMETER(ndims, boolean, clone_if_not_contiguous);
        --ndims;
      }
      AprilUtils::UniquePtr<int []> dims( new int[ndims] );
      for (int i=1; i <= ndims; i++) {
        LUABIND_GET_PARAMETER(i, int, dims[i-1]);
        if (dims[i-1] <= 0) {
          LUABIND_FERROR1("incorrect argument to matrix dimension "
                          "(arg %d must be >0)",i);
        }
      }
      Matrix<T> *new_obj = obj->rewrap(dims.get(), ndims,
                                       clone_if_not_contiguous);
      AprilUtils::LuaTable::pushInto(L, new_obj);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "get_reference_string"
    BEGIN_METHOD(get_reference_string)
    {
      char buff[128];
      sprintf(buff,"%p data= %p",
              (void*)obj,
              (void*)obj->getRawDataAccess());
      lua_pushstring(L, buff);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "copy_from_table"
    BEGIN_METHOD(copy_from_table)
    {
      LUABIND_CHECK_ARGN(==, 1);
      LUABIND_CHECK_PARAMETER(1, table);
      int veclen;
      LUABIND_TABLE_GETN(1, veclen);
      if (veclen != obj->size())
        LUABIND_FERROR2("wrong size %d instead of %d",veclen,obj->size());
      int i=1;
      for (typename Matrix<T>::iterator it(obj->begin());
           it != obj->end(); ++it, ++i) {
        lua_rawgeti(L,1,i);
        if (!AprilUtils::LuaTable::checkType<T>(L,-1)) {
          LUABIND_FERROR1("The given table has a no number value at position %d, "
                          "the table could be smaller than matrix size", i);
        }
        *it = AprilUtils::LuaTable::convertTo<T>(L,-1);
        lua_remove(L,-1);
      }
      AprilUtils::LuaTable::pushInto(L, obj);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "get"
    BEGIN_METHOD(get)
    {
      int argn = lua_gettop(L); // number of arguments
      if (argn != obj->getNumDim())
        LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim());
      T ret;
      if (obj->getNumDim() == 1) {
        int v1;
        LUABIND_GET_PARAMETER(1,int,v1);
        if (v1<1 || v1 > obj->getDimSize(0)) {
          LUABIND_FERROR2("wrong index parameter 1, %d is not <= %d",
                          v1, obj->getDimSize(0));
        }
        ret = (*obj)(v1-1);
      }
      else if (obj->getNumDim() == 2) {
        int v1, v2;
        LUABIND_GET_PARAMETER(1,int,v1);
        LUABIND_GET_PARAMETER(2,int,v2);
        if (v1<1 || v1 > obj->getDimSize(0)) {
          LUABIND_FERROR2("wrong index parameter 1, %d is not <= %d or is not >= 1",
                          v1, obj->getDimSize(0));
        }
        if (v2<1 || v2 > obj->getDimSize(1)) {
          LUABIND_FERROR2("wrong index parameter 2, %d is not <= %d or is not >= 1",
                          v2, obj->getDimSize(1));
        }
        ret = (*obj)(v1-1, v2-1);
      }
      else {
        AprilUtils::UniquePtr<int []> coords( new int[obj->getNumDim()] );
        for (int i=0; i<obj->getNumDim(); ++i) {
          LUABIND_GET_PARAMETER(i+1,int,coords[i]);
          if (coords[i]<1 || coords[i] > obj->getDimSize(i)) {
            LUABIND_FERROR2("wrong index parameter %d, %d is not <= %d or is not >= 1",
                            coords[i], obj->getDimSize(i));
          }
          coords[i]--;
        }
        ret = (*obj)(coords.get(), obj->getNumDim());
      }
      AprilUtils::LuaTable::pushInto(L, ret);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "set"
    BEGIN_METHOD(set)
    {
      int argn = lua_gettop(L); // number of arguments
      if (argn != obj->getNumDim()+1)
        LUABIND_FERROR2("wrong size %d instead of %d",argn,obj->getNumDim()+1);
      T f;
      if (obj->getNumDim() == 1) {
        int v1;
        LUABIND_GET_PARAMETER(1,int,v1);
        if (v1<1 || v1 > obj->getDimSize(0)) {
          LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
                          v1, obj->getDimSize(0));
        }
        f = AprilUtils::LuaTable::convertTo<T>(L, obj->getNumDim()+1);
        (*obj)(v1-1) = f;
      }
      else if (obj->getNumDim() == 2) {
        int v1, v2;
        LUABIND_GET_PARAMETER(1,int,v1);
        LUABIND_GET_PARAMETER(2,int,v2);
        if (v1<1 || v1 > obj->getDimSize(0)) {
          LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
                          v1, obj->getDimSize(0));
        }
        if (v2<1 || v2 > obj->getDimSize(1)) {
          LUABIND_FERROR2("wrong index parameter: 2 <= %d <= %d is incorrect",
                          v2, obj->getDimSize(1));
        }
        f = AprilUtils::LuaTable::convertTo<T>(L, obj->getNumDim()+1);
        (*obj)(v1-1, v2-1) = f;
      }
      else {
        AprilUtils::UniquePtr<int []> coords( new int[obj->getNumDim()] );
        for (int i=0; i<obj->getNumDim(); ++i) {
          LUABIND_GET_PARAMETER(i+1,int,coords[i]);
          if (coords[i]<1 || coords[i] > obj->getDimSize(i)) {
            LUABIND_FERROR2("wrong index parameter: 1 <= %d <= %d is incorrect",
                            coords[i], obj->getDimSize(i));
          }
          coords[i]--;
        }
        f = AprilUtils::LuaTable::convertTo<T>(L, obj->getNumDim()+1);
        (*obj)(coords.get(), obj->getNumDim()) = f;
      }
      AprilUtils::LuaTable::pushInto(L, obj);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "raw_get"
    BEGIN_METHOD(raw_get)
    {
      int raw_pos;
      LUABIND_GET_PARAMETER(1, int, raw_pos);
      AprilUtils::LuaTable::pushInto(L, (*obj)[raw_pos]);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "raw_set"
    BEGIN_METHOD(raw_set)
    {
      int raw_pos;
      LUABIND_GET_PARAMETER(1, int, raw_pos);
      T value = AprilUtils::LuaTable::convertTo<T>(L, 2);      
      (*obj)[raw_pos] = value;
      AprilUtils::LuaTable::pushInto(L, obj);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "set_use_cuda"
    BEGIN_METHOD(set_use_cuda)
    {
      LUABIND_CHECK_ARGN(==, 1);
      LUABIND_CHECK_PARAMETER(1, bool);
      bool v;
      LUABIND_GET_PARAMETER(1,bool, v);
      obj->setUseCuda(v);
      AprilUtils::LuaTable::pushInto(L, obj);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "dim"
    BEGIN_METHOD(dim)
    {
      LUABIND_CHECK_ARGN(>=, 0);
      LUABIND_CHECK_ARGN(<=, 1);
      int pos;
      const int *d=obj->getDimPtr();
      LUABIND_GET_OPTIONAL_PARAMETER(1, int, pos, -1);
      if (pos < 1) {
        AprilUtils::LuaTable vec(d, obj->getNumDim(), L);
        vec.pushTable(L);
      }
      else {
        lua_pushint(L, d[pos-1]);
      }
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "stride"
    BEGIN_METHOD(stride)
    {
      LUABIND_CHECK_ARGN(>=, 0);
      LUABIND_CHECK_ARGN(<=, 1);
      int pos;
      const int *s=obj->getStridePtr();
      LUABIND_GET_OPTIONAL_PARAMETER(1, int, pos, -1);
      if (pos < 1) {
        AprilUtils::LuaTable vec(s, obj->getNumDim(), L);
        vec.pushTable(L);
      }
      else {
        lua_pushint(L, s[pos-1]);
      }
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "slice"
    BEGIN_METHOD(slice)
    {
      LUABIND_CHECK_ARGN(>=,2);
      LUABIND_CHECK_ARGN(<=,3);
      LUABIND_CHECK_PARAMETER(1, table);
      LUABIND_CHECK_PARAMETER(2, table);
      AprilUtils::UniquePtr<int []> coords, sizes;
      int coords_len, sizes_len;
      bool clone;
      LUABIND_TABLE_GETN(1, coords_len);
      LUABIND_TABLE_GETN(2, sizes_len);
      if (coords_len != sizes_len || coords_len != obj->getNumDim())
        LUABIND_FERROR3("Incorrect number of dimensions, expected %d, "
                        "found %d and %d\n",
                        obj->getNumDim(), coords_len, sizes_len);
      coords = new int[coords_len];
      sizes  = new int[sizes_len];
      LUABIND_TABLE_TO_VECTOR_SUB1(1, int, coords, coords_len);
      LUABIND_TABLE_TO_VECTOR(2, int, sizes,  sizes_len);
      LUABIND_GET_OPTIONAL_PARAMETER(3, bool, clone, false);
      for (int i=0; i<sizes_len; ++i) {
        if (coords[i] < 0 || sizes[i] < 1 ||
            sizes[i]+coords[i] > obj->getDimSize(i)) {
          LUABIND_FERROR1("Incorrect size or coord at position %d\n", i+1);
        }
      }
      Matrix<T> *obj2 = new Matrix<T>(obj, coords.get(), sizes.get(), clone);
      AprilUtils::LuaTable::pushInto(L, obj2);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "select"
    BEGIN_METHOD(select)
    {
      LUABIND_CHECK_ARGN(>=,2);
      LUABIND_CHECK_ARGN(<=,3);
      LUABIND_CHECK_PARAMETER(1, int);
      LUABIND_CHECK_PARAMETER(2, int);
      int dim, index;
      Matrix<T> *dest = 0;
      LUABIND_GET_PARAMETER(1, int, dim);
      LUABIND_GET_PARAMETER(2, int, index);
      int n = lua_gettop(L);
      if (n == 3) dest = AprilUtils::LuaTable::convertTo<Matrix<T>*>(L, 3);
      Matrix<T> *obj2 = obj->select(dim-1, index-1, dest);
      AprilUtils::LuaTable::pushInto(L, obj2);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "as"
    BEGIN_CLASS_METHOD(as)
    {
      LUABIND_CHECK_ARGN(==, 1);
      Matrix<T> *m;
      m = AprilUtils::LuaTable::convertTo<Matrix<T>*>(L, 1);
      AprilUtils::LuaTable::pushInto(L, m->cloneOnlyDims());
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "transpose"
    BEGIN_METHOD(transpose)
    {
      int argn;
      argn = lua_gettop(L);
      if (argn == 0) {
        AprilUtils::LuaTable::pushInto(L, obj->transpose());
      }
      else {
        int d1,d2;
        LUABIND_GET_PARAMETER(1, int, d1);
        LUABIND_GET_PARAMETER(2, int, d2);
        AprilUtils::LuaTable::pushInto(L, obj->transpose(d1-1, d2-1));
      }
      return 1;
    }
#undef FUNCTION_NAME
    
#define FUNCTION_NAME "deserialize"
    BEGIN_CLASS_METHOD(deserialize)
    {
      check_table_fields(L, 1, "stride", "sizes", "data", "offset", 
                         (const char *)0);
      int offset;
      AprilUtils::UniquePtr<int []> sizes;
      AprilUtils::UniquePtr<int []> stride;
      AprilMath::GPUMirroredMemoryBlock<T> *data;
      lua_getfield(L, 1, "data");
      data = AprilUtils::LuaTable::
        convertTo<AprilMath::GPUMirroredMemoryBlock<T>*>(L, -1);
      lua_pop(L, 1);
      lua_getfield(L, 1, "offset");
      offset = lua_toint(L, -1);
      lua_pop(L, 1);
      lua_getfield(L, 1, "sizes");
      int len = luaL_len(L, -1);
      sizes  = new int[len];
      stride = new int[len];
      for (int i=0; i<len; ++i) {
        lua_rawgeti(L, -1, i+1);
        sizes[i] = lua_toint(L, -1);
        lua_pop(L, 1);
      }
      lua_pop(L, 1);
      lua_getfield(L, 1, "stride");
      if (luaL_len(L, -1) != len) {
        LUABIND_ERROR("Incompatible table sizes");
      }
      for (int i=0; i<len; ++i) {
        lua_rawgeti(L, -1, i+1);
        stride[i] = lua_toint(L, -1);
        lua_pop(L, 1);
      }
      lua_pop(L, 1);
      Matrix<T> *obj;
      obj = new Matrix<T>(len, sizes.get(), data, offset, stride.get());
      AprilUtils::LuaTable::pushInto<Matrix<T>*>(L, obj);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "read"
    BEGIN_CLASS_METHOD(read)
    {
      AprilIO::StreamInterface *stream =
        lua_toAuxStreamInterface<AprilIO::StreamInterface>(L,1);
      if (stream == 0) LUABIND_ERROR("Needs a stream as first argument");
      AprilUtils::SharedPtr<AprilIO::StreamInterface> ptr(stream);
      AprilUtils::LuaTable options(L,2);
      Matrix<T> *obj = Matrix<T>::read(ptr.get(), options);
      if (obj == 0) {
        LUABIND_ERROR("Error happens reading from file stream");
      }
      else {
        AprilUtils::LuaTable::pushInto<Matrix<T>*>(L, obj);
      }
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "fromMMap"
    BEGIN_CLASS_METHOD(fromMMap)
    {
      LUABIND_CHECK_ARGN(>=, 1);
      LUABIND_CHECK_ARGN(<=, 3);
      LUABIND_CHECK_PARAMETER(1, string);
      const char *filename;
      bool write, shared;
      LUABIND_GET_PARAMETER(1,string,filename);
      LUABIND_GET_OPTIONAL_PARAMETER(2,bool,write,true);
      LUABIND_GET_OPTIONAL_PARAMETER(3,bool,shared,true);
      AprilUtils::SharedPtr<AprilUtils::MMappedDataReader> mmapped_data;
      mmapped_data = new AprilUtils::MMappedDataReader(filename,write,shared);
      Matrix<T> *obj = Matrix<T>::fromMMappedDataReader(mmapped_data.get());
      AprilUtils::LuaTable::pushInto<Matrix<T>*>(L, obj);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "toMMap"
    BEGIN_METHOD(toMMap)
    {
      LUABIND_CHECK_ARGN(==, 1);
      const char *filename;
      LUABIND_GET_PARAMETER(1, string, filename);
      AprilUtils::SharedPtr<AprilUtils::MMappedDataWriter> mmapped_data;
      mmapped_data = new AprilUtils::MMappedDataWriter(filename);
      obj->toMMappedDataWriter(mmapped_data.get());
      return 0;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "map"
    BEGIN_METHOD(map)
    {
      int argn;
      int N;
      argn = lua_gettop(L); // number of arguments
      N = argn-1;
      AprilUtils::UniquePtr<Matrix<T>* []> v;
      AprilUtils::UniquePtr<typename Matrix<T>::const_iterator []> list_it;
      if (N > 0) {
        v = new Matrix<T>*[N];
        list_it = new typename Matrix<T>::const_iterator[N];
      }
      for (int i=0; i<N; ++i) {
        if (!AprilUtils::LuaTable::checkType<Matrix<T>*>(L, i+1)) {
          LUABIND_FERROR1("Expected a matrix at position: ", i+1);
        }
        v[i] = AprilUtils::LuaTable::convertTo<Matrix<T>*>(L, i+1);
        if (!v[i]->sameDim(obj)) {
          LUABIND_ERROR("The given matrices must have the same dimension sizes\n");
        }
        list_it[i] = v[i]->begin();
      }
      LUABIND_CHECK_PARAMETER(argn, function);
      for (typename Matrix<T>::iterator it(obj->begin()); it!=obj->end(); ++it) {
        // copy the Lua function, lua_call will pop this copy
        lua_pushvalue(L, argn);
        // push the self matrix value
        AprilUtils::LuaTable::pushInto(L, *it);
        // push the value of the rest of given matrices
        for (int j=0; j<N; ++j) {
          AprilUtils::LuaTable::pushInto(L, *list_it[j]);
          ++list_it[j];
        }
        // CALL
        lua_call(L, N+1, 1);
        // pop the result, a number
        if (!lua_isnil(L, -1)) {
          if (!AprilUtils::LuaTable::checkType<T>(L, -1)) {
            LUABIND_ERROR("Incorrect returned value type");
          }
          *it = AprilUtils::LuaTable::convertTo<T>(L, -1);
        }
        lua_pop(L, 1);
      }
      AprilUtils::LuaTable::pushInto(L, obj);
      return 1;
    }
#undef FUNCTION_NAME

#define FUNCTION_NAME "to_index"
    BEGIN_METHOD(to_index)
    {
      MatrixInt32 *m = AprilMath::MatrixExt::Misc::matNonZeroIndices(obj);
      AprilUtils::LuaTable::pushInto(L, m);
      return 1;
    }
#undef FUNCTION_NAME
    
#define FUNCTION_NAME "equals"
    BEGIN_METHOD(equals)
    {
      Matrix<T> *other;
      float epsilon;
      other = AprilUtils::LuaTable::convertTo<Matrix<T>*>(L,1);
      LUABIND_GET_OPTIONAL_PARAMETER(2, float, epsilon, 0.05f); // 5% error
#ifdef USE_CUDA
      obj->update();
      other->update();
#endif
      if (AprilMath::MatrixExt::Reductions::matEquals(obj, other, epsilon)) {
        lua_pushboolean(L, true);
      }
      else {
        lua_pushboolean(L, false);
      }
      return 1;
    }
#undef FUNCTION_NAME

  };
  
#undef BEGIN_METHOD
#undef BEGIN_CLASS_METHOD
}
//BIND_END

//BIND_STRING_CONSTANT matrix.options.tab Basics::MatrixIO::TAB_OPTION
//BIND_STRING_CONSTANT matrix.options.ascii Basics::MatrixIO::ASCII_OPTION
//BIND_STRING_CONSTANT matrix.options.delim Basics::MatrixIO::DELIM_OPTION
//BIND_STRING_CONSTANT matrix.options.empty Basics::MatrixIO::EMPTY_OPTION
//BIND_STRING_CONSTANT matrix.options.default Basics::MatrixIO::DEFAULT_OPTION
//BIND_STRING_CONSTANT matrix.options.ncols Basics::MatrixIO::NCOLS_OPTION
//BIND_STRING_CONSTANT matrix.options.nrows Basics::MatrixIO::NROWS_OPTION
//BIND_STRING_CONSTANT matrix.options.map Basics::MatrixIO::MAP_OPTION

//BIND_LUACLASSNAME MatrixFloat matrix
//BIND_CPP_CLASS MatrixFloat
//BIND_LUACLASSNAME Serializable aprilio.serializable
//BIND_SUBCLASS_OF MatrixFloat Serializable

//BIND_LUACLASSNAME SlidingWindow matrix.__sliding_window__
//BIND_CPP_CLASS SlidingWindow

//BIND_CONSTRUCTOR SlidingWindow
{
  LUABIND_ERROR("Use matrix.sliding_window");
}
//BIND_END

//BIND_METHOD SlidingWindow get_matrix
{
  MatrixFloat *dest;
  LUABIND_GET_OPTIONAL_PARAMETER(1, MatrixFloat, dest, 0);
  LUABIND_RETURN(MatrixFloat, obj->getMatrix(dest));
}
//BIND_END

//BIND_METHOD SlidingWindow next
{
  LUABIND_RETURN(SlidingWindow, obj->next());
}
//BIND_END

//BIND_METHOD SlidingWindow set_at_window
{
  int windex;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, int, windex);
  if (windex < 1) LUABIND_ERROR("Index must be >= 1\n");
  obj->setAtWindow(windex-1);
  LUABIND_RETURN(SlidingWindow, obj);
}
//BIND_END

//BIND_METHOD SlidingWindow num_windows
{
  LUABIND_RETURN(int, obj->numWindows());
}
//BIND_END

//BIND_METHOD SlidingWindow coords
{
  LUABIND_VECTOR_TO_NEW_TABLE(int, obj->getCoords(), obj->getNumDim());
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD SlidingWindow is_end
{
  LUABIND_RETURN(bool, obj->isEnd());
}
//BIND_END

//BIND_METHOD SlidingWindow iterate
{
  LUABIND_CHECK_ARGN(==, 0);
  LUABIND_RETURN(cfunction,sliding_window_iterator_function);
  LUABIND_RETURN(SlidingWindow,obj);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_CONSTRUCTOR MatrixFloat
//DOC_BEGIN
// matrix(int dim1, int dim2, ..., table mat=nil)
/// Constructor con una secuencia de valores que son las dimensiones de
/// la matriz el ultimo argumento puede ser una tabla, en cuyo caso
/// contiene los valores adecuadamente serializados, si solamente
/// aparece la matriz, se trata de un vector cuya longitud viene dada
/// implicitamente.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::constructor(L));
}
//BIND_END

//BIND_METHOD MatrixFloat size
{
  LUABIND_RETURN(int, obj->size());
}
//BIND_END

//BIND_METHOD MatrixFloat rewrap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::rewrap(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat squeeze
{
  LUABIND_RETURN(MatrixFloat,obj->squeeze());
}
//BIND_END

//BIND_METHOD MatrixFloat get_reference_string
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::
                               get_reference_string(L,obj));
}
//BIND_END


//BIND_METHOD MatrixFloat copy_from_table
//DOC_BEGIN
// void copy_from_table(table matrix_values)
/// Permite dar valores a una matriz. Require una tabla con un numero
/// de argumentos igual al numero de elementos de la matriz.
///@param matrix_values Tabla con los elementos de la matriz.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::
                               copy_from_table(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat get
//DOC_BEGIN
// float get(coordinates)
/// Permite ver valores de una matriz. Requiere tantos indices como dimensiones tenga la matriz.
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat set
//DOC_BEGIN
// float set(coordinates,value)
/// Permite cambiar el valor de un elemento en la matriz. Requiere
/// tantos indices como dimensiones tenga la matriz y adicionalmente
/// el valor a cambiar
///@param coordinates Tabla con la posición exacta del punto de la matriz que queremos obtener.
//DOC_END
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat offset
{
  LUABIND_RETURN(int, obj->getOffset());
}
//BIND_END

//BIND_METHOD MatrixFloat raw_get
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::raw_get(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat raw_set
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::raw_set(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat get_use_cuda
{
  LUABIND_RETURN(bool, obj->getCudaFlag());
}
//BIND_END

//BIND_METHOD MatrixFloat set_use_cuda
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::set_use_cuda(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat dim
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::dim(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat num_dim
{
  LUABIND_RETURN(int, obj->getNumDim());
}
//BIND_END

//BIND_METHOD MatrixFloat stride
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::stride(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat slice
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::slice(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat select
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::select(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat clone
//DOC_BEGIN
// matrix *clone()
/// Devuelve un <em>clon</em> de la matriz.
//DOC_END
{
  LUABIND_RETURN(MatrixFloat, obj->clone());
}
//BIND_END

// returns a matrix with size as the given matrix, but without data copy
//BIND_CLASS_METHOD MatrixFloat as
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::as(L));
}
//BIND_END

//BIND_METHOD MatrixFloat transpose
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::transpose(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat isfinite
//DOC_BEGIN
// bool isfinite
/// Devuelve false si algun valor es nan o infinito.
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  bool resul=true;
  for (MatrixFloat::iterator it(obj->begin()); resul && it!=obj->end(); ++it)
    if ((*it) - (*it) != 0.0f) resul = false;
  LUABIND_RETURN(boolean,resul);
}
//BIND_END

//BIND_METHOD MatrixFloat toTable
// Permite salvar una matriz en una tabla lua
// TODO: Tener en cuenta las dimensiones de la matriz
  {
    LUABIND_CHECK_ARGN(==, 0);
    LUABIND_FORWARD_CONTAINER_TO_NEW_TABLE(MatrixFloat, float, *obj);
    LUABIND_INCREASE_NUM_RETURNS(1);
  }
//BIND_END

//BIND_METHOD MatrixFloat contiguous
{
  if (obj->getIsContiguous()) LUABIND_RETURN(MatrixFloat, obj);
  else LUABIND_RETURN(MatrixFloat, obj->clone());
}
//BIND_END

//BIND_METHOD MatrixFloat map
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::map(L, obj));
}
//BIND_END

//BIND_METHOD MatrixFloat diagonalize
{
#ifdef USE_CUDA
  obj->update();
#endif
  MatrixFloat *resul = obj->diagonalize();
  LUABIND_RETURN(MatrixFloat, resul);
}
//BIND_END

//BIND_METHOD MatrixFloat get_shared_count
{
  LUABIND_RETURN(uint, obj->getSharedCount());
}
//BIND_END

//BIND_METHOD MatrixFloat reset_shared_count
{
  obj->resetSharedCount();
}
//BIND_END

//BIND_METHOD MatrixFloat add_to_shared_count
{
  unsigned int count;
  LUABIND_GET_PARAMETER(1,uint,count);
  obj->addToSharedCount(count);
}
//BIND_END

//BIND_METHOD MatrixFloat update
{
  obj->update();
}
//BIND_END

//BIND_METHOD MatrixFloat padding_all
{
  int padding;
  LUABIND_GET_PARAMETER(1, int, padding);
  float default_value;
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, default_value, 0.0f);
  MatrixFloat *result = obj->padding(padding, default_value);
  LUABIND_RETURN(MatrixFloat, result);
}
//BIND_END

//BIND_METHOD MatrixFloat padding
{
  AprilUtils::UniquePtr<int []> begin_padding, end_padding;
  LUABIND_CHECK_ARGN(>=,obj->getNumDim()*2);
  LUABIND_CHECK_ARGN(<=,obj->getNumDim()*2 + 1);
  begin_padding = new int[obj->getNumDim()];
  end_padding = new int[obj->getNumDim()];
  int j=1;
  for (int i=0; i<obj->getNumDim(); ++i, j+=2) {
    LUABIND_GET_PARAMETER(j, int, begin_padding[i]);
    LUABIND_GET_PARAMETER(j+1, int, end_padding[i]);
  }
  float default_value;
  LUABIND_GET_OPTIONAL_PARAMETER(j, float, default_value, 0.0f);
  MatrixFloat *result = obj->padding(begin_padding.get(),
                                     end_padding.get(),
                                     default_value);
  LUABIND_RETURN(MatrixFloat, result);
}
//BIND_END

//BIND_METHOD MatrixFloat uniform
{
  int lower, upper;
  MTRand *random;
  LUABIND_GET_PARAMETER(1, int, lower);
  LUABIND_GET_PARAMETER(2, int, upper);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MTRand, random, 0);
  
  if (lower > upper)
    LUABIND_ERROR("First argument must be <= second argument");
  if (random == 0) random = new MTRand();
  IncRef(random);
  for (MatrixFloat::iterator it(obj->begin()); it != obj->end(); ++it) {
    *it = static_cast<float>(random->randInt(upper - lower)) + lower;
  }
  DecRef(random);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat uniformf
{
  float lower, upper;
  MTRand *random;
  LUABIND_GET_OPTIONAL_PARAMETER(1, float, lower, 0.0f);
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, upper, 1.0f);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MTRand, random, 0);
  if (lower > upper)
    LUABIND_ERROR("First argument must be <= second argument");
  if (random == 0) random = new MTRand();
  IncRef(random);
  for (MatrixFloat::iterator it(obj->begin()); it != obj->end(); ++it)
    *it = random->rand(upper - lower) + lower;
  DecRef(random);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat linspace
{
  int size_1 = obj->size()-1;
  float inf,sup;
  LUABIND_GET_OPTIONAL_PARAMETER(1, float, inf, 1.0f);
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, sup, static_cast<float>(size_1+1));
  int i = 0;
  float diff = sup-inf;
  if (diff == size_1) {
    i = static_cast<int>(inf);
    for (MatrixFloat::iterator it(obj->begin()); it!=obj->end(); ++it, ++i) {
      april_assert(i <= static_cast<int>(sup));
      *it = static_cast<float>(i);
    }
  }
  else {
    for (MatrixFloat::iterator it(obj->begin()); it!=obj->end(); ++it, ++i) {
      *it = (diff*i)/size_1 + inf;
    }
  }
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat logspace
{
  int size = obj->size()-1;
  float inf,sup,base;
  LUABIND_GET_OPTIONAL_PARAMETER(3, float, base, 10.0f);
  LUABIND_GET_OPTIONAL_PARAMETER(1, float, inf, 1.0f);
  LUABIND_GET_OPTIONAL_PARAMETER(2, float, sup, size+1);
  int i=0;
  inf = logf(inf)/logf(base);
  sup = logf(sup)/logf(base);
  float diff = sup-inf;
  for (MatrixFloat::iterator it(obj->begin()); it!=obj->end(); ++it, ++i)
    *it = powf(base, (diff*i)/size + inf);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat linear
{
  int lower, step;
  MTRand *random;
  LUABIND_GET_OPTIONAL_PARAMETER(1, int, lower, 0);
  LUABIND_GET_OPTIONAL_PARAMETER(2, int, step,  1);
  int k=lower;
  for (MatrixFloat::iterator it(obj->begin()); it != obj->end(); ++it, k+=step) {
    *it = static_cast<float>(k);
  }
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat sliding_window
{
  AprilUtils::UniquePtr<int []> sub_matrix_size, offset, step, num_steps, order_step;
  int argn = lua_gettop(L); // number of arguments
  const int num_dim = obj->getNumDim();
  if (argn > 1)
    LUABIND_ERROR("incorrect number of arguments");
  if (argn == 1) {
    LUABIND_CHECK_PARAMETER(1, table);
    check_table_fields(L, 1,
                       "offset",
                       "size",
                       "step",
                       "numSteps",
                       "orderStep",
                       (const char*)0);
    
    offset = read_vector(L, "offset", num_dim, 0);
    sub_matrix_size = read_vector(L, "size", num_dim, 0);
    step = read_vector(L, "step", num_dim, 0);
    num_steps = read_vector(L, "numSteps", num_dim, 0);
    order_step = read_vector(L, "orderStep", num_dim, -1);
  }
  SlidingWindow *window = new SlidingWindow(obj,
                                            sub_matrix_size.get(),
                                            offset.get(),
                                            step.get(),
                                            num_steps.get(),
                                            order_step.get());
  LUABIND_RETURN(SlidingWindow, window);
}
//BIND_END

//BIND_METHOD MatrixFloat is_contiguous
{
  LUABIND_RETURN(bool, obj->getIsContiguous());
}
//BIND_END

//BIND_METHOD MatrixFloat prune_subnormal_and_check_normal
{
  obj->pruneSubnormalAndCheckNormal();
}
//BIND_END

////////////////// MATH EXTENSIONS //////////////////

//BIND_METHOD MatrixFloat adjust_range
//DOC_BEGIN
// void adjust_range(float min, float max)
/// Ajusta el rango de valores de la matriz para que estÃ© en [min,
/// max].
//DOC_END
{
  float rmin,rmax;
  LUABIND_CHECK_ARGN(==, 2);
  LUABIND_CHECK_PARAMETER(1, float);
  LUABIND_CHECK_PARAMETER(2, float);
  LUABIND_GET_PARAMETER(1,float,rmin);
  LUABIND_GET_PARAMETER(2,float,rmax);
  LUABIND_RETURN(MatrixFloat,
                 matAdjustRange(obj, rmin, rmax));
}
//BIND_END

//BIND_METHOD MatrixFloat diag
{
  LUABIND_CHECK_ARGN(==,1);
  float v;
  LUABIND_GET_PARAMETER(1, float, v);
  matDiag(obj, v);
  LUABIND_RETURN(MatrixFloat, obj);
}
//BIND_END

//BIND_METHOD MatrixFloat fill
{
  LUABIND_CHECK_ARGN(==, 1);
  float value;
  if (lua_isMatrixFloat(L, 1)) {
    MatrixFloat *aux;
    LUABIND_GET_PARAMETER(1,MatrixFloat,aux);
    for (int i=0; i<aux->getNumDim(); ++i)
      if (aux->getDimSize(i) != 1)
	LUABIND_ERROR("Needs a float or a matrix with only one element\n");
    value = *(aux->begin());
  }
  else {
    LUABIND_CHECK_PARAMETER(1, float);
    LUABIND_GET_PARAMETER(1,float,value);
  }
  LUABIND_RETURN(MatrixFloat, 
                 matFill(obj,value));
}
//BIND_END

//BIND_METHOD MatrixFloat zeros
{
  LUABIND_RETURN(MatrixFloat, 
                 matZeros(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat ones
//DOC_BEGIN
// void onex(float value)
/// Permite poner todos los valores de la matriz a un mismo valor.
//DOC_END
{
  LUABIND_RETURN(MatrixFloat, 
                 matOnes(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat min
{
#ifdef USE_CUDA
  obj->update();
#endif
  LUABIND_CHECK_ARGN(>=,0);
  LUABIND_CHECK_ARGN(<=,3);
  int argn = lua_gettop(L);
  if (argn > 0) {
    int dim;
    MatrixFloat *dest;
    MatrixInt32 *argmin;
    LUABIND_GET_PARAMETER(1, int, dim);
    LUABIND_GET_OPTIONAL_PARAMETER(2, MatrixFloat, dest, 0);
    LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixInt32, argmin, 0);
    AprilUtils::UniquePtr<int []> aux;
    if (argmin == 0) {
      aux = new int[obj->getNumDim()];
      for (int i=0; i<obj->getNumDim(); ++i) aux[i] = obj->getDimSize(i);
      aux[dim-1] = 1;
      argmin = new MatrixInt32(obj->getNumDim(), aux.get());
    }
    IncRef(argmin);
    if (dim < 1 || dim > obj->getNumDim())
      LUABIND_FERROR2("Incorrect dimension, found %d, expect in [1,%d]",
                      dim, obj->getNumDim());
    LUABIND_RETURN(MatrixFloat, 
                   matMin(obj, dim-1, dest, argmin));
    LUABIND_RETURN(MatrixInt32, argmin);
    DecRef(argmin);
  }
  else {
    int arg_min, raw_pos;
    LUABIND_RETURN(float, 
                   matMin(obj, arg_min, raw_pos));
    LUABIND_RETURN(int, arg_min+1);
  }
}
//BIND_END

//BIND_METHOD MatrixFloat max
{
#ifdef USE_CUDA
  obj->update();
#endif
  LUABIND_CHECK_ARGN(>=,0);
  LUABIND_CHECK_ARGN(<=,3);
  int argn = lua_gettop(L);
  if (argn > 0) {
    int dim;
    MatrixFloat *dest;
    MatrixInt32 *argmax;
    LUABIND_GET_PARAMETER(1, int, dim);
    LUABIND_GET_OPTIONAL_PARAMETER(2, MatrixFloat, dest, 0);
    LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixInt32, argmax, 0);
    AprilUtils::UniquePtr<int []> aux;
    if (argmax == 0) {
      aux = new int[obj->getNumDim()];
      for (int i=0; i<obj->getNumDim(); ++i) aux[i] = obj->getDimSize(i);
      aux[dim-1] = 1;
      argmax = new MatrixInt32(obj->getNumDim(), aux.get());
    }
    IncRef(argmax);
    if (dim < 1 || dim > obj->getNumDim())
      LUABIND_FERROR2("Incorrect dimension, found %d, expect in [1,%d]",
                      dim, obj->getNumDim());
    LUABIND_RETURN(MatrixFloat, 
                   matMax(obj, dim-1, dest, argmax));
    LUABIND_RETURN(MatrixInt32, argmax);
    DecRef(argmax);
  }
  else {
    int arg_max, raw_pos;
    LUABIND_RETURN(float, 
                   matMax(obj, arg_max, raw_pos));
    LUABIND_RETURN(int, arg_max+1);
  }
}
//BIND_END

//BIND_METHOD MatrixFloat equals
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::equals(L,obj));
}
//BIND_END

//BIND_METHOD MatrixFloat clamp
{
  LUABIND_CHECK_ARGN(==, 2);
  float lower,upper;
  LUABIND_GET_PARAMETER(1, float, lower);
  LUABIND_GET_PARAMETER(2, float, upper);
  LUABIND_RETURN(MatrixFloat, 
                 matClamp(obj,lower,upper));
}
//BIND_END

//BIND_METHOD MatrixFloat add
{
  int argn;
  argn = lua_gettop(L); // number of arguments
  LUABIND_CHECK_ARGN(==, 1);
  MatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
  if (!obj->sameDim(mat)) {
    LUABIND_ERROR("matrix add wrong dimensions");
  }
#ifdef USE_CUDA
  mat->update();
#endif
  LUABIND_RETURN(MatrixFloat, 
                 matAddition(obj, mat));
}
//BIND_END

//BIND_METHOD MatrixFloat scalar_add
{
  int argn;
  argn = lua_gettop(L); // number of arguments
  LUABIND_CHECK_ARGN(==, 1);
  float scalar;
  LUABIND_GET_PARAMETER(1, float, scalar);
  LUABIND_RETURN(MatrixFloat, 
                 matScalarAdd(obj, scalar));
}
//BIND_END

//BIND_METHOD MatrixFloat sub
{
  LUABIND_CHECK_ARGN(==, 1);
  MatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
  if (!obj->sameDim(mat))
    LUABIND_ERROR("matrix sub wrong dimensions");
#ifdef USE_CUDA
  mat->update();
#endif
  LUABIND_RETURN(MatrixFloat, 
                 matSubstraction(obj, mat));
}
//BIND_END

//BIND_METHOD MatrixFloat mul
{
  LUABIND_CHECK_ARGN(==, 1);
  MatrixFloat *mat,*resul;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
#ifdef USE_CUDA
  mat->update();
#endif
  LUABIND_RETURN(MatrixFloat, 
                 matMultiply(obj, mat));
}
//BIND_END

//BIND_METHOD MatrixFloat cmul
{
  LUABIND_CHECK_ARGN(==, 1);
  MatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
#ifdef USE_CUDA
  mat->update();
#endif
  LUABIND_RETURN(MatrixFloat, 
                 matCmul(obj, mat));
}
//BIND_END

//BIND_METHOD MatrixFloat plogp
{
  LUABIND_RETURN(MatrixFloat, 
                 matPlogp(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat log
{
  LUABIND_RETURN(MatrixFloat, 
                 matLog(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat log1p
{
  LUABIND_RETURN(MatrixFloat, 
                 matLog1p(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat exp
{
  LUABIND_RETURN(MatrixFloat, 
                 matExp(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sqrt
{
  LUABIND_RETURN(MatrixFloat, 
                 matSqrt(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat pow
{
  float value;
  LUABIND_CHECK_ARGN(==,1);
  LUABIND_GET_PARAMETER(1, float, value);
  LUABIND_RETURN(MatrixFloat, 
                 matPow(obj, value));
}
//BIND_END

//BIND_METHOD MatrixFloat tan
{
  LUABIND_RETURN(MatrixFloat, 
                 matTan(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat tanh
{
  LUABIND_RETURN(MatrixFloat, 
                 matTanh(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat atan
{
  LUABIND_RETURN(MatrixFloat, 
                 matAtan(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat atanh
{
  LUABIND_RETURN(MatrixFloat, 
                 matAtanh(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sin
{
  LUABIND_RETURN(MatrixFloat, 
                 matSin(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sinh
{
  LUABIND_RETURN(MatrixFloat, 
                 matSinh(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat asin
{
  LUABIND_RETURN(MatrixFloat, 
                 matAsin(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat asinh
{
  LUABIND_RETURN(MatrixFloat, 
                 matAsinh(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat cos
{
  LUABIND_RETURN(MatrixFloat, 
                 matCos(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat cosh
{
  LUABIND_RETURN(MatrixFloat, 
                 matCosh(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat acos
{
  LUABIND_RETURN(MatrixFloat, 
                 matAcos(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat acosh
{
  LUABIND_RETURN(MatrixFloat, 
                 matAcosh(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat abs
{
  LUABIND_RETURN(MatrixFloat, 
                 matAbs(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat complement
{
  LUABIND_RETURN(MatrixFloat, 
                 matComplement(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sign
{
  LUABIND_RETURN(MatrixFloat, 
                 matSign(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat sum
{
#ifdef USE_CUDA
  obj->update();
#endif
  LUABIND_CHECK_ARGN(>=, 0);
  LUABIND_CHECK_ARGN(<=, 2);
  int argn = lua_gettop(L); // number of arguments
  if (argn > 0 && !lua_isnil(L,1)) {
    int dim;
    LUABIND_GET_PARAMETER(1, int, dim);
    MatrixFloat *dest;
    LUABIND_GET_OPTIONAL_PARAMETER(2, MatrixFloat, dest, 0);
    if (dim < 1 || dim > obj->getNumDim())
      LUABIND_FERROR2("Incorrect dimension, found %d, expect in [1,%d]",
                      dim, obj->getNumDim());
    LUABIND_RETURN(MatrixFloat, 
                   matSum(obj, dim-1, dest));
  }
  else {
    LUABIND_RETURN(float, 
                   matSum(obj));
  }
}
//BIND_END

//BIND_METHOD MatrixFloat copy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 1);
  MatrixFloat *mat;
  LUABIND_GET_PARAMETER(1, MatrixFloat, mat);
#ifdef USE_CUDA
  mat->update();
#endif
  LUABIND_RETURN(MatrixFloat, 
                 matCopy(obj,mat));
}
//BIND_END

//BIND_METHOD MatrixFloat axpy
{
  int argn;
  LUABIND_CHECK_ARGN(==, 2);
  float alpha;
  LUABIND_GET_PARAMETER(1, float, alpha);
  if (lua_isMatrixFloat(L,2)) {
    MatrixFloat *mat;
    LUABIND_GET_PARAMETER(2, MatrixFloat, mat);
#ifdef USE_CUDA
    mat->update();
#endif
    LUABIND_RETURN(MatrixFloat, 
                   matAxpy(obj, alpha, mat));
  }
  else if (lua_isSparseMatrixFloat(L,2)) {
    SparseMatrixFloat *mat;
    LUABIND_GET_PARAMETER(2, SparseMatrixFloat, mat);
    LUABIND_RETURN(MatrixFloat, 
                   matAxpy(obj, alpha, mat));
  }
  else {
    LUABIND_ERROR("Expected matrix or matrix.sparse as 2nd argument");
  }
}
//BIND_END

//BIND_METHOD MatrixFloat gemm
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L,1, "trans_A", "trans_B", "alpha", "A", "B", "beta",
                     (const char *)0);
  bool trans_A, trans_B;
  float alpha;
  float beta;
  MatrixFloat *matA,*matB;
  LUABIND_GET_TABLE_PARAMETER(1, A, MatrixFloat, matA);
  LUABIND_GET_TABLE_PARAMETER(1, B, MatrixFloat, matB);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_A, bool, trans_A, false);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_B, bool, trans_B, false);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, alpha, float, alpha, 1.0f);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, beta, float, beta, 1.0f);
#ifdef USE_CUDA
  matA->update();
  matB->update();
#endif
  LUABIND_RETURN(MatrixFloat,
                 
                 matGemm(obj,
                         trans_A ? CblasTrans : CblasNoTrans,
                         trans_B ? CblasTrans : CblasNoTrans,
                         alpha, matA, matB,
                         beta));
}
//BIND_END

//BIND_METHOD MatrixFloat sparse_mm
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L,1, "trans_A", "trans_B",
                     "alpha", "A", "B", "beta",
                     (const char *)0);
  bool trans_A, trans_B;
  float alpha;
  float beta;
  SparseMatrixFloat *matA;
  MatrixFloat *matB;
  LUABIND_GET_TABLE_PARAMETER(1, A, SparseMatrixFloat, matA);
  LUABIND_GET_TABLE_PARAMETER(1, B, MatrixFloat, matB);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_A, bool, trans_A, false);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_B, bool, trans_B, false);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, alpha, float, alpha, 1.0f);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, beta, float, beta, 1.0f);
  LUABIND_RETURN(MatrixFloat,
                 
                 matSparseMM(obj,
                             trans_A ? CblasTrans : CblasNoTrans,
                             trans_B ? CblasTrans : CblasNoTrans,
                             alpha, matA, matB,
                             beta));
}
//BIND_END

//BIND_METHOD MatrixFloat gemv
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L,1, "trans_A", "alpha", "A", "X", "beta",
                     (const char *)0);
  bool trans_A;
  float alpha;
  float beta;
  MatrixFloat *matX;
  LUABIND_GET_TABLE_PARAMETER(1, X, MatrixFloat, matX);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_A, bool, trans_A, false);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, alpha, float, alpha, 1.0f);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, beta, float, beta, 1.0f);
    
  lua_getfield(L, 1, "A");
  if (lua_isMatrixFloat(L,-1)) {
    lua_pop(L,1);
    MatrixFloat *matA;
    LUABIND_GET_TABLE_PARAMETER(1, A, MatrixFloat, matA);
#ifdef USE_CUDA
    matA->update();
    matX->update();
#endif
    LUABIND_RETURN(MatrixFloat,
                   
                   matGemv(obj,
                           trans_A ? CblasTrans : CblasNoTrans,
                           alpha, matA, matX,
                           beta));
  }
  else {
    lua_pop(L,1);
    SparseMatrixFloat *matA;
    LUABIND_GET_TABLE_PARAMETER(1, A, SparseMatrixFloat, matA);
    LUABIND_RETURN(MatrixFloat,
                   
                   matGemv(obj,
                           trans_A ? CblasTrans : CblasNoTrans,
                           alpha, matA, matX,
                           beta));
  }
}
//BIND_END

//BIND_METHOD MatrixFloat ger
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  check_table_fields(L,1, "alpha", "X", "Y",
                     (const char *)0);
  float alpha;
  MatrixFloat *matX,*matY;
  LUABIND_GET_TABLE_PARAMETER(1, X, MatrixFloat, matX);
  LUABIND_GET_TABLE_PARAMETER(1, Y, MatrixFloat, matY);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, alpha, float, alpha, 1.0f);
#ifdef USE_CUDA
  matX->update();
  matY->update();
#endif
  LUABIND_RETURN(MatrixFloat, 
                 matGer(obj, alpha, matX, matY));
}
//BIND_END

//BIND_METHOD MatrixFloat dot
{
  LUABIND_CHECK_ARGN(==, 1);
  if (lua_isMatrixFloat(L,1)) {
    LUABIND_CHECK_PARAMETER(1, MatrixFloat);
    MatrixFloat *matX;
    LUABIND_GET_PARAMETER(1, MatrixFloat, matX);
#ifdef USE_CUDA
    obj->update();
    matX->update();
#endif
    LUABIND_RETURN(float, 
                   matDot(obj, matX));
  }
  else if (lua_isSparseMatrixFloat(L,1)) {
    LUABIND_CHECK_PARAMETER(1, SparseMatrixFloat);
    SparseMatrixFloat *matX;
    LUABIND_GET_PARAMETER(1, SparseMatrixFloat, matX);
#ifdef USE_CUDA
    obj->update();
    matX->update();
#endif
    LUABIND_RETURN(float, 
                   matDot(obj, matX));
  }
}
//BIND_END

//BIND_METHOD MatrixFloat scal
{
  LUABIND_CHECK_ARGN(==, 1);
  float value;
  LUABIND_GET_PARAMETER(1, float, value);
  LUABIND_RETURN(MatrixFloat, 
                 matScal(obj, value));
}
//BIND_END

//BIND_METHOD MatrixFloat masked_fill
{
  float value;
  MatrixBool *mask;
  MatrixFloat *dest;
  LUABIND_GET_PARAMETER(1, MatrixBool, mask);
  LUABIND_GET_PARAMETER(2, float, value);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixFloat, dest, 0);
  matMaskedFill(obj, mask, value, dest);
}
//BIND_END

//BIND_METHOD MatrixFloat masked_copy
{
  MatrixFloat *value;
  MatrixBool *mask;
  MatrixFloat *dest;
  LUABIND_GET_PARAMETER(1, MatrixBool, mask);
  LUABIND_GET_PARAMETER(2, MatrixFloat, value);
  LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixFloat, dest, 0);
  matMaskedCopy(obj, mask, value, dest);
}
//BIND_END

//BIND_METHOD MatrixFloat div
{
  LUABIND_CHECK_ARGN(==, 1);
  if (lua_isMatrixFloat(L,1)) {
    MatrixFloat *other;
    LUABIND_GET_PARAMETER(1, MatrixFloat, other);
    LUABIND_RETURN(MatrixFloat, 
                   matDiv(obj, other));
  }
  else {
    float value;
    LUABIND_GET_PARAMETER(1, float, value);
    LUABIND_RETURN(MatrixFloat, 
                   matDiv(obj, value));
  }
}
//BIND_END
 
//BIND_METHOD MatrixFloat norm2
{
#ifdef USE_CUDA
  obj->update();
#endif
  LUABIND_RETURN(float, 
                 matNorm2(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat inv
{
  LUABIND_RETURN(MatrixFloat, 
                 matInv(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat logdet
{
  float sign;
  LUABIND_RETURN(float, 
                 matLogDeterminant(obj, sign).log());
  LUABIND_RETURN(float, sign);
}
//BIND_END

//BIND_METHOD MatrixFloat det
{
  LUABIND_RETURN(double, 
                 matDeterminant(obj));
}
//BIND_END

//BIND_METHOD MatrixFloat cholesky
{
  char uplo;
  LUABIND_GET_OPTIONAL_PARAMETER(1, char, uplo, 'U');
  uplo = toupper(uplo);
  if (uplo != 'U' && uplo != 'L') {
    LUABIND_ERROR("Incorrect argument, expected character L or U");
  }
  LUABIND_RETURN(MatrixFloat, 
                 matCholesky(obj, uplo));
}
//BIND_END

//BIND_METHOD MatrixFloat svd
{
  MatrixFloat *U,*V;
  SparseMatrixFloat *S;
  matSVD(obj, &U, &S, &V);
  LUABIND_RETURN(MatrixFloat, U);
  LUABIND_RETURN(SparseMatrixFloat, S);
  LUABIND_RETURN(MatrixFloat, V);
}
//BIND_END

//BIND_METHOD MatrixFloat lt
{
  if (lua_isMatrixFloat(L, 1)) {
    MatrixFloat *value;
    LUABIND_GET_PARAMETER(1, MatrixFloat, value);
    LUABIND_RETURN(MatrixBool, 
                   matLT(obj, value));
  }
  else {
    float value;
    LUABIND_GET_PARAMETER(1, float, value);
    LUABIND_RETURN(MatrixBool, 
                   matLT(obj, value));
  }
}
//BIND_END

//BIND_METHOD MatrixFloat gt
{
  if (lua_isMatrixFloat(L, 1)) {
    MatrixFloat *value;
    LUABIND_GET_PARAMETER(1, MatrixFloat, value);
    LUABIND_RETURN(MatrixBool, 
                   matGT(obj, value));
  }
  else {
    float value;
    LUABIND_GET_PARAMETER(1, float, value);
    LUABIND_RETURN(MatrixBool, 
                   matGT(obj, value));
  }
}
//BIND_END

//BIND_METHOD MatrixFloat eq
{
  if (lua_isMatrixFloat(L, 1)) {
    MatrixFloat *value;
    LUABIND_GET_PARAMETER(1, MatrixFloat, value);
    LUABIND_RETURN(MatrixBool, 
                   matEQ(obj, value));
  }
  else {
    float value;
    LUABIND_GET_PARAMETER(1, float, value);
    LUABIND_RETURN(MatrixBool, 
                   matEQ(obj, value));
  }
}
//BIND_END

//BIND_METHOD MatrixFloat neq
{
  if (lua_isMatrixFloat(L, 1)) {
    MatrixFloat *value;
    LUABIND_GET_PARAMETER(1, MatrixFloat, value);
    LUABIND_RETURN(MatrixBool, 
                   matNEQ(obj, value));
  }
  else {
    float value;
    LUABIND_GET_PARAMETER(1, float, value);
    LUABIND_RETURN(MatrixBool, 
                   matNEQ(obj, value));
    
  }
}
//BIND_END

///////////////////////////////////////////////////////////////////////

//BIND_FUNCTION matrix.ext.convolution
{
  MatrixFloat *obj, *kernel, *result; //, *unrolled_kernel, *unrolled_self;
  AprilUtils::UniquePtr<int []> step;
  int D;
  
  LUABIND_CHECK_ARGN(>=, 2);
  LUABIND_CHECK_ARGN(<=, 3);
  LUABIND_GET_PARAMETER(1, MatrixFloat, obj);
  LUABIND_CHECK_PARAMETER(2, table);
  LUABIND_GET_TABLE_PARAMETER(2, D, int, D);
  LUABIND_GET_TABLE_PARAMETER(2, kernel, MatrixFloat, kernel);
  /*
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, unrolled_kernel,
    MatrixFloat, unrolled_kernel, 0);
    LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, unrolled_self,
    MatrixFloat, unrolled_self, 0);
  */
  LUABIND_GET_OPTIONAL_PARAMETER(3, MatrixFloat, result, 0);
  lua_getfield(L, 2, "step");
  if (!lua_isnil(L, -1)) {
    step = new int[D];
    int len;
    LUABIND_TABLE_GETN(-1, len);
    if (len != D) {
      LUABIND_FERROR2("Incorrect length of step table, found %d, expected %d",
                      len, D);
    }
    LUABIND_TABLE_TO_VECTOR(-1, int, step.get(), D);
  }
  lua_pop(L, 1);
  
  LUABIND_RETURN(MatrixFloat,
                 
                 matConvolution(obj, D, step.get(), kernel, result));
  //&unrolled_kernel, &unrolled_self);
  /*LUABIND_RETURN(MatrixFloat, unrolled_kernel);
    LUABIND_RETURN(MatrixFloat, unrolled_self);*/
}
//BIND_END

//BIND_FUNCTION matrix.ext.real_fftwh
{
  MatrixFloat *obj, *dest;
  int wsize, wadvance;
  LUABIND_GET_PARAMETER(1, MatrixFloat, obj);
  LUABIND_GET_OPTIONAL_PARAMETER(2, int, wsize, obj->size());
  LUABIND_GET_OPTIONAL_PARAMETER(3, int, wadvance, wsize);
  LUABIND_GET_OPTIONAL_PARAMETER(4, MatrixFloat, dest, 0);
  LUABIND_RETURN(MatrixFloat,
		 
		 matRealFFTwithHamming(obj, wsize, wadvance, dest));
}
//BIND_END

//// MATRIX SERIALIZATION ////

//BIND_CLASS_METHOD MatrixFloat deserialize
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::deserialize(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixFloat read
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::read(L));
}
//BIND_END

//BIND_CLASS_METHOD MatrixFloat fromMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::fromMMap(L));
}
//BIND_END

//BIND_METHOD MatrixFloat toMMap
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::toMMap(L, obj));
}
//BIND_END

//BIND_CLASS_METHOD MatrixFloat fromPNM
//DOC_BEGIN
// matrix *fromPNM(string pnm_image)
/// constructor con un argumento que es una cadena con una imagen en
/// formato de netpbm P5 o P6 (binario PGM o PNM)
///@param pnm_image String que contiene la imagen.
//DOC_END
// TODO: poder forzar niveles de gris o color, poder leer PBM
{
  LUABIND_CHECK_ARGN(>=, 1);
  LUABIND_CHECK_ARGN(<=, 2);
  LUABIND_CHECK_PARAMETER(1, string);
  bool forcecolor=false,forcegray=false;
  AprilUtils::constString cs,csopt;
  LUABIND_GET_PARAMETER(1,constString,cs);
  LUABIND_GET_OPTIONAL_PARAMETER(2,constString,csopt,
                                 AprilUtils::constString());
  if (csopt == "color") forcecolor = true;
  if (csopt == "gray")  forcegray  = true;
  MatrixFloat *obj;
  if ((obj = readMatrixFloatPNM(cs,forcecolor,forcegray))== 0)
    LUABIND_ERROR("bad format");
  else LUABIND_RETURN(MatrixFloat,obj);
}
//BIND_END

//BIND_CLASS_METHOD MatrixFloat fromHEX
//DOC_BEGIN
// matrix *fromHEX(width, height, string hex_image)
/// constructor con 3 argumentos que es una cadena con una imagen en
/// escala de grises, 2 caracteres hexadecimales por pixel
///@param width
///@param height
///@param hex_image
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 3);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(1, int);
  LUABIND_CHECK_PARAMETER(1, string);
  int width,height;
  AprilUtils::constString cs;
  LUABIND_GET_PARAMETER(1,int,width);
  LUABIND_GET_PARAMETER(2,int,height);
  LUABIND_GET_PARAMETER(3,constString,cs);
  MatrixFloat *obj;
  obj = readMatrixFloatHEX(width,height,cs);
  LUABIND_RETURN(MatrixFloat,obj);
}
//BIND_END

//BIND_METHOD MatrixFloat toHEX
//DOC_BEGIN
// string toHEX()
//DOC_END
{
  char *buffer;
  int   width, height;
  int   longitud = saveMatrixFloatHEX(obj,&buffer, &width, &height);
  if (!buffer) {
    LUABIND_ERROR("bad format");
  }
  LUABIND_RETURN(int, width);
  LUABIND_RETURN(int, height);
  lua_pushlstring(L,buffer,longitud);
  delete[] buffer;
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD MatrixFloat toPNM
//DOC_BEGIN
// string toPNM()
/// Devuelve una cadena correspondiente a un fichero PNM (P5 o P6).  La
/// matriz debe ser de dimension 2 o, si es de dimension 3, la tercera
/// dimension debe tener 3 componentes correspondientes respectivamente
/// a los colores RGB. El 0 se interpreta como negro, el 1 como blanco
/// y saturan (es decir, un -1 es como 0 y un 5 es como 1).
//DOC_END
{
  LUABIND_CHECK_ARGN(==, 0);
  char *buffer;
  int longitud = saveMatrixFloatPNM(obj,&buffer);
  if (!buffer)
    LUABIND_ERROR("bad format");
  lua_pushlstring(L,buffer,longitud);
  delete[] buffer;
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

//BIND_METHOD MatrixFloat data
{
  LUABIND_RETURN(FloatGPUMirroredMemoryBlock, obj->getRawDataAccess());
}
//BIND_END

//BIND_METHOD MatrixFloat order
{
  MatrixInt32 *dest;
  LUABIND_GET_OPTIONAL_PARAMETER(1, MatrixInt32, dest, 0);
  dest = matOrder(obj, dest);
  LUABIND_RETURN(MatrixInt32, dest);
}
//BIND_END

//BIND_METHOD MatrixFloat order_rank
{
  MatrixInt32 *dest;
  LUABIND_GET_OPTIONAL_PARAMETER(1, MatrixInt32, dest, 0);
  dest = matOrderRank(obj, dest);
  LUABIND_RETURN(MatrixInt32, dest);
}
//BIND_END

//BIND_METHOD MatrixFloat convert_to
{
  LUABIND_INCREASE_NUM_RETURNS(MatrixBindings<float>::convert_to(L,obj));
}
//BIND_END
