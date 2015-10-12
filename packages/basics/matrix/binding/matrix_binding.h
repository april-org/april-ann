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
#ifndef MATRIX_BINDING_H
#define MATRIX_BINDING_H
extern "C" {
#include <ctype.h>
#include <lua.h>
}
#include <typeinfo>

#include "complex_number.h"
#include "bind_april_io.h"
#include "bind_mtrand.h"
#include "gpu_mirrored_memory_block.h"
#include "matrixFloat.h"
#include "mystring.h"
#include "luabindmacros.h"
#include "luabindutil.h"
#include "utilLua.h"

namespace Basics {

  /// Implements binding functions reusable in different Matrix flavors.
  template<typename T>
  class MatrixBindings {

    static void parse_slice(const char *slice,
                            int &a,
                            int &b,
                            const int min,
                            const int max) {
      /* Impmentation in C of:
         a = slice:match("^([+-]?%d+)%:.*$") or min
         b = slice:match("^.*%:([+-]?%d+)$") or max
      */
      bool neg_a = false, neg_b = false;
      // [+-]?
      if ( *slice == '+' || *slice == '-') {
        if ( *slice == '-') neg_a = true;
        ++slice;
      }
      // ^:
      else if ( *slice == ':') {
        a = min;
        ++slice;
      }
      // %d+:
      else {
        const char *next = strchr(slice, ':');
        if (next == NULL) ERROR_EXIT(256, "Unable to locate ':' character\n");
        a = atoi(slice);
        slice = next+1;
      }
      // [+-]?
      if ( *slice == '+' || *slice == '-') {
        if ( *slice == '-' ) neg_b = true;
        ++slice;
      }
      // empty string
      else if ( *slice == '\0' ) {
        b = max;
      }
      // %d+
      else {
        b = atoi(slice);
      }
      if (neg_a) a = -a;
      if (neg_b) b = -b;
    }
    
    static void extract_range(lua_State *L,
                              const int n,
                              const int k, 
                              int &a,
                              int &b,
                              const int min,
                              const int max) {
      int tt = lua_type(L, n);
      if (tt == LUA_TTABLE) {
        const int len = lua_rawlen(L, n);
        if (len == 0) {
          a = min;
          b = max;
        }
        else if (len == 2) {
          lua_rawgeti(L, n, 1);
          a = lua_toint(L, -1);
          lua_pop(L, 1);
          lua_rawgeti(L, n, 2);
          b = lua_toint(L, -1);
          lua_pop(L, 1);
        }
        else {
          ERROR_EXIT1(256, "The table for dimension %d must contain two numbers or none\n", k+1);
        }
      } // if (tt == LUA_TTABLE)
      else if (tt == LUA_TNUMBER) {
        a = b = lua_toint(L, n);
      }
      else if (tt == LUA_TSTRING) {
        const char *slice = lua_tostring(L, n);
        MatrixBindings<T>::parse_slice(slice, a, b, min, max);
      }
      else { // not a table, not a number and neither a string :S
        ERROR_EXIT1(256, "The argument %d is not a table neither a string or a number\n", k+1);
      }
      if (a < 0) a = max + a;
      if (b < 0) b = max + b;
      
      if (a < min || a > max) {
        ERROR_EXIT2(256, "Range limit %d out of bounds for dim %d\n", a, k+1);
      }
      if (b < min || b > max) {
        ERROR_EXIT2(256, "Range limit %d out of bounds for dim %d\n", b, k+1);
      }
      if (a > b) {
        ERROR_EXIT(256, "Expecting an increasing range\n");
      }
    }

    static Matrix<T> *buildMatrixSliceFromTable(lua_State *L, const int n,
                                                Matrix<T> *obj) {
      const int len = lua_rawlen(L, n);
      if (len == 0) return obj;
      const int ndims = obj->getNumDim();
      const int *dims = obj->getDimPtr();
      if (len > ndims) {
        ERROR_EXIT(128, "Number of dimensions out-of-bounds\n");
      }
      AprilUtils::UniquePtr<int []> coords = new int[ndims];
      AprilUtils::UniquePtr<int []> sizes = new int[ndims];
      int a=0, b=0;
      for (int i=1, k=0; i<=len; ++i, ++k) {
        lua_rawgeti(L, n, i);
        MatrixBindings<T>::extract_range(L, -1, k, a, b, 1, dims[k]);
        
        coords[k] = a - 1; // -1 because in C we start at 0
        sizes[k] = b - a + 1;
        
        lua_pop(L, 1);
      } // for (int i=2, k=0; i<=n; ++i, ++k)
      for (int k=len; k<ndims; ++k) {
        coords[k] = 0;
        sizes[k] = dims[k];
      }
      return new Matrix<T>(obj, coords.get(), sizes.get(), false); 
    }

    static Matrix<T> *buildMatrixSliceFromArgs(lua_State *L,
                                               const int first,
                                               Matrix<T> *obj) {
      const int n = lua_gettop(L);
      const int given_ndims = n - first + 1;
      if (given_ndims == 0) return obj;
      const int ndims = obj->getNumDim();
      const int *dims = obj->getDimPtr();
      if (given_ndims > ndims) {
        ERROR_EXIT(128, "Number of dimensions out-of-bounds\n");
      }
      AprilUtils::UniquePtr<int []> coords = new int[ndims];
      AprilUtils::UniquePtr<int []> sizes = new int[ndims];
      int a=0, b=0;
      for (int i=first, k=0; i<=n; ++i, ++k) {
        MatrixBindings<T>::extract_range(L, i, k, a, b, 1, dims[k]);
        
        coords[k] = a - 1; // -1 because in C we start at 0
        sizes[k] = b - a + 1;
      } // for (int i=2, k=0; i<=n; ++i, ++k)
      for (int k=given_ndims; k<ndims; ++k) {
        coords[k] = 0;
        sizes[k] = dims[k];
      }
      return new Matrix<T>(obj, coords.get(), sizes.get(), false);
    }
    
    class LuaFunctionWrapper {
    public:
      LuaFunctionWrapper(lua_State *L, int n) : L(L), n(n) {}
      Matrix<T> *operator()(Matrix<T> *a, const Matrix<T> *b) const {
        lua_pushvalue(L,n);
        lua_push(L,a);
        lua_push(L,const_cast<Matrix<T>*>(b));
        if (lua_pcall(L, 2, 1, 0) != LUA_OK) {
          AprilUtils::string str(lua_tostring(L,-1));
          lua_pop(L,1);
          ERROR_EXIT1(128, "%s", str.c_str());
        }
        AprilUtils::SharedPtr< Matrix<T> > c = lua_to<Matrix<T>*>(L,-1);
        lua_pop(L,1);
        return c.weakRelease();
      }
    private:
      mutable lua_State *L;
      const int n; // position at Lua stack
    };

    template<typename K>
    static K lua_rawto(lua_State *L, int n) {
      return AprilUtils::LuaTable::lua_rawgetudata<K>(L,n);
    }
    
    template<typename K>
    static bool lua_is(lua_State *L, int n) {
      return AprilUtils::LuaTable::checkType<K>(L,n);
    }
    
    template<typename K>
    static K lua_to(lua_State *L, int n) {
      if (!lua_is<K>(L,n)) {
        ERROR_EXIT1(128, "Incorrect argument type at position %d\n", n);
      }
      return AprilUtils::LuaTable::convertTo<K>(L, n);
    }

    template<typename K>
    static K lua_opt(lua_State *L, int n, K default_value) {
      if (lua_isnil(L, n) || lua_type(L, n) == LUA_TNONE) return default_value;
      else return lua_to<K>(L,n);
    }

    template<typename K>
    static void lua_push(lua_State *L, K obj) {
      AprilUtils::LuaTable::pushInto(L, obj);
    }

#define FUNCTION_NAME "read_vector"
    template<typename K>
    static K *read_vector(lua_State *L, const char *key, int num_dim, K add) {
      AprilUtils::UniquePtr<K []> v;
      lua_getfield(L, 1, key);
      if (!lua_isnil(L, -1)) {
        LUABIND_CHECK_PARAMETER(-1, table);
        int table_len;
        LUABIND_TABLE_GETN(-1, table_len);
        if (table_len != num_dim)
          LUABIND_FERROR3("Table '%s' with incorrect size, expected %d, found %d",
                          key, num_dim, table_len);
        v = new K[num_dim];
        for(int i=0; i < num_dim; i++) {
          lua_rawgeti(L, -1, i+1);
          v[i] = lua_to<K>(L,-1) + add;
          lua_pop(L,1);
        }
      }
      lua_pop(L, 1);
      return v.release();
    }
#undef FUNCTION_NAME

    static int sliding_window_iterator_function(lua_State *L) {
      typename Matrix<T>::sliding_window *obj = lua_to<typename Matrix<T>::sliding_window*>(L,1);
      if (obj->isEnd()) {
        lua_pushnil(L);
        return 1;
      }
      Matrix<T> *mat = obj->getMatrix();
      lua_push(L, mat);
      obj->next();
      return 1;
    }
    
  public:
#define BEGIN_METHOD(name)       static int name(lua_State *L, Matrix<T> *obj)
#define BEGIN_CLASS_METHOD(name) static int name(lua_State *L)
#define BEGIN_SW_METHOD(name)    static int name(lua_State *L, typename Matrix<T>::sliding_window *obj)

    //////////////////////////////////////////////////////////////////////////

#define FUNCTION_NAME "sliding_window binding"
    BEGIN_SW_METHOD(get_matrix)
    {
      lua_push(L, obj->getMatrix(lua_opt<Matrix<T>*>(L, 1, 0)));
      return 1;
    }

    BEGIN_SW_METHOD(next)
    {
      lua_push(L, obj->next());
      return 1;
    }
    
    BEGIN_SW_METHOD(set_at_window)
    {
      int windex;
      LUABIND_CHECK_ARGN(==,1);
      LUABIND_GET_PARAMETER(1, int, windex);
      if (windex < 1) LUABIND_ERROR("Index must be >= 1\n");
      obj->setAtWindow(windex-1);
      lua_push(L, obj);
      return 1;
    }

    BEGIN_SW_METHOD(num_windows)
    {
      lua_push(L, obj->numWindows());
      return 1;
    }

    BEGIN_SW_METHOD(coords)
    {
      LUABIND_VECTOR_TO_NEW_TABLE(int, obj->getCoords(), obj->getNumDim());
      return 1;
    }

    BEGIN_SW_METHOD(is_end)
    {
      lua_push(L, obj->isEnd());
      return 1;
    }

    BEGIN_SW_METHOD(iterate)
    {
      LUABIND_CHECK_ARGN(==, 0);
      lua_pushcfunction(L, MatrixBindings<T>::sliding_window_iterator_function);
      lua_push(L, obj);
      return 2;
    }

#undef FUNCTION_NAME
    
    //////////////////////////////////////////////////////////////////////////
    
#define FUNCTION_NAME "matrix binding"
    
    BEGIN_CLASS_METHOD(MMapped)
    {
      bool aux = AprilMath::GPUMirroredMemoryBlockBase::getUseMMapAllocation();
      AprilMath::GPUMirroredMemoryBlockBase::setUseMMapAllocation(true);
      int n = MatrixBindings<T>::constructor(L);
      AprilMath::GPUMirroredMemoryBlockBase::setUseMMapAllocation(aux);
      return n;
    }

    BEGIN_CLASS_METHOD(constructor)
    {
      if (typeid(T) == typeid(char) && lua_type(L,1) == LUA_TSTRING) { // for matrixChar case
        int len = luaL_len(L,1);
        const char *data = lua_tostring(L,1);
        AprilUtils::UniquePtr<int []> dim = new int[1];
        dim[0] = len;
        AprilUtils::SharedPtr<Matrix<T> > obj = new Matrix<T>(1, dim.get());
        for (typename Matrix<T>::iterator it(obj->begin()); it != obj->end(); ++it, ++data) {
          april_assert(data != '\0');
          *it = static_cast<T>(*data);
        }
        lua_push(L, obj.get());
      }
      else {
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
        AprilUtils::SharedPtr< Matrix<T> > obj;
        if (lua_is<AprilMath::GPUMirroredMemoryBlock<T>*>(L,argn)) {
          AprilUtils::SharedPtr< AprilMath::GPUMirroredMemoryBlock<T> > block;
          block = lua_to<AprilMath::GPUMirroredMemoryBlock<T>*>(L, argn);
          if (dim[0] == -1) dim[0] = block->getSize();
          obj = new Matrix<T>(ndims, dim.get(), block.get());
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
              if (!lua_is<T>(L, -1)) {
                LUABIND_FERROR1("The given table has an invalid value at position"
                                " %d, check table size and its content", i);
              }
              *it = lua_to<T>(L, -1);
              lua_remove(L,-1);
            } // for each matrix position
          } // if lua_istable(L,argn)
          else {
            if (ndims == 1 && dim[0] == -1) {
              LUABIND_ERROR("Incorrect matrix dimensions");
            }
            obj = new Matrix<T>(ndims, dim.get());
            if (typeid(T) == typeid(char) && lua_type(L,argn) == LUA_TSTRING) { // for matrixChar case
              int len = luaL_len(L,argn);
              if (len != obj->size()) {
                LUABIND_ERROR("Not matching sizes");
              }
              const char *data = lua_tostring(L,argn);
              for (typename Matrix<T>::iterator it(obj->begin());
                   it != obj->end(); ++it, ++data) {
                april_assert(data != '\0');
                *it = static_cast<T>(*data);
              }
            }
          }
        } // else { !lua_is(L,argn) }
        lua_push(L, obj.get());
      }
      return 1;
    }

    BEGIN_METHOD(convert_to)
    {
      LUABIND_CHECK_ARGN(==,1);
      const char *type;
      LUABIND_GET_PARAMETER(1, string, type);
      if (!strcmp(type,"float")) {
        Matrix<float> *obj2 = AprilMath::MatrixExt::Misc::
          matConvertTo<T,float>(obj);
        lua_push(L, obj2);
      }
      else if (!strcmp(type,"bool")) {
        Matrix<bool> *obj2 = AprilMath::MatrixExt::Misc::
          matConvertTo<T,bool>(obj);
        lua_push(L, obj2);
      }
      else if (!strcmp(type,"int32")) {
        Matrix<int32_t> *obj2 = AprilMath::MatrixExt::Misc::
          matConvertTo<T,int32_t>(obj);
        lua_push(L, obj2);
      }
      else if (!strcmp(type,"double")) {
        Matrix<double> *obj2 = AprilMath::MatrixExt::Misc::
          matConvertTo<T,double>(obj);
        lua_push(L, obj2);
      }
      else if (!strcmp(type,"char")) {
        Matrix<char> *obj2 = AprilMath::MatrixExt::Misc::
          matConvertTo<T,char>(obj);
        lua_push(L, obj2);
      }
      else if (!strcmp(type,"complex")) {
        Matrix<AprilMath::ComplexF> *obj2 = AprilMath::MatrixExt::Misc::
          matConvertTo<T,AprilMath::ComplexF>(obj);
        lua_push(L, obj2);
      }
      else {
        LUABIND_FERROR1("Not implemented casting for type %s", type);
      }
      return 1;
    }

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
      lua_push(L, new_obj);
      return 1;
    }

    BEGIN_METHOD(get_reference_string)
    {
      char buff[128];
      sprintf(buff,"%p data= %p",
              (void*)obj,
              (void*)obj->getRawDataAccess());
      lua_pushstring(L, buff);
      return 1;
    }

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
        if (!lua_is<T>(L,-1)) {
          LUABIND_FERROR1("The given table has a no number value at position %d, "
                          "the table could be smaller than matrix size", i);
        }
        *it = lua_to<T>(L,-1);
        lua_remove(L,-1);
      }
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(get)
    {
      const Matrix<T> *cobj = obj;
      int argn = lua_gettop(L); // number of arguments
      if (argn != cobj->getNumDim())
        LUABIND_FERROR2("wrong size %d instead of %d",argn,cobj->getNumDim());
      T ret;
      if (cobj->getNumDim() == 1) {
        int v1;
        LUABIND_GET_PARAMETER(1,int,v1);
        if (v1<1 || v1 > cobj->getDimSize(0)) {
          LUABIND_FERROR2("wrong index parameter 1, %d is not <= %d",
                          v1, cobj->getDimSize(0));
        }
        ret = (*cobj)(v1-1);
      }
      else if (cobj->getNumDim() == 2) {
        int v1, v2;
        LUABIND_GET_PARAMETER(1,int,v1);
        LUABIND_GET_PARAMETER(2,int,v2);
        if (v1<1 || v1 > cobj->getDimSize(0)) {
          LUABIND_FERROR2("wrong index parameter 1, %d is not <= %d or is not >= 1",
                          v1, cobj->getDimSize(0));
        }
        if (v2<1 || v2 > cobj->getDimSize(1)) {
          LUABIND_FERROR2("wrong index parameter 2, %d is not <= %d or is not >= 1",
                          v2, cobj->getDimSize(1));
        }
        ret = (*cobj)(v1-1, v2-1);
      }
      else {
        AprilUtils::UniquePtr<int []> coords( new int[cobj->getNumDim()] );
        for (int i=0; i<cobj->getNumDim(); ++i) {
          LUABIND_GET_PARAMETER(i+1,int,coords[i]);
          if (coords[i]<1 || coords[i] > cobj->getDimSize(i)) {
            LUABIND_FERROR2("wrong index parameter %d, %d is not <= %d or is not >= 1",
                            coords[i], cobj->getDimSize(i));
          }
          coords[i]--;
        }
        ret = (*cobj)(coords.get(), cobj->getNumDim());
      }
      lua_push(L, ret);
      return 1;
    }

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
        f = lua_to<T>(L, obj->getNumDim()+1);
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
        f = lua_to<T>(L, obj->getNumDim()+1);
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
        f = lua_to<T>(L, obj->getNumDim()+1);
        (*obj)(coords.get(), obj->getNumDim()) = f;
      }
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(raw_get)
    {
      int raw_pos;
      LUABIND_GET_PARAMETER(1, int, raw_pos);
      lua_push(L, (*obj)[raw_pos]);
      return 1;
    }

    BEGIN_METHOD(raw_set)
    {
      int raw_pos;
      LUABIND_GET_PARAMETER(1, int, raw_pos);
      T value = lua_to<T>(L, 2);      
      (*obj)[raw_pos] = value;
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(set_use_cuda)
    {
      LUABIND_CHECK_ARGN(==, 1);
      LUABIND_CHECK_PARAMETER(1, bool);
      bool v;
      LUABIND_GET_PARAMETER(1,bool, v);
      obj->setUseCuda(v);
      lua_push(L, obj);
      return 1;
    }

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
      lua_push(L, obj2);
      return 1;
    }

    BEGIN_METHOD(select)
    {
      LUABIND_CHECK_ARGN(>=,2);
      LUABIND_CHECK_ARGN(<=,3);
      LUABIND_CHECK_PARAMETER(1, int);
      LUABIND_CHECK_PARAMETER(2, int);
      int dim, index;
      LUABIND_GET_PARAMETER(1, int, dim);
      LUABIND_GET_PARAMETER(2, int, index);
      Matrix<T> *dest = lua_opt<Matrix<T>*>(L, 3, 0);
      Matrix<T> *obj2 = obj->select(dim-1, index-1, dest);
      lua_push(L, obj2);
      return 1;
    }

    BEGIN_CLASS_METHOD(as)
    {
      LUABIND_CHECK_ARGN(==, 1);
      Matrix<T> *m;
      m = lua_to<Matrix<T>*>(L, 1);
      lua_push(L, m->cloneOnlyDims());
      return 1;
    }

    BEGIN_METHOD(transpose)
    {
      int argn;
      argn = lua_gettop(L);
      if (argn == 0) {
        lua_push(L, obj->transpose());
      }
      else {
        int d1,d2;
        LUABIND_GET_PARAMETER(1, int, d1);
        LUABIND_GET_PARAMETER(2, int, d2);
        lua_push(L, obj->transpose(d1-1, d2-1));
      }
      return 1;
    }

    BEGIN_CLASS_METHOD(deserialize)
    {
      check_table_fields(L, 1, "stride", "sizes", "data", "offset", 
                         (const char *)0);
      int offset;
      AprilUtils::UniquePtr<int []> sizes;
      AprilUtils::UniquePtr<int []> stride;
      AprilMath::GPUMirroredMemoryBlock<T> *data;
      lua_getfield(L, 1, "data");
      data = lua_to<AprilMath::GPUMirroredMemoryBlock<T>*>(L, -1);
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
      lua_push<Matrix<T>*>(L, obj);
      return 1;
    }

    BEGIN_CLASS_METHOD(read)
    {
      AprilIO::StreamInterface *stream =
        lua_toAuxStreamInterface<AprilIO::StreamInterface>(L,1);
      if (!stream) LUABIND_ERROR("Needs a stream as first argument");
      AprilUtils::SharedPtr<AprilIO::StreamInterface> ptr(stream);
      AprilUtils::LuaTable options(L,2);
      Matrix<T> *obj = Matrix<T>::read(ptr.get(), options);
      if (!obj) {
        LUABIND_ERROR("Error happens reading from file stream");
      }
      else {
        lua_push<Matrix<T>*>(L, obj);
      }
      return 1;
    }

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
      lua_push<Matrix<T>*>(L, obj);
      return 1;
    }

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
        if (!lua_is<Matrix<T>*>(L, i+1)) {
          LUABIND_FERROR1("Expected a matrix at position: ", i+1);
        }
        v[i] = lua_rawto<Matrix<T>*>(L, i+1);
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
        lua_push(L, *it);
        // push the value of the rest of given matrices
        for (int j=0; j<N; ++j) {
          lua_push(L, *list_it[j]);
          ++list_it[j];
        }
        // CALL
        if (lua_pcall(L, N+1, 1, 0) != LUA_OK) {
          AprilUtils::string str(lua_tostring(L,-1));
          lua_pop(L,1);
          ERROR_EXIT1(128, "%s", str.c_str());
        }
        // pop the result, a number
        if (!lua_isnil(L, -1)) {
          if (!lua_is<T>(L, -1)) {
            LUABIND_ERROR("Incorrect returned value type");
          }
          *it = lua_to<T>(L, -1);
        }
        lua_pop(L, 1);
      }
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(num_dim)
    {
      lua_push(L, obj->getNumDim());
      return 1;
    }
    
    BEGIN_METHOD(to_index)
    {
      MatrixInt32 *m = AprilMath::MatrixExt::Misc::matNonZeroIndices(obj);
      lua_push(L, m);
      return 1;
    }

    BEGIN_METHOD(equals)
    {
      Matrix<T> *other;
      float epsilon;
      other = lua_to<Matrix<T>*>(L,1);
      LUABIND_GET_OPTIONAL_PARAMETER(2, float, epsilon, 0.05f); // 5% error
#ifdef USE_CUDA
      obj->sync();
      other->sync();
#endif
      if (AprilMath::MatrixExt::Reductions::matEquals(obj, other, epsilon)) {
        lua_pushboolean(L, true);
      }
      else {
        lua_pushboolean(L, false);
      }
      return 1;
    }

    BEGIN_METHOD(add_to_shared_count)
    {
      unsigned int count;
      LUABIND_GET_PARAMETER(1,uint,count);
      obj->addToSharedCount(count);
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(padding_all)
    {
      int padding;
      LUABIND_GET_PARAMETER(1, int, padding);
      T default_value = lua_opt(L, 2, AprilMath::Limits<T>::zero());
      Matrix<T> *result = obj->padding(padding, default_value);
      lua_push(L, result);
      return 1;
    }

    BEGIN_METHOD(padding)
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
      T default_value = lua_opt(L, j, AprilMath::Limits<T>::zero());
      Matrix<T> *result = obj->padding(begin_padding.get(),
                                       end_padding.get(),
                                       default_value);
      lua_push(L, result);
      return 1;
    }

    BEGIN_METHOD(uniform)
    {
      int lower, upper;
      AprilUtils::SharedPtr<Basics::MTRand> random;
      LUABIND_GET_PARAMETER(1, int, lower);
      LUABIND_GET_PARAMETER(2, int, upper);
      LUABIND_GET_OPTIONAL_PARAMETER(3, MTRand, random, 0);
      if (lower > upper) {
        LUABIND_ERROR("First argument must be <= second argument");
      }
      if (!random) random = new Basics::MTRand();
      for (typename Matrix<T>::iterator it(obj->begin()); it != obj->end(); ++it) {
        *it = T(random->randInt(upper - lower)) + lower;
      }
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(uniformf)
    {
      T lower = AprilMath::Limits<T>::zero(), upper = AprilMath::Limits<T>::one();
      AprilUtils::SharedPtr<Basics::MTRand> random;

      LUABIND_GET_OPTIONAL_PARAMETER(1, float, lower, 0.0f);
      LUABIND_GET_OPTIONAL_PARAMETER(2, float, upper, 1.0f);
      LUABIND_GET_OPTIONAL_PARAMETER(3, MTRand, random, 0);
      if (lower > upper) {
        LUABIND_ERROR("First argument must be <= second argument");
      }
      if (!random) random = new Basics::MTRand();
      for (typename Matrix<T>::iterator it(obj->begin()); it != obj->end(); ++it) {
        *it = T(random->rand(upper - lower) + lower);
      }
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(linspace)
    {
      int size_1 = obj->size()-1;
      T inf = lua_opt(L, 1, AprilMath::Limits<T>::one());
      T sup = lua_opt(L, 2, static_cast<T>(size_1+1));
      int i = 0;
      T diff = sup-inf;
      if (diff == size_1) {
        i = static_cast<int>(inf);
        for (typename Matrix<T>::iterator it(obj->begin()); it!=obj->end(); ++it, ++i) {
          april_assert(i <= static_cast<int>(sup));
          *it = static_cast<T>(i);
        }
      }
      else {
        for (typename Matrix<T>::iterator it(obj->begin()); it!=obj->end(); ++it, ++i) {
          *it = (diff*i)/size_1 + inf;
        }
      }
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(logspace)
    {
      int size = obj->size()-1;
      T inf  = lua_opt(L, 1, T(1.0f));
      T sup  = lua_opt(L, 2, T(size+1));
      T base = lua_opt(L, 3, T(10.0f));
      int i=0;
      inf = AprilMath::m_log(inf)/AprilMath::m_log(base);
      sup = AprilMath::m_log(sup)/AprilMath::m_log(base);
      T diff = sup-inf;
      for (typename Matrix<T>::iterator it(obj->begin()); it!=obj->end(); ++it, ++i) {
        *it = AprilMath::m_pow(base, (diff*i)/size + inf);
      }
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(linear)
    {
      int lower, step;
      LUABIND_GET_OPTIONAL_PARAMETER(1, int, lower, 0);
      LUABIND_GET_OPTIONAL_PARAMETER(2, int, step,  1);
      int k=lower;
      for (typename Matrix<T>::iterator it(obj->begin()); it != obj->end(); ++it, k+=step) {
        *it = T(k);
      }
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(sliding_window)
    {
      AprilUtils::UniquePtr<int []> sub_matrix_size, offset,
        step, num_steps, order_step;
      int argn = lua_gettop(L); // number of arguments
      const int num_dim = obj->getNumDim();
      if (argn > 1) {
        LUABIND_ERROR("incorrect number of arguments");
      }
      if (argn == 1) {
        LUABIND_CHECK_PARAMETER(1, table);
        check_table_fields(L, 1,
                           "offset",
                           "size",
                           "step",
                           "numSteps",
                           "orderStep",
                           (const char*)0);
        offset = read_vector<int>(L, "offset", num_dim, 0);
        sub_matrix_size = read_vector<int>(L, "size", num_dim, 0);
        step = read_vector<int>(L, "step", num_dim, 0);
        num_steps = read_vector<int>(L, "numSteps", num_dim, 0);
        order_step = read_vector<int>(L, "orderStep", num_dim, -1);
      }
      typename Matrix<T>::sliding_window *window =
        new typename Matrix<T>::sliding_window(obj,
                                               sub_matrix_size.get(),
                                               offset.get(),
                                               step.get(),
                                               num_steps.get(),
                                               order_step.get());
      lua_push(L, window);
      return 1;
    }

    BEGIN_METHOD(adjust_range)
    {
      T rmin = lua_to<T>(L,1);
      T rmax = lua_to<T>(L,2);
      lua_push(L, AprilMath::MatrixExt::Operations::
               matAdjustRange(obj, rmin, rmax));
      return 1;
    }

    BEGIN_METHOD(size)
    {
      lua_push(L, obj->size());
      return 1;
    }

    BEGIN_METHOD(squeeze)
    {
      lua_push(L, obj->squeeze());
      return 1;
    }

    BEGIN_METHOD(left_inflate)
    {
      lua_push(L, obj->leftInflate(luaL_optint(L,1,1)));
      return 1;
    }

    BEGIN_METHOD(right_inflate)
    {
      lua_push(L, obj->rightInflate(luaL_optint(L,1,1)));
      return 1;
    }

    BEGIN_METHOD(offset)
    {
      lua_push(L, obj->getOffset());
      return 1;
    }

    BEGIN_METHOD(get_use_cuda)
    {
      lua_push(L, obj->getCudaFlag());
      return 1;
    }

    BEGIN_METHOD(clone)
    {
      lua_push(L, obj->clone());
      return 1;
    }

    BEGIN_METHOD(isfinite)
    {
      lua_push(L, AprilMath::MatrixExt::Reductions::matIsFinite(obj));
      return 1;
    }

    BEGIN_METHOD(toTable)
    {
      AprilUtils::LuaTable t(L);
      int i=1;
      for(typename Matrix<T>::const_iterator it = obj->begin();
          it != obj->end(); ++it, ++i) {
        t[i] = *it;
      }
      t.pushTable(L);
      return 1;
    }

    BEGIN_METHOD(contiguous)
    {
      lua_push(L, obj->getIsContiguous() ? obj : obj->clone());
      return 1;
    }

    BEGIN_METHOD(diagonalize)
    {
      lua_push(L, obj->diagonalize());
      return 1;
    }

    BEGIN_METHOD(get_shared_count)
    {
      lua_push(L, obj->getSharedCount());
      return 1;
    }

    BEGIN_METHOD(reset_shared_count)
    {
      obj->resetSharedCount();
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(sync)
    {
      obj->sync();
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(is_contiguous)
    {
      lua_push(L, obj->getIsContiguous());
      return 1;
    }

    BEGIN_METHOD(prune_subnormal_and_check_normal)
    {
      obj->pruneSubnormalAndCheckNormal();
      lua_push(L, obj);
      return 1;
    }
    
    
    BEGIN_METHOD(diag)
    {
      LUABIND_CHECK_ARGN(==,1);
      T v = lua_to<T>(L,1);
      AprilMath::MatrixExt::Initializers::matDiag(obj, v);
      lua_push(L, obj);
      return 1;
    }

    BEGIN_METHOD(fill)
    {
      LUABIND_CHECK_ARGN(==, 1);
      T value;
      if (lua_is<Matrix<T>*>(L,1)) {
        Matrix<T> *aux = lua_rawto<Matrix<T>*>(L,1);
        for (int i=0; i<aux->getNumDim(); ++i) {
          if (aux->getDimSize(i) != 1) {
            LUABIND_ERROR("Needs a number or a matrix with only one element\n");
          }
        }
        value = *(aux->begin());
      }
      else {
        value = lua_to<T>(L,1);
      }
      lua_push(L, AprilMath::MatrixExt::Initializers::matFill(obj,value));
      return 1;
    }

    BEGIN_METHOD(zeros)
    {
      lua_push(L, AprilMath::MatrixExt::Initializers::matZeros(obj));
      return 1;
    }

    BEGIN_METHOD(ones)
    {
      lua_push(L, AprilMath::MatrixExt::Initializers::matOnes(obj));
      return 1;
    }

    BEGIN_METHOD(min)
    {
#ifdef USE_CUDA
      obj->sync();
#endif
      LUABIND_CHECK_ARGN(>=,0);
      LUABIND_CHECK_ARGN(<=,3);
      int argn = lua_gettop(L);
      if (argn > 0) {
        // case over a dimension
        int dim;
        LUABIND_GET_PARAMETER(1, int, dim);
        AprilUtils::SharedPtr<Matrix<T> > dest;
        AprilUtils::SharedPtr<Matrix<int32_t> > argmin;
        dest = lua_opt<Matrix<T>*>(L,2,0);
        argmin = lua_opt<MatrixInt32*>(L,3,0);
        AprilUtils::UniquePtr<int []> aux;
        if (!argmin) {
          aux = new int[obj->getNumDim()];
          for (int i=0; i<obj->getNumDim(); ++i) aux[i] = obj->getDimSize(i);
          aux[dim-1] = 1;
          argmin = new Matrix<int32_t>(obj->getNumDim(), aux.get());
        }
        if (dim < 1 || dim > obj->getNumDim()) {
          LUABIND_FERROR2("Incorrect dimension, found %d, expect in [1,%d]",
                          dim, obj->getNumDim());
        }
        lua_push(L, AprilMath::MatrixExt::Reductions::
                 matMin(obj, dim-1, dest.get(), argmin.get()));
        lua_push(L, argmin.get());
      }
      else {
        // case over whole matrix
        int arg_min, raw_pos;
        lua_push(L, AprilMath::MatrixExt::Reductions::
                 matMin(obj, arg_min, raw_pos));
        lua_push(L, arg_min+1);
      }
      return 2;
    }

    BEGIN_METHOD(max)
    {
#ifdef USE_CUDA
      obj->sync();
#endif
      LUABIND_CHECK_ARGN(>=,0);
      LUABIND_CHECK_ARGN(<=,3);
      int argn = lua_gettop(L);
      if (argn > 0) {
        // case over a dimension
        int dim;
        LUABIND_GET_PARAMETER(1, int, dim);
        AprilUtils::SharedPtr<Matrix<T> > dest;
        AprilUtils::SharedPtr<Matrix<int32_t> > argmax;
        dest = lua_opt<Matrix<T>*>(L,2,0);
        argmax = lua_opt<MatrixInt32*>(L,3,0);
        AprilUtils::UniquePtr<int []> aux;
        if (!argmax) {
          aux = new int[obj->getNumDim()];
          for (int i=0; i<obj->getNumDim(); ++i) aux[i] = obj->getDimSize(i);
          aux[dim-1] = 1;
          argmax = new Matrix<int32_t>(obj->getNumDim(), aux.get());
        }
        if (dim < 1 || dim > obj->getNumDim()) {
          LUABIND_FERROR2("Incorrect dimension, found %d, expect in [1,%d]",
                          dim, obj->getNumDim());
        }
        lua_push(L, AprilMath::MatrixExt::Reductions::
                 matMax(obj, dim-1, dest.get(), argmax.get()));
        lua_push(L, argmax.get());
      }
      else {
        // case over whole matrix
        int arg_max, raw_pos;
        lua_push(L, AprilMath::MatrixExt::Reductions::
                 matMax(obj, arg_max, raw_pos));
        lua_push(L, arg_max+1);
      }
      return 2;
    }

    BEGIN_METHOD(clamp)
    {
      LUABIND_CHECK_ARGN(==, 2);
      T lower = lua_to<T>(L,1), upper = lua_to<T>(L,2);
      lua_push(L, AprilMath::MatrixExt::Operations::
               matClamp(obj, lower, upper));
      return 1;
    }

    BEGIN_METHOD(floor)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::
               matFloor(obj));
      return 1;
    }

    BEGIN_METHOD(ceil)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::
               matCeil(obj));
      return 1;
    }

    BEGIN_METHOD(round)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::
               matRound(obj));
      return 1;
    }

    BEGIN_METHOD(add)
    {
      int argn;
      argn = lua_gettop(L); // number of arguments
      LUABIND_CHECK_ARGN(==, 1);
      Matrix<T> *mat = lua_to<Matrix<T>*>(L,1);
      if (!obj->sameDim(mat)) {
        LUABIND_ERROR("matrix add wrong dimensions");
      }
#ifdef USE_CUDA
      mat->sync();
#endif
      lua_push(L, AprilMath::MatrixExt::Misc::
               matAddition(obj, mat));
      return 1;
    }

    BEGIN_METHOD(scalar_add)
    {
      int argn;
      argn = lua_gettop(L); // number of arguments
      LUABIND_CHECK_ARGN(==, 1);
      T scalar = lua_to<T>(L, 1);
      lua_push(L, AprilMath::MatrixExt::Operations::
               matScalarAdd(obj, scalar));
      return 1;
    }

    BEGIN_METHOD(sub)
    {
      LUABIND_CHECK_ARGN(==, 1);
      Matrix<T> *mat = lua_to<Matrix<T>*>(L, 1);
      if (!obj->sameDim(mat)) {
        LUABIND_ERROR("incompatible dimensions");
      }
#ifdef USE_CUDA
      mat->sync();
#endif
      lua_push(L, AprilMath::MatrixExt::Misc::
               matSubstraction(obj, mat));
      return 1;
    }

    BEGIN_METHOD(mul)
    {
      LUABIND_CHECK_ARGN(==, 1);
      Matrix<T> *mat = lua_to<Matrix<T>*>(L, 1);
#ifdef USE_CUDA
      mat->sync();
#endif
      lua_push(L, AprilMath::MatrixExt::Misc::
               matMultiply(obj, mat));
      return 1;
    }

    BEGIN_METHOD(cmul)
    {
      LUABIND_CHECK_ARGN(==, 1);
      Matrix<T> *mat = lua_to<Matrix<T>*>(L, 1);
#ifdef USE_CUDA
      mat->sync();
#endif
      lua_push(L, AprilMath::MatrixExt::Operations::
               matCmul(obj, mat));
      return 1;
    }

    BEGIN_METHOD(plogp)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matPlogp(obj));
      return 1;
    }

    BEGIN_METHOD(log)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matLog(obj));
      return 1;
    }

    BEGIN_METHOD(log1p)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matLog1p(obj));
      return 1;
    }

    BEGIN_METHOD(exp)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matExp(obj));
      return 1;
    }

    BEGIN_METHOD(expm1)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matExpm1(obj));
      return 1;
    }

    BEGIN_METHOD(sqrt)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matSqrt(obj));
      return 1;
    }

    BEGIN_METHOD(pow)
    {
      T value = lua_to<T>(L, 1);
      lua_push(L, AprilMath::MatrixExt::Operations::matPow(obj, value));
      return 1;
    }

    BEGIN_METHOD(tan)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matTan(obj));
      return 1;
    }

    BEGIN_METHOD(tanh)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matTanh(obj));
      return 1;
    }

    BEGIN_METHOD(atan)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matAtan(obj));
      return 1;
    }

    BEGIN_METHOD(atanh)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matAtanh(obj));
      return 1;
    }

    BEGIN_METHOD(sin)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matSin(obj));
      return 1;
    }

    BEGIN_METHOD(sinh)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matSinh(obj));
      return 1;
    }

    BEGIN_METHOD(asin)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matAsin(obj));
      return 1;
    }

    BEGIN_METHOD(asinh)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matAsinh(obj));
      return 1;
    }

    BEGIN_METHOD(cos)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matCos(obj));
      return 1;
    }

    BEGIN_METHOD(cosh)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matCosh(obj));
      return 1;
    }

    BEGIN_METHOD(acos)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matAcos(obj));
      return 1;
    }

    BEGIN_METHOD(acosh)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matAcosh(obj));
      return 1;
    }

    BEGIN_METHOD(abs)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matAbs(obj));
      return 1;
    }

    BEGIN_METHOD(complement)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matComplement(obj));
      return 1;
    }

    BEGIN_METHOD(cinv)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::
               matDiv(obj, AprilMath::Limits<T>::one()));
      return 1;
    }

    BEGIN_METHOD(sign)
    {
      lua_push(L, AprilMath::MatrixExt::Operations::matSign(obj));
      return 1;
    }

    BEGIN_METHOD(sum)
    {
#ifdef USE_CUDA
      obj->sync();
#endif
      LUABIND_CHECK_ARGN(>=, 0);
      LUABIND_CHECK_ARGN(<=, 2);
      int argn = lua_gettop(L); // number of arguments
      if (argn > 0 && !lua_isnil(L,1)) {
        int dim;
        LUABIND_GET_PARAMETER(1, int, dim);
        Matrix<T> *dest = lua_opt<Matrix<T>*>(L, 2, 0);
        if (dim < 1 || dim > obj->getNumDim()) {
          LUABIND_FERROR2("Incorrect dimension, found %d, expect in [1,%d]",
                          dim, obj->getNumDim());
        }
        lua_push(L, AprilMath::MatrixExt::Reductions::
                 matSum(obj, dim-1, dest));
      }
      else {
        lua_push(L, AprilMath::MatrixExt::Reductions::
                 matSum(obj));
      }
      return 1;
    }

    BEGIN_METHOD(copy)
    {
      int argn;
      LUABIND_CHECK_ARGN(==, 1);
      Matrix<T> *mat = lua_to<Matrix<T>*>(L, 1);
#ifdef USE_CUDA
      mat->sync();
#endif
      lua_push(L, AprilMath::MatrixExt::BLAS::matCopy(obj,mat));
      return 1;
    }

    BEGIN_METHOD(axpy)
    {
      int argn;
      LUABIND_CHECK_ARGN(==, 2);
      T alpha = lua_to<T>(L, 1);
      if (lua_is<Matrix<T>*>(L, 2)) {
        Matrix<T> *mat = lua_rawto<Matrix<T>*>(L, 2);
#ifdef USE_CUDA
        mat->sync();
#endif
        lua_push(L, AprilMath::MatrixExt::BLAS::matAxpy(obj, alpha, mat));
      }
      else if (lua_is<SparseMatrix<T>*>(L, 2)) {
        SparseMatrix<T> *mat = lua_to<SparseMatrix<T>*>(L, 2);
        lua_push(L, AprilMath::MatrixExt::BLAS::matAxpy(obj, alpha, mat));
      }
      else {
        LUABIND_ERROR("Expected matrix or sparse matrix as 2nd argument");
      }
      return 1;
    }
    
    BEGIN_METHOD(gemm)
    {
      LUABIND_CHECK_ARGN(==, 1);
      LUABIND_CHECK_PARAMETER(1, table);
      check_table_fields(L,1, "trans_A", "trans_B", "alpha", "A", "B", "beta",
                         (const char *)0);
      bool trans_A, trans_B;
      LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_A, bool, trans_A, false);
      LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_B, bool, trans_B, false);
      AprilUtils::LuaTable t(L, 1);
      T alpha         = t["alpha"].opt<T>( AprilMath::Limits<T>::one() );
      T beta          = t["beta"].opt<T>( AprilMath::Limits<T>::one() );
      Matrix<T> *matA = t["A"].get<Matrix<T>*>();
      Matrix<T> *matB = t["B"].get<Matrix<T>*>();
#ifdef USE_CUDA
      matA->sync();
      matB->sync();
#endif
      lua_push(L, AprilMath::MatrixExt::BLAS::
               matGemm(obj,
                       trans_A ? CblasTrans : CblasNoTrans,
                       trans_B ? CblasTrans : CblasNoTrans,
                       alpha, matA, matB,
                       beta));
      return 1;
    }

    BEGIN_METHOD(sparse_mm)
    {
      LUABIND_CHECK_ARGN(==, 1);
      LUABIND_CHECK_PARAMETER(1, table);
      check_table_fields(L,1, "trans_A", "trans_B",
                         "alpha", "A", "B", "beta",
                         (const char *)0);
      bool trans_A, trans_B;
      LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_A, bool, trans_A, false);
      LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_B, bool, trans_B, false);
      AprilUtils::LuaTable t(L, 1);
      T alpha         = t["alpha"].opt<T>( AprilMath::Limits<T>::one() );
      T beta          = t["beta"].opt<T>( AprilMath::Limits<T>::one() );
      SparseMatrix<T> *matA = t["A"].get<SparseMatrix<T>*>();
      Matrix<T> *matB = t["B"].get<Matrix<T>*>();
      lua_push(L, AprilMath::MatrixExt::BLAS::
               matSparseMM(obj,
                           trans_A ? CblasTrans : CblasNoTrans,
                           trans_B ? CblasTrans : CblasNoTrans,
                           alpha, matA, matB,
                           beta));
      return 1;
    }
    
    BEGIN_METHOD(gemv)
    {
      LUABIND_CHECK_ARGN(==, 1);
      LUABIND_CHECK_PARAMETER(1, table);
      check_table_fields(L,1, "trans_A", "alpha", "A", "X", "beta",
                         (const char *)0);
      bool trans_A;
      LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, trans_A, bool, trans_A, false);
      AprilUtils::LuaTable t(L, 1);
      T alpha         = t["alpha"].opt<T>( AprilMath::Limits<T>::one() );
      T beta          = t["beta"].opt<T>( AprilMath::Limits<T>::one() );
      Matrix<T> *matX = t["X"].get<Matrix<T>*>();
      lua_getfield(L, 1, "A");
      if (lua_is<Matrix<T>*>(L,-1)) {
        Matrix<T> *matA = lua_rawto<Matrix<T>*>(L,-1);
        lua_pop(L,1);
#ifdef USE_CUDA
        matA->sync();
        matX->sync();
#endif
        lua_push(L, AprilMath::MatrixExt::BLAS::
                 matGemv(obj,
                         trans_A ? CblasTrans : CblasNoTrans,
                         alpha, matA, matX,
                         beta));
      }
      else {
        SparseMatrix<T> *matA = lua_to<SparseMatrix<T>*>(L,-1);
        lua_pop(L,1);
        lua_push(L, AprilMath::MatrixExt::BLAS::
                 matGemv(obj,
                         trans_A ? CblasTrans : CblasNoTrans,
                         alpha, matA, matX,
                         beta));
      }
      return 1;
    }

    BEGIN_METHOD(ger)
    {
      LUABIND_CHECK_ARGN(==, 1);
      LUABIND_CHECK_PARAMETER(1, table);
      check_table_fields(L,1, "alpha", "X", "Y", (const char *)0);
      AprilUtils::LuaTable t(L,1);
      T alpha = t["alpha"].opt<T>( AprilMath::Limits<T>::one() );
      Matrix<T> *matX = t["X"].get<Matrix<T>*>();
      Matrix<T> *matY = t["Y"].get<Matrix<T>*>();
#ifdef USE_CUDA
      matX->sync();
      matY->sync();
#endif
      lua_push(L, AprilMath::MatrixExt::BLAS::
               matGer(obj, alpha, matX, matY));
      return 1;
    }

    BEGIN_METHOD(dot)
    {
      LUABIND_CHECK_ARGN(==, 1);
      if (lua_is<Matrix<T>*>(L, 1)) {
        Matrix<T> *matX = lua_rawto<Matrix<T>*>(L, 1);
#ifdef USE_CUDA
        obj->sync();
        matX->sync();
#endif
        lua_push(L, AprilMath::MatrixExt::BLAS::
                 matDot(obj, matX));
      }
      else if (lua_is<SparseMatrix<T>*>(L, 2)) {
        SparseMatrix<T> *matX = lua_to<SparseMatrix<T>*>(L, 1);
#ifdef USE_CUDA
        obj->sync();
        matX->sync();
#endif
        lua_push(L, AprilMath::MatrixExt::BLAS::
                 matDot(obj, matX));
        
      }
      return 1;
    }

    BEGIN_METHOD(scal)
    {
      LUABIND_CHECK_ARGN(==, 1);
      T value = lua_to<T>(L, 1);
      lua_push(L, AprilMath::MatrixExt::Operations::
               matScal(obj, value));
      return 1;
    }

    BEGIN_METHOD(masked_fill)
    {
      MatrixBool *mask = lua_to<MatrixBool*>(L, 1);
      T value          = lua_to<T>(L, 2);
      Matrix<T> *dest  = lua_opt<Matrix<T>*>(L, 3, 0);
      lua_push(L, AprilMath::MatrixExt::Operations::
               matMaskedFill(obj, mask, value, dest));
      return 1;
    }

    BEGIN_METHOD(masked_copy)
    {
      MatrixBool *mask = lua_to<MatrixBool*>(L, 1);
      Matrix<T> *value = lua_to<Matrix<T>*>(L, 2);
      Matrix<T> *dest  = lua_opt<Matrix<T>*>(L, 3, 0);
      lua_push(L, AprilMath::MatrixExt::Operations::
               matMaskedCopy(obj, mask, value, dest));
      return 1;
    }

    BEGIN_METHOD(div)
    {
      LUABIND_CHECK_ARGN(==, 1);
      if (lua_is<Matrix<T>*>(L, 1)) {
        Matrix<T> *other = lua_rawto<Matrix<T>*>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Operations::
                 matDiv(obj, other));
      }
      else {
        T value = lua_to<T>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Operations::
                 matDiv(obj, value));
      }
      return 1;
    }

    BEGIN_METHOD(idiv)
    {
      LUABIND_CHECK_ARGN(==, 1);
      if (lua_is<Matrix<T>*>(L, 1)) {
        Matrix<T> *other = lua_rawto<Matrix<T>*>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Operations::
                 matDiv(obj, other));
      }
      else {
        T value = lua_to<T>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Operations::
                 matIDiv(obj, value));
      }
      return 1;
    }

    BEGIN_METHOD(mod)
    {
      LUABIND_CHECK_ARGN(==, 1);
      if (lua_is<Matrix<T>*>(L, 1)) {
        Matrix<T> *other = lua_rawto<Matrix<T>*>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Operations::
                 matMod(obj, other));
      }
      else {
        T value = lua_to<T>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Operations::
                 matMod(obj, value));
      }
      return 1;
    }

    BEGIN_METHOD(norm2)
    {
#ifdef USE_CUDA
      obj->sync();
#endif
      lua_push(L, AprilMath::MatrixExt::BLAS::
               matNorm2(obj));
      return 1;
    }

    BEGIN_METHOD(inv)
    {
      lua_push(L, AprilMath::MatrixExt::LAPACK::
               matInv(obj));
      return 1;
    }

    BEGIN_METHOD(logdet)
    {
      T sign;
      lua_push(L, AprilMath::MatrixExt::LAPACK::
               matLogDeterminant(obj, sign).log());
      lua_push<T>(L, sign);
      return 2;
    }

    BEGIN_METHOD(det)
    {
      lua_push(L, AprilMath::MatrixExt::LAPACK::
               matDeterminant(obj));
      return 1;
    }

    BEGIN_METHOD(cholesky)
    {
      char uplo;
      LUABIND_GET_OPTIONAL_PARAMETER(1, char, uplo, 'U');
      uplo = toupper(uplo);
      if (uplo != 'U' && uplo != 'L') {
        LUABIND_ERROR("Incorrect argument, expected character L or U");
      }
      lua_push(L, AprilMath::MatrixExt::LAPACK::
               matCholesky(obj, uplo));
      return 1;
    }

    BEGIN_METHOD(svd)
    {
      Matrix<T> *U,*V;
      SparseMatrix<T> *S;
      AprilMath::MatrixExt::LAPACK::matSVD(obj, &U, &S, &V);
      lua_push(L, U);
      lua_push(L, S);
      lua_push(L, V);
      return 3;
    }

    BEGIN_METHOD(lt)
    {
      if (lua_is<Matrix<T>*>(L, 1)) {
        Matrix<T> *value = lua_rawto<Matrix<T>*>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Boolean::matLT(obj, value));
      }
      else {
        T value = lua_to<T>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Boolean::matLT(obj, value));
      }
      return 1;
    }

    BEGIN_METHOD(gt)
    {
      if (lua_is<Matrix<T>*>(L, 1)) {
        Matrix<T> *value = lua_rawto<Matrix<T>*>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Boolean::matGT(obj, value));
      }
      else {
        T value = lua_to<T>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Boolean::matGT(obj, value));
      }
      return 1;
    }

    BEGIN_METHOD(eq)
    {
      if (lua_is<Matrix<T>*>(L, 1)) {
        Matrix<T> *value = lua_rawto<Matrix<T>*>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Boolean::matEQ(obj, value));
      }
      else {
        T value = lua_to<T>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Boolean::matEQ(obj, value));
      }
      return 1;
    }

    BEGIN_METHOD(neq)
    {
      if (lua_is<Matrix<T>*>(L, 1)) {
        Matrix<T> *value = lua_rawto<Matrix<T>*>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Boolean::matNEQ(obj, value));
      }
      else {
        T value = lua_to<T>(L, 1);
        lua_push(L, AprilMath::MatrixExt::Boolean::matNEQ(obj, value));
      }
      return 1;
    }

    BEGIN_METHOD(data)
    {
      lua_push(L, obj->getRawDataAccess());
      return 1;
    }
    
    BEGIN_METHOD(order)
    {
      MatrixInt32 *dest = lua_opt<MatrixInt32*>(L, 1, 0);
      dest = AprilMath::MatrixExt::Misc::matOrder(obj, dest);
      lua_push(L, dest);
      return 1;
    }

    BEGIN_METHOD(order_rank)
    {
      MatrixInt32 *dest = lua_opt<MatrixInt32*>(L, 1, 0);
      dest = AprilMath::MatrixExt::Misc::matOrderRank(obj, dest);
      lua_push(L, dest);
      return 1;
    }
    
    BEGIN_METHOD(stringfy)
    {
      OutputLuaStringStream stream(L);
      for (typename Matrix<T>::const_iterator it = obj->begin();
           it != obj->end(); ++it) {
        const T *str = &(*it);
        stream.put(str, sizeof(*it));
      }
      stream.push(L);
      return 1;
    }
    
    BEGIN_METHOD(same_dim)
    {
      if (lua_is<Matrix<T>*>(L,1)) {
        Matrix<T> *other = lua_rawto<Matrix<T>*>(L,1);
        lua_pushboolean(L, obj->sameDim(other));
        return 1;
      }
      else {
        if (!lua_istable(L,1)) {
          LUABIND_ERROR("Expecting a matrix or a table as argument");
        }
        bool result=true;
        int len = luaL_len(L,1);
        if (len != obj->getNumDim()) LUABIND_ERROR("Incorrect number of dimensions");
        const int *dims = obj->getDimPtr();
        for (int i=1; i<=len && result; ++i) {
          lua_rawgeti(L, 1, i);
          int j = lua_toint(L, -1);
          lua_pop(L, 1);
          if (j != dims[i-1]) result=false;
        }
        lua_pushboolean(L, result);
        return 1;
      }
    }

    BEGIN_METHOD(index)
    {
      Matrix<T> *result;
      int dim = lua_to<int>(L,1);
      if (lua_is<Matrix<bool>*>(L,2)) {
        Matrix<bool> *idx = lua_rawto<Matrix<bool>*>(L,2);
        result = AprilMath::MatrixExt::Misc::
          matIndex(obj, dim-1, idx); // WARNING!!! -1
      }
      else {
        AprilUtils::SharedPtr< Matrix<int32_t> > idx;
        if (lua_istable(L,2)) {
          lua_getglobal(L, "matrixInt32"); // dim table ... matrixInt32
          lua_pushvalue(L, 2); // dim table ... matrixInt32 table
          if (lua_pcall(L, 1, 1, 0) != LUA_OK) { // dim table ... object
            AprilUtils::string str(lua_tostring(L,-1));
            lua_pop(L,1);
            ERROR_EXIT1(128, "%s", str.c_str());
          }
          idx = lua_rawto<Matrix<int32_t>*>(L,-1);
          lua_pop(L,1);
        }
        else {
          if (!lua_is<Matrix<int32_t>*>(L,2)) {
            ERROR_EXIT(128, "Expecting matrixInt32 as second argument\n");
          }
          idx = lua_rawto<Matrix<int32_t>*>(L,2);
        }
        result = AprilMath::MatrixExt::Misc::
          matIndex(obj, dim-1, idx.get()); // WARNING!!! -1
      }
      lua_push(L, result);
      return 1;
    }

    BEGIN_METHOD(indexed_fill)
    {
      Matrix<T> *result;
      int dim = lua_to<int>(L,1);
      T val = lua_to<T>(L,3);
      if (lua_is<Matrix<bool>*>(L,2)) {
        Matrix<bool> *idx = lua_rawto<Matrix<bool>*>(L,2);
        result = AprilMath::MatrixExt::Misc::
          matIndexedFill(obj, dim-1, idx, val); // WARNING!!! -1
      }
      else {
        AprilUtils::SharedPtr< Matrix<int32_t> > idx;
        if (lua_istable(L,2)) {
          lua_getglobal(L, "matrixInt32"); // dim table ... matrixInt32
          lua_pushvalue(L, 2); // dim table ... matrixInt32 table
          if (lua_pcall(L, 1, 1, 0) != LUA_OK) { // dim table ... object
            AprilUtils::string str(lua_tostring(L,-1));
            lua_pop(L,1);
            ERROR_EXIT1(128, "%s", str.c_str());
          }
          idx = lua_rawto<Matrix<int32_t>*>(L,-1);
          lua_pop(L,1);
        }
        else {
          idx = lua_to<Matrix<int32_t>*>(L,2);
        }
        result = AprilMath::MatrixExt::Misc::
          matIndexedFill(obj, dim-1, idx.get(), val); // WARNING!!! -1
      }
      lua_push(L, result);
      return 1;
    }

    BEGIN_METHOD(indexed_copy)
    {
      Matrix<T> *result;
      int dim = lua_to<int>(L,1);
      Matrix<T> *other = lua_rawto<Matrix<T>*>(L,3);
      if (lua_is<Matrix<bool>*>(L,2)) {
        Matrix<bool> *idx = lua_rawto<Matrix<bool>*>(L,2);
        result = AprilMath::MatrixExt::Misc::
          matIndexedCopy(obj, dim-1, idx, other); // WARNING!!! -1
      }
      else {
        AprilUtils::SharedPtr< Matrix<int32_t> > idx;
        if (lua_istable(L,2)) {
          lua_getglobal(L, "matrixInt32"); // dim table ... matrixInt32
          lua_pushvalue(L, 2); // dim table ... matrixInt32 table
          if (lua_pcall(L, 1, 1, 0) != LUA_OK) { // dim table ... object
            AprilUtils::string str(lua_tostring(L,-1));
            lua_pop(L,1);
            ERROR_EXIT1(128, "%s", str.c_str());
          }
          idx = lua_rawto<Matrix<int32_t>*>(L,-1);
          lua_pop(L,1);
        }
        else {
          idx = lua_to<Matrix<int32_t>*>(L,2);
        }
        result = AprilMath::MatrixExt::Misc::
          matIndexedCopy(obj, dim-1, idx.get(), other); // WARNING!!! -1
      }
      lua_push(L, result);
      return 1;
    }
          
    BEGIN_CLASS_METHOD(__broadcast__)
    {
      Matrix<T> *a = lua_to<Matrix<T>*>(L,2);
      Matrix<T> *b = lua_to<Matrix<T>*>(L,3);
      Matrix<T> *aux = lua_opt<Matrix<T>*>(L,4,0);
      AprilUtils::SharedPtr< Matrix<T> > result = aux;
      if (lua_isfunction(L,1)) {
        result.reset( AprilMath::MatrixExt::Misc::
                      matBroadcast(LuaFunctionWrapper(L,1),
                                   a, b, result.get()) );
      }
      else if (luaL_getmetafield(L,1,"__call")) {
        result.reset( AprilMath::MatrixExt::Misc::
                      matBroadcast(LuaFunctionWrapper(L,lua_absindex(L,-1)),
                                   a, b, result.get()) );
        lua_pop(L,1);
      }
      else {
        LUABIND_ERROR("Needs a function as first argument");
      }
      lua_push(L, result.get());
      return 1;
    }

    BEGIN_CLASS_METHOD(__call_function__)
    {
      AprilUtils::SharedPtr< Matrix<T> > obj = lua_rawto<Matrix<T>*>(L, 1);
      obj = buildMatrixSliceFromArgs(L, 2, obj.get());
      lua_push(L, obj.get());
      return 1;
    }

    BEGIN_CLASS_METHOD(__index_function__)
    {
      int tt = lua_type(L,2);
      if (tt == LUA_TNUMBER) {
        const Matrix<T> *cobj = lua_rawto<Matrix<T>*>(L, 1);
        const int key = lua_toint(L,2) - 1; // -1 because we start at 0 in C
        if (cobj->getNumDim() > 1) {
          lua_push(L, cobj->select(0, key));
        }
        else {
          lua_push(L, (*cobj)(key));
        }
        return 1;
      }
      else if (tt == LUA_TTABLE) {
        AprilUtils::SharedPtr< Matrix<T> >obj = lua_rawto<Matrix<T>*>(L, 1);
        obj = buildMatrixSliceFromTable(L, 2, obj.get());
        lua_push(L, obj);
        return 1;
      }
      return 0;
    }

    BEGIN_CLASS_METHOD(__newindex_function__)
    {
      AprilUtils::SharedPtr< Matrix<T> > obj = lua_rawto<Matrix<T>*>(L, 1);
      int tk = lua_type(L, 2);
      
      // specialization for case m[k] = v with k and v being of type T
      if (obj->getNumDim() == 1 && tk == LUA_TNUMBER && lua_is<T>(L, 3)) {
        // because we start at 0 in C)
        (*obj)(lua_toint(L, 2) - 1) = lua_to<T>(L, 3);
        return 0; // early stop here
      }
      else if (tk != LUA_TNUMBER && tk != LUA_TTABLE) { // it should be a MatrixBool
        if (!lua_is<MatrixBool*>(L,2)) {
          ERROR_EXIT(128, "Needs a table, a number or a matrixBool as key\n");
        }
        lua_remove(L, 1);
        if (lua_is<T>(L, 2)) {
          MatrixBindings::masked_fill(L, obj.get());
        }
        else {
          MatrixBindings::masked_copy(L, obj.get());
        }
      } // if (lua_is<MatrixBool*>(L, 2))
      else { // not a MatrixBool
        if (tk == LUA_TNUMBER) {
          const int key = lua_toint(L, 2) - 1; // because we start at 0 in C
          if (obj->getNumDim() > 1) {
            obj = obj->select(0, key);
          }
          else {
            if (lua_is<T>(L, 3)) {
              (*obj)(key) = lua_to<T>(L, 3);
            }
            else if (lua_is<Matrix<T>*>(L, 3)) {
              Matrix<T> *other = lua_rawto<Matrix<T>*>(L, 3);
              if (other->size() != 1) {
                ERROR_EXIT(128, "Expecting a one element matrix\n");
              }
              (*obj)(key) = *(other->begin());
            }
            else {
              ERROR_EXIT(128, "Found incorrect type as value");
            }
            return 0; // early return here
          }
        }
        else if (tk == LUA_TTABLE) {
          obj = buildMatrixSliceFromTable(L, 2, obj.get());
          if (obj->size() == 1) {
            if (lua_is<T>(L, 3)) {
              *(obj->begin()) = lua_to<T>(L, 3);
            }
            else if (lua_is<Matrix<T>*>(L, 3)) {
              Matrix<T> *other = lua_rawto<Matrix<T>*>(L, 3);
              if (other->size() != 1) {
                ERROR_EXIT(128, "Expecting a one element matrix\n");
              }
              *(obj->begin()) = *(other->begin());
            }
            else {
              ERROR_EXIT(128, "Found incorrect type as value");
            }
            return 0; // early return here
          }
        }
        else {
          ERROR_EXIT(128, "Needs a table, a number or a matrixBool as key\n");
        }
        if (lua_is<T>(L, 3)) {
          T value = lua_to<T>(L, 3);
          AprilMath::MatrixExt::Initializers::matFill(obj.get(), value);
        }
        else if (lua_is<Matrix<T>*>(L, 3)) {
          Matrix<T> *other = lua_rawto<Matrix<T>*>(L, 3);
          AprilMath::MatrixExt::BLAS::matCopy(obj.get(), other);
        }
        else {
          ERROR_EXIT(128, "Found incorrect type as value");
        }
      } // if key is not a MatrixBool
    
      return 0;      
    } // __newindex_function__
    
  };

#undef FUNCTION_NAME
#undef BEGIN_METHOD
#undef BEGIN_CLASS_METHOD
}

#endif // MATRIX_BINDING_H
