/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}

#include "error_print.h"
#include "generic_options.h"

namespace april_utils {
  
  /************************************************************************/
  static char lua_isCHAR(lua_State *L, int i) {
    return lua_isstring(L,i) && luaL_len(L,i) == 1;
  }
  static char lua_toCHAR(lua_State *L, int i) {
    const char *str = lua_tostring(L,i);
    return str[0];
  }
  static void lua_pushCHAR(lua_State *L, char c) {
    lua_pushlstring(L, &c, 1u);
  }
  /************************************************************************/
  
  /************************************************************************/  
  static Referenced *april_rawgetReferenced(lua_State *L, int index) {
    Referenced **pre_obj = static_cast<Referenced**>(lua_touserdata(L,index));
    Referenced *obj = 0;
    if (pre_obj != 0) obj = (*pre_obj);
    return obj;
  }
  
  static int april_deleteReferenced(lua_State *L) {
    // stops garbage collector to avoid problems with reference counting
    lua_gc(L, LUA_GCSTOP, 0);
    Referenced *obj = april_rawgetReferenced(L,1);
    if (obj != 0) {
      DecRef(obj);
    }
    // restart the garbage collector
    lua_gc(L, LUA_GCRESTART, 0);
    return 0;
  }
  
  static void april_pushReferenced(lua_State *L, Referenced *obj) {
    IncRef(obj);
    Referenced **ptr;
    ptr = static_cast<Referenced**>
      (lua_newuserdata(L,sizeof(Referenced*)) );
    *ptr = obj;
    // stack: ptr
    lua_newtable(L); // a metatable
    // stack: ptr mt
    lua_pushcfunction(L, april_deleteReferenced);
    // stack: ptr mt func
    lua_setfield(L,-2,"__gc");
    // stack: ptr mt
    lua_setmetatable(L,-2);
    // stack: ptr
  }
  
  static Referenced *april_toReferenced(lua_State *L, int index) {
    return april_rawgetReferenced(L,index);
  }
  /************************************************************************/
  
  // int HashTableOptions::pushToLua(lua_State *L, const char *name) {
  //   Value *v = dict.find(name);
  //   if (v == 0) lua_pushnil(L);
  //   switch(v->type) {
  //   case DOUBLE:
  //     lua_pushnumber(L,v->dbl);
  //     break;
  //   case FLOAT:
  //     lua_pushnumber(L,static_cast<double>(v->flt));
  //     break;
  //   case CHAR:
  //     lua_pushCHAR(L,v->chr);
  //     break;
  //   case STRING:
  //     lua_pushstring(L,v->str.c_str());
  //     break;
  //   case INT32:
  //     lua_pushnumber(L,static_cast<double>(v->i32));
  //     break;
  //   case UINT32:
  //     lua_pushnumber(L,static_cast<double>(v->u32));
  //     break;
  //   case INT64:
  //     lua_pushnumber(L,static_cast<double>(v->i64));
  //     break;
  //   case UINT64:
  //     lua_pushnumber(L,static_cast<double>(v->u64));
  //     break;
  //   case BOOL:
  //     lua_pushboolean(L,v->bl);
  //     break;
  //   case REFERENCED:
  //     april_pushReferenced(L,v->ref_ptr.get());
  //     break;
  //   default:
  //     ERROR_EXIT(128, "Unknown type\n");;
  //   }
  //   return 1;
  // }

#define METHODS_IMPLEMENTATION(method, c_type, enum_type, value_name)   \
  GenericOptions *HashTableOptions::put##method(const char *name, c_type value) { \
    Value v;                                                            \
    v.value_name = value;                                               \
    v.type = enum_type;                                                 \
    dict[name] = v;                                                     \
    return this;                                                        \
  }                                                                     \
  c_type HashTableOptions::get##method(const char *name) const {        \
    Value *v = dict.find(name);                                         \
    if (v == 0) ERROR_EXIT1(128, "Unable to find field %s\n", name);    \
    if (v->type != enum_type) {                                         \
      ERROR_EXIT(128, "Unexpected type error\n");                       \
    }                                                                   \
    return v->value_name;                                               \
  }                                                                     \
  c_type HashTableOptions::getOptional##method(const char *name,        \
                                               c_type const opt) const { \
    Value *v = dict.find(name);                                         \
    if (v == 0) return opt;                                             \
    else if (v->type != enum_type) {                                    \
      ERROR_EXIT(128, "Unexpected type error\n");                       \
    }                                                                   \
    return v->value_name;                                               \
  }
  
  METHODS_IMPLEMENTATION(Double, double, DOUBLE, dbl);
  METHODS_IMPLEMENTATION(Float, float, FLOAT, flt);
  METHODS_IMPLEMENTATION(Char, char, CHAR, chr);
  METHODS_IMPLEMENTATION(String, const char *, STRING, str);
  METHODS_IMPLEMENTATION(Int32, int32_t, INT32, i32);
  METHODS_IMPLEMENTATION(UInt32, uint32_t, UINT32, u32);
  METHODS_IMPLEMENTATION(Int64, int64_t, INT64, i64);
  METHODS_IMPLEMENTATION(UInt64, uint64_t, UINT64, u64);
  METHODS_IMPLEMENTATION(Boolean, bool, BOOL, bl);
  
#undef METHODS_IMPLEMENTATION

  GenericOptions *HashTableOptions::putReferenced(const char *name,
                                                  Referenced *value) {
    Value v;
    v.ref_ptr = value;
    v.type = REFERENCED;
    dict[name] = v;
    return this;
  }
  
  Referenced *HashTableOptions::privateGetReferenced(const char *name) const {
    Value *v = dict.find(name);
    if (v == 0) return 0;
    if (v->type != REFERENCED) {
      ERROR_EXIT(128, "Unexpected type error\n");
    }
    return v->ref_ptr.get();
  }

  ////////////////////////////////////////////////////////////////////////////
  
  LuaTableOptions::LuaTableOptions(lua_State *L) : GenericOptions() {
    lua_newtable(L);
    init(L, -1);
    lua_pop(L,1);
  }
  
  LuaTableOptions::LuaTableOptions(lua_State *L, int i) : GenericOptions() {
    init(L, i);
  }
  
  LuaTableOptions::~LuaTableOptions() {
    luaL_unref(L, LUA_REGISTRYINDEX, ref);
  }

  void LuaTableOptions::init(lua_State *L, int i) {
    this->L = L;
    if (lua_isnil(L,i) || lua_type(L,i) == LUA_TNONE) ref = LUA_NOREF;
    else {
      lua_pushvalue(L,i);
      if (!lua_istable(L,-1)) {
        ERROR_EXIT1(128,"Expected a table parameter at pos %d\n", i);
      }
      this->ref = luaL_ref(L, LUA_REGISTRYINDEX);
    }
  }
  
  // int LuaTableOptions::pushToLua(lua_State *L, const char *name) {
  //   if (ref == LUA_NOREF) ERROR_EXIT(128, "Unable to get options\n");
  //   lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
  //   lua_getfield(L, -1, name);
  //   return 1;
  // }

#define METHODS_IMPLEMENTATION(method, c_type, lua_type)                \
  GenericOptions *LuaTableOptions::put##method(const char *name, c_type value) { \
    if (ref == LUA_NOREF) ERROR_EXIT(128, "Unable to put options\n");   \
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref);                             \
    lua_push##lua_type(L, value);                                       \
      lua_setfield(L, -2, name);                                        \
      lua_pop(L,1);                                                     \
      return this;                                                      \
  }                                                                     \
  c_type LuaTableOptions::get##method(const char *name) const {         \
    if (ref == LUA_NOREF) ERROR_EXIT(128, "Unable to get options\n");   \
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref);                             \
    lua_getfield(L, -1, name);                                          \
    if (lua_isnil(L,-1)) ERROR_EXIT1(128, "Unable to find field %s\n", name); \
    if (!lua_is##lua_type(L, -1)) {                                     \
      ERROR_EXIT(128, "Unexpected type error\n");                       \
    }                                                                   \
    c_type v = static_cast<c_type>(lua_to##lua_type(L,-1));             \
    lua_pop(L,2);                                                       \
    return v;                                                           \
  }                                                                     \
  c_type LuaTableOptions::getOptional##method(const char *name,         \
                                              c_type const opt) const { \
    if (ref == LUA_NOREF) return opt;                                   \
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref);                             \
    lua_getfield(L, -1, name);                                          \
    if (lua_isnil(L,-1)) {                                              \
      lua_pop(L,2);                                                     \
      return opt;                                                       \
    }                                                                   \
    if (!lua_is##lua_type(L, -1)) {                                     \
      ERROR_EXIT(128, "Unexpected type error\n");                       \
    }                                                                   \
    c_type v = static_cast<c_type>(lua_to##lua_type(L,-1));             \
    lua_pop(L,2);                                                       \
    return v;                                                           \
  }

  METHODS_IMPLEMENTATION(Double, double, number);
  METHODS_IMPLEMENTATION(Float, float, number);
  METHODS_IMPLEMENTATION(Char, char, CHAR);
  METHODS_IMPLEMENTATION(String, const char *, string);
  METHODS_IMPLEMENTATION(Int32, int32_t, number);
  METHODS_IMPLEMENTATION(UInt32, uint32_t, number);
  METHODS_IMPLEMENTATION(Int64, int64_t, number);
  METHODS_IMPLEMENTATION(UInt64, uint64_t, number);
  METHODS_IMPLEMENTATION(Boolean, bool, boolean);
  
#undef METHODS_IMPLEMENTATION

  GenericOptions *LuaTableOptions::putReferenced(const char *name,
                                                 Referenced *value) {
    if (ref == LUA_NOREF) ERROR_EXIT(128, "Unable to put options\n");
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
    april_pushReferenced(L, value);
    lua_setfield(L, -2, name);
    lua_pop(L,1);
    return this;
  }
  
  Referenced *LuaTableOptions::privateGetReferenced(const char *name) const {
    if (ref == LUA_NOREF) ERROR_EXIT(128, "Unable to get options\n");
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
    lua_getfield(L, -1, name);
    if (lua_isnil(L,-1)) {
      lua_pop(L,2);
      return 0;
    }
    Referenced *v = april_toReferenced(L,-1);
    if (v == 0) {
      ERROR_EXIT(128, "Unexpected type error\n");
    }
    lua_pop(L,2);
    return v;
  }
  
} // namespace april_utils
