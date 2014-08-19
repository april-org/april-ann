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

#define METHODS_IMPLEMENTATION(method, c_type, enum_type, value_name)   \
  GenericOptions *HashTableOptions::put##method(const char *name, c_type value) {  \
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
      ERROR_EXIT(128, "Unexpected types error\n");                      \
    }                                                                   \
    return v->value_name;                                               \
  }                                                                     \
  c_type HashTableOptions::getOptional##method(const char *name,        \
                                               c_type const opt) const { \
    Value *v = dict.find(name);                                         \
    if (v == 0) return opt;                                             \
    else if (v->type != enum_type) {                                    \
      ERROR_EXIT(128, "Unexpected types error\n");                      \
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
  
#define METHODS_IMPLEMENTATION(method, c_type, lua_type)                \
  GenericOptions *LuaTableOptions::put##method(const char *name, c_type value) {   \
    if (ref == LUA_NOREF) ERROR_EXIT(128, "Unable to put options\n");   \
    lua_push##lua_type(L, value);                                       \
    lua_setfield(L, ref, name);                                         \
    return this;                                                        \
  }                                                                     \
  c_type LuaTableOptions::get##method(const char *name) const {         \
    if (ref == LUA_NOREF) ERROR_EXIT(128, "Unable to get options\n");   \
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref);                             \
    lua_getfield(L, -1, name);                                          \
    if (lua_isnil(L,-1)) ERROR_EXIT1(128, "Unable to find field %s\n", name); \
    if (!lua_is##lua_type(L, -1)) {                                     \
      ERROR_EXIT(128, "Unexpected types error\n");                      \
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
      ERROR_EXIT(128, "Unexpected types error\n");                      \
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
  
} // namespace april_utils
