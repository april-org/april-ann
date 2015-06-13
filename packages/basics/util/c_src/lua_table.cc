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
#include "lua_table.h"

namespace AprilUtils {

  LuaTable::LuaTable(lua_State *L) {
    lua_newtable(L);
    init(L, -1);
    lua_pop(L, 1);
  }
  
  LuaTable::LuaTable(lua_State *L, int i) {
    init(L, i);
  }

  LuaTable::LuaTable(const LuaTable &other) {
    other.checkAndPushRef();
    init(other.L, -1);
    lua_pop(L, 1);
  }

  LuaTable::~LuaTable() {
    luaL_unref(L, LUA_REGISTRYINDEX, ref);
  }

  bool LuaTable::empty() const {
    bool result;
    if (!checkAndPushRef()) {
      result = true;
    }
    else {
      lua_pushnil(L);
      if (lua_next(L, -2)) {
        lua_pop(L, 2);
        result = false;
      }
      else {
        result = true;
      }
    }
    lua_pop(L, 1);
    return result;
  }
  
  size_t LuaTable::length() const {
    if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
    int len = luaL_len(L, -1);
    lua_pop(L, 1);
    return static_cast<size_t>(len);
  }

  LuaTable &LuaTable::operator=(const LuaTable &other) {
    luaL_unref(L, LUA_REGISTRYINDEX, ref);
    other.checkAndPushRef();
    init(other.L, -1);
    lua_pop(L, 1);
    return *this;
  }
  
  void LuaTable::init(lua_State *L, int i) {
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

  string LuaTable::toLuaString() {
    /* the function name */
    lua_getglobal(L, "util");
    lua_getfield(L, -1, "to_lua_string");
    pushTable(L);
    lua_pushstring(L, "binary");
    lua_call(L, 2, 1);
    string str(lua_tostring(L,-1));
    lua_pop(L,2);
    return str;
  }
    
  void LuaTable::pushTable(lua_State *L) {
    if (this->L != L) ERROR_EXIT(128, "Given incorrect lua_State\n");
    checkAndPushRef();
  }
  
  bool LuaTable::checkAndPushRef() const {
    if (ref == LUA_NOREF) {
      lua_pushnil(L); // just to be coherent, it always pushes a value
      return false;
    }
    else {
      lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
      return true;
    }
  }

  ///////////////////////////////////////////////////////////////////////////

  template<>
  uint32_t LuaTable::convertTo<uint32_t>(lua_State *L, int idx) {
    return static_cast<uint32_t>(lua_tonumber(L, idx));
  }
  
  template<>
  int LuaTable::convertTo<int32_t>(lua_State *L, int idx) {
    return static_cast<int32_t>(lua_tonumber(L, idx));
  }

  template<>
  float LuaTable::convertTo<float>(lua_State *L, int idx) {
    return static_cast<float>(lua_tonumber(L, idx));
  }

  template<>
  double LuaTable::convertTo<double>(lua_State *L, int idx) {
    return lua_tonumber(L, idx);
  }
  
  template<>
  bool LuaTable::convertTo<bool>(lua_State *L, int idx) {
    return lua_toboolean(L, idx);
  }

  template<>
  const char *LuaTable::convertTo<const char *>(lua_State *L, int idx) {
    UNUSED_VARIABLE(L);
    UNUSED_VARIABLE(idx);
    ERROR_EXIT(128, "Not implemented for 'const char *', use 'string'\n");
    return 0;
  }

  template<>
  string LuaTable::convertTo<string>(lua_State *L, int idx) {
    string aux(lua_tostring(L, idx), luaL_len(L, idx));
    return aux;
  }
  
  template<>
  void LuaTable::pushInto<uint32_t>(lua_State *L, uint32_t value) {
    lua_pushnumber(L, static_cast<double>(value));
  }
  
  template<>
  void LuaTable::pushInto<int32_t>(lua_State *L, int32_t value) {
    lua_pushnumber(L, static_cast<double>(value));
  }

  template<>
  void LuaTable::pushInto<float>(lua_State *L, float value) {
    lua_pushnumber(L, static_cast<double>(value));
  }

  template<>
  void LuaTable::pushInto<double>(lua_State *L, double value) {
    lua_pushnumber(L, value);
  }

  template<>
  void LuaTable::pushInto<bool>(lua_State *L, bool value) {
    lua_pushboolean(L, value);
  }

  template<>
  void LuaTable::pushInto<string>(lua_State *L, string value) {
    size_t len = value.size();
    if (len>0 && value.back() == '\0') {
      april_assert(len-1 == strlen(value.c_str()));
      lua_pushstring(L, value.c_str());
    }
    else {
      lua_pushlstring(L, value.c_str(), len);
    }
  }
  
  template<>
  void LuaTable::pushInto<const string &>(lua_State *L,
                                          const string &value) {
    size_t len = value.size();
    if (len>0 && value.back() == '\0') {
      april_assert(len-1 == strlen(value.c_str()));
      lua_pushstring(L, value.c_str());
    }
    else {
      lua_pushlstring(L, value.c_str(), len);
    }
  }

  template<>
  void LuaTable::pushInto<const char *>(lua_State *L,
                                        const char *value) {
    lua_pushstring(L, value);
  }

  template<>
  bool LuaTable::checkType<uint32_t>(lua_State *L, int idx) {
    return lua_isnumber(L, idx);
  }
  
  template<>
  bool LuaTable::checkType<int32_t>(lua_State *L, int idx) {
    return lua_isnumber(L, idx);
  }
  
  template<>
  bool LuaTable::checkType<float>(lua_State *L, int idx) {
    return lua_isnumber(L, idx);
  }
  
  template<>
  bool LuaTable::checkType<double>(lua_State *L, int idx) {
    return lua_isnumber(L, idx);
  }
  
  template<>
  bool LuaTable::checkType<bool>(lua_State *L, int idx) {
    return lua_isboolean(L, idx);
  }
  
  template<>
  bool LuaTable::checkType<const char *>(lua_State *L, int idx) {
    return lua_isstring(L, idx);
  }
  
  template<>
  bool LuaTable::checkType<string>(lua_State *L, int idx) {
    return lua_isstring(L, idx);
  }
  
  // overload of get for const char *
  template<>
  const char *LuaTable::get<const char *>(const char *name) const {
    if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
    lua_getfield(L, -1, name);
    if (lua_isnil(L,-1)) ERROR_EXIT1(128, "Unable to find field %s\n", name);
    const char *str = lua_tostring(L, -1);
    // NOTE: it is safe to pop because the string is referenced in a table, so,
    // as far as the table exists, the string will also exists.
    lua_pop(L, 2);
    return str;
  }

  // overload of opt for const char *
  template<>
  const char *LuaTable::opt<const char *>(const char *name, const char *def) const {
    if (!checkAndPushRef()) {
      lua_pop(L, 1);
      return def;
    }
    else {
      lua_getfield(L, -1, name);
      if (lua_isnil(L,-1)) {
        lua_pop(L, 2);
        return def;
      }
      const char *str = lua_tostring(L,-1);
      // NOTE: it is safe to pop because the string is referenced in a table,
      // so, as far as the table exists, the string will also exists.
      lua_pop(L, 2);
      return str;
    }
    // return T();
  }
  
}
