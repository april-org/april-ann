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
    other.checkAndGetRef();
    init(other.L, -1);
    lua_pop(L, 1);
  }

  LuaTable::~LuaTable() {
    luaL_unref(L, LUA_REGISTRYINDEX, ref);
  }

  LuaTable &LuaTable::operator=(const LuaTable &other) {
    luaL_unref(L, LUA_REGISTRYINDEX, ref);
    other.checkAndGetRef();
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
    
  void LuaTable::pushTable(lua_State *L) {
    if (this->L != L) ERROR_EXIT(128, "Given incorrect lua_State\n");
    checkAndGetRef();
  }
  
  bool LuaTable::checkAndGetRef() const {
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
  int LuaTable::convertTo<int>(lua_State *L, int idx) {
    return lua_tointeger(L, idx);
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
  string LuaTable::convertTo<string>(lua_State *L, int idx) {
    string aux(lua_tostring(L, idx), luaL_len(L, idx));
    return aux;
  }
  
  template<>
  void LuaTable::pushInto<int>(lua_State *L, int value) {
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
  void LuaTable::pushInto<const string &>(lua_State *L,
                                          const string &value) {
    lua_pushlstring(L, value.c_str(), value.size());
  }

  template<>
  void LuaTable::pushInto<const char *>(lua_State *L,
                                        const char *value) {
    lua_pushstring(L, value);
  }
}
