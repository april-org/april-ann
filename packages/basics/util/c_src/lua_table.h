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
#ifndef LUA_TABLE_H
#define LUA_TABLE_H
extern "C" {
#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}
#include <typeinfo>

#include "base.h"
#include "error_print.h"
#include "mystring.h"
#include "referenced.h"
#include "unused_variable.h"

namespace AprilUtils {

  /**
   * @brief A class which allow to put and get data into a Lua table.
   */
  class LuaTable {
  public:
        
    LuaTable(lua_State *L = Base::getGlobalLuaState());
    
    LuaTable(lua_State *L, int n);
    
    LuaTable(const LuaTable &other);
    
    ~LuaTable();

    LuaTable &operator=(const LuaTable &other);
    
    template<typename T> LuaTable &put(const char *name, T value) {
      if (!checkAndGetRef()) ERROR_EXIT(128, "Invalid reference\n");
      pushInto(L, value);
      lua_setfield(L, -2, name);
      lua_pop(L, 1);
      return *this;
    }

    template<typename T> T get(const char *name) const {
      if (!checkAndGetRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_getfield(L, -1, name);
      if (lua_isnil(L,-1)) ERROR_EXIT1(128, "Unable to find field %s\n", name);
      T v = convertTo<T>(L, -1);
      lua_pop(L,2);
      return v;
    }
    
    template<typename T> T opt(const char *name, const T def_value = T()) const {
      if (!checkAndGetRef()) {
        lua_pop(L, 1);
        return def_value;
      }
      else {
        lua_getfield(L, -1, name);
        if (lua_isnil(L,-1)) {
          lua_pop(L,2);
          return def_value;
        }
        else {
          T v = convertTo<T>(L, -1);
          lua_pop(L,2);
          return v;
        }
      }
      // return T();
    }
    
    void pushTable(lua_State *L);
    
    /// Converts the value at the top of Lua stack, without removing it.
    template<typename T>
    static T convertTo(lua_State *L, int idx) {
      UNUSED_VARIABLE(L);
      UNUSED_VARIABLE(idx);
      ERROR_EXIT1(128, "NOT IMPLEMENTED FOR TYPE %s\n", typeid(T).name());
    }
    
    /// Pushes a value into the Lua stack.
    template<typename T>
    static void pushInto(lua_State *L, T value) {
      UNUSED_VARIABLE(L);
      UNUSED_VARIABLE(value);
      ERROR_EXIT1(128, "NOT IMPLEMENTED FOR TYPE %s\n", typeid(value).name());
    }
    
  private:
    /// The lua_State where the table is allocated.
    mutable lua_State *L;
    /// The reference in the registry where the table can be retrieved.
    int ref;
    
    /// Auxiliary method to simplify constructors.
    void init(lua_State *L, int i);
    /// Checks ref != LUA_NOREF and pushes it into the Lua stack.
    bool checkAndGetRef() const;
    
  };

  // Basic data types specializations.
  template<> int LuaTable::convertTo<int>(lua_State *L, int idx);
  template<> float LuaTable::convertTo<float>(lua_State *L, int idx);
  template<> double LuaTable::convertTo<double>(lua_State *L, int idx);
  template<> bool LuaTable::convertTo<bool>(lua_State *L, int idx);
  template<> string LuaTable::convertTo<string>(lua_State *L, int idx);
  template<> LuaTable LuaTable::convertTo<LuaTable>(lua_State *L, int idx);
  
  template<> void LuaTable::pushInto<int>(lua_State *L, int value);
  template<> void LuaTable::pushInto<float>(lua_State *L, float value);
  template<> void LuaTable::pushInto<double>(lua_State *L, double value);
  template<> void LuaTable::pushInto<bool>(lua_State *L, bool value);
  template<> void LuaTable::pushInto<const string &>(lua_State *L,
                                                     const string &value);
  template<> void LuaTable::pushInto<const char *>(lua_State *L,
                                                   const char *value);
  template<> void LuaTable::pushInto<LuaTable>(lua_State *L, LuaTable value);
  
} // namespace AprilUtils

#endif // LUA_TABLE_H
