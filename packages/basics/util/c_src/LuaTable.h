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
#include <typeinfo>
#include "base.h"
#include "error_print.h"
#include "my_string.h"
#include "referenced.h"
#include "unused_variable.h"

namespace AprilUtils {

  /**
   * @brief A class which allow to put and get data into a Lua table.
   */
  class LuaTable {
  public:
    
    /// Converts the value at the top of Lua stack, without removing it.
    template<typename T>
    static T convertTo(lua_State *L) {
      ERROR_EXIT1(128, "NOT IMPLEMENTED FOR TYPE %s\n",  typeid(T).name());
    }
    
    /// Pushes a value into the Lua stack.
    template<typename T>
    static void pushInto(lua_State *L, T value) {
      ERROR_EXIT1(128, "NOT IMPLEMENTED FOR TYPE %s\n",  typeid(T).name());
    }
    
    static void registarLuaState(lua_State *l) { L = l; }
    
    LuaTable(lua_State *L = Base::getGlobalLuaState());
    
    LuaTable(lua_State *L, int n);
    
    LuaTable(const LuaTable &other);
    
    ~LuaTable();

    LuaTable &operator=(const LuaTable &other);
    
    template<typename T>
    LuaTable &put(const char *name, T value) {
      checkAndGetRef();
      pushInto(L, T);
      lua_setfield(L, -2, name);
      lua_pop(L, 1);
      return *this;
    }

    template<typename T>
    T get(const char *name) {
      checkAndGetRef();
      lua_getfield(L, -1, name);
      if (lua_isnil(L,-1)) ERROR_EXIT(128, "Unable to find field %s\n", name);
      T v = convertTo<T>(L);
      lua_pop(L,2);
      return v;
    }
    
    template<typename T>
    T opt(const char *name, const T def_value) {
      checkAndGetRef();
      lua_getfield(L, -1, name);
      if (lua_isnil(L,-1)) return def_value;
      T v = convertTo<T>(L);
      lua_pop(L,2);
      return v;
    }
    
    void pushTable(lua_State *L) {
      if (this->L != L) ERROR_EXIT(128, "Given incorrect lua_State\n");
    }
    
  private:
    /// The lua_State where the table is allocated.
    lua_State *L;
    /// The reference in the registry where the table can be retrieved.
    int ref;
    
    /// Auxiliary method to simplify constructors.
    void init(lua_State *L, int i);
    /// Checks ref != LUA_NOREF and pushes it into the Lua stack.
    void checkAndGetRef();
    
  };

  // Basic data types specializations.
  template<>
  static int convertTo<int>(lua_State *L);
  template<>
  static float convertTo<float>(lua_State *L);
  template<>
  static double convertTo<double>(lua_State *L);
  template<>
  static bool convertTo<bool>(lua_State *L);
  template<>
  static string convertTo<string>(lua_State *L);
  
  template<>
  static void pushInto<int>(lua_State *L, int value);
  template<>
  static void pushInto<float>(lua_State *L, float value);
  template<>
  static void pushInto<double>(lua_State *L, double value);
  template<>
  static void pushInto<bool>(lua_State *L, bool value);
  template<>
  static void pushInto<string>(lua_State *L, const string &value);

} // namespace AprilUtils
