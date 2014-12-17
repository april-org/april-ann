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
   *
   * This class allow to access from C++ to Lua values stored into a Lua table.
   * The Lua table is stored at the C registry, and will be remove in the
   * destructor. Template methods for put(), get(), and opt() operations are
   * defined. It is possible to specialize this methods in new C++ types by
   * specializing the static template methods: convertTo(), pushInto() and
   * checkType().
   *
   * @note All keys are forced to be strings, so, it works with dictionary style
   * Lua tables, not with arrays.
   */
  class LuaTable {
  public:
    
    /// Constructor for a new LuaTable in the registry.
    LuaTable(lua_State *L = Base::getGlobalLuaState());
    
    /// Constructor for a LuaTable in a given Lua stack position.
    LuaTable(lua_State *L, int n);
    
    /// Copy constructor.
    LuaTable(const LuaTable &other);
    
    /// Destructor.
    ~LuaTable();

    /// Copy operator.
    LuaTable &operator=(const LuaTable &other);
    
    /// Returns a C++ string with the Lua representation of the table.
    string toLuaString();
    
    /// Puts a new value into the table, using the given key name.
    template<typename T>
    LuaTable &put(const string &name, T value) {
      return put<T>(name.c_str(), value);
    }
    
    /// Checks if the field at the given key name is nil.
    bool checkNil(const string &name) const {
      return checkNil(name.c_str());
    }

    /// Checks if the field at the given key name is of the given type (a nil
    /// value will be taken as true).
    template<typename T>
    bool checkNilOrType(const string &name) const {
      return checkNilOrType<T>(name.c_str());
    }

    /// Returns the value stored at the given key name field.
    template<typename T>
    T get(const string &name) const {
      return get<T>(name.c_str());
    }

    /// Returns the value stored at the given key name field. In case the field
    /// is empty, it returns the given def_value argument.
    template<typename T>
    T opt(const string &name, const T def_value = T()) const {
      return opt<T>(name.c_str(), def_value);
    }

    /// Puts a new value into the table, using the given key name.
    template<typename T>
    LuaTable &put(const char *name, T value) {
      if (!checkAndGetRef()) ERROR_EXIT(128, "Invalid reference\n");
      pushInto(L, value);
      lua_setfield(L, -2, name);
      lua_pop(L, 1);
      return *this;
    }

    /// Checks if the field at the given key name is nil.    
    bool checkNil(const char *name) const {
      if (!checkAndGetRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_getfield(L, -1, name);
      bool ret =  lua_isnil(L, -1);
      lua_pop(L, 2);
      return ret;
    }

    /// Checks if the field at the given key name is of the given type (a nil
    /// value will be taken as true).
    template<typename T>
    bool checkNilOrType(const char *name) const {
      if (!checkAndGetRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_getfield(L, -1, name);
      bool ret = lua_isnil(L, -1) || checkType<T>(L, -1);
      lua_pop(L, 2);
      return ret;
    }

    /// Returns the value stored at the given key name field.    
    template<typename T>
    T get(const char *name) const {
      if (!checkAndGetRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_getfield(L, -1, name);
      if (lua_isnil(L,-1)) ERROR_EXIT1(128, "Unable to find field %s\n", name);
      if (!checkType<T>(L, -1)) ERROR_EXIT(128, "Incorrect type\n");
      T v = convertTo<T>(L, -1);
      lua_pop(L,2);
      return v;
    }

    /// Returns the value stored at the given key name field. In case the field
    /// is empty, it returns the given def_value argument.    
    template<typename T>
    T opt(const char *name, const T def_value = T()) const {
      if (!checkAndGetRef()) {
        lua_pop(L, 1);
        return def_value;
      }
      else {
        lua_getfield(L, -1, name);
        T v(def_value);
        if (!lua_isnil(L,-1)) {
          if (!checkType<T>(L, -1)) ERROR_EXIT(128, "Incorrect type\n");
          v = convertTo<T>(L, -1);
        }
        lua_pop(L,2);
        return v;
      }
      // return T();
    }
    
    /// Pushes into Lua stack the Lua table associated with the object.
    void pushTable(lua_State *L);
    
    /// Converts the value at the given Lua stack index, without removing it.
    template<typename T>
    static T convertTo(lua_State *L, int idx) {
      UNUSED_VARIABLE(L);
      UNUSED_VARIABLE(idx);
      ERROR_EXIT1(128, "NOT IMPLEMENTED FOR TYPE %s\n", typeid(T).name());
      return T();
    }
    
    /// Pushes a value into the Lua stack.
    template<typename T>
    static void pushInto(lua_State *L, T value) {
      UNUSED_VARIABLE(L);
      UNUSED_VARIABLE(value);
      ERROR_EXIT1(128, "NOT IMPLEMENTED FOR TYPE %s\n", typeid(value).name());
    }
    
    /// Checks the expected type of the value at the given Lua stack index.
    template<typename T>
    static bool checkType(lua_State *L, int idx) {
      UNUSED_VARIABLE(L);
      UNUSED_VARIABLE(idx);
      ERROR_EXIT1(128, "NOT IMPLEMENTED FOR TYPE %s\n", typeid(T).name());
      return false;
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
  template<> const char *LuaTable::convertTo<const char *>(lua_State *L, int idx);
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

  template<> bool LuaTable::checkType<int>(lua_State *L, int idx);
  template<> bool LuaTable::checkType<float>(lua_State *L, int idx);
  template<> bool LuaTable::checkType<double>(lua_State *L, int idx);
  template<> bool LuaTable::checkType<bool>(lua_State *L, int idx);
  template<> bool LuaTable::checkType<const char *>(lua_State *L, int idx);
  template<> bool LuaTable::checkType<string>(lua_State *L, int idx);
  template<> bool LuaTable::checkType<LuaTable>(lua_State *L, int idx);

  // overload of get for const char *
  template<>
  const char *LuaTable::get<const char *>(const char *name) const;
  
  // overload of opt for const char *
  template<>
  const char *LuaTable::opt<const char *>(const char *name, const char *def) const;
  
} // namespace AprilUtils

#endif // LUA_TABLE_H
