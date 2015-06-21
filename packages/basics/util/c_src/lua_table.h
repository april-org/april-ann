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
#include <stdint.h>

#include "lauxlib.h"
#include "lualib.h"
#include "lua.h"
}
#include <typeinfo>

#include "base.h"
#include "error_print.h"
#include "mystring.h"
#include "referenced.h"
#include "smart_ptr.h"
#include "unused_variable.h"

/**
 * @brief Macro for declaration of template specializations for
 * AprilUtils::LuaTable class.
 *
 * Example of use:
 * @code
 * DECLARE_LUA_TABLE_BIND_SPECIALIZATION(Basics::MatrixFloat);
 * @endcode
 *
 * @note This macro accepts any argument, even with namespaces.
 */
#define DECLARE_LUA_TABLE_BIND_SPECIALIZATION(type)                     \
  namespace AprilUtils {                                                \
    template<> type *LuaTable::convertTo<type *>(lua_State *L, int idx); \
    template<> void LuaTable::pushInto<type *>(lua_State *L, type *value); \
    template<> bool LuaTable::checkType<type *>(lua_State *L, int idx); \
  }

/**
 * @brief Macro for implementation of template specializations for
 * AprilUtils::LuaTable class.
 *
 * @note This macro doesn't accepts arguments with namespaces.
 *
 * This macro is normally called in C++ bindings. Example of use:
 * @code
 * using Basics::MatrixFloat; // requires namespace using
 * ...
 * IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(MatrixFloat);
 * @endcode
 */
#define IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION(type)                   \
  namespace AprilUtils {                                                \
    template<> type *LuaTable::convertTo<type *>(lua_State *L, int idx) { \
      return lua_to##type(L, idx);                                      \
    }                                                                   \
    template<> void LuaTable::pushInto<type *>(lua_State *L, type *value) { \
      lua_push##type(L, value);                                         \
    }                                                                   \
    template<> bool LuaTable::checkType<type *>(lua_State *L, int idx) { \
      return lua_is##type(L, idx);                                      \
    }                                                                   \
  }

namespace AprilUtils {

  /**
   * @brief A class which allow to put and get data into a Lua table.
   *
   * This class allow to access from C++ to Lua values stored into a Lua table.
   * The Lua table is stored at the C registry, and will be remove in the
   * destructor. Template methods for put(), get(), and opt() operations are
   * defined. It is possible to specialize this methods in new C++ types by
   * specializing the static template methods: convertTo(), pushInto() and
   * checkType(). Operator[] has been overloaded to allow array and
   * dictionary access, as the following code.
   * @code
   * table["foo"] = bar1;
   * table[1] = bar2;
   * bar1 = table["foo"].get<BarType>();
   * bar1 = table["foo"].opt<BarType>(DefaultFooValue);
   * bar2 = table[1].opt<BarType>(DefaultFooValue);
   * @endcode
   *
   * @note Keys can be strings and or integers.
   *
   * @note Static template methods convertTo(), pushInto() and checkType() can
   * be used in C++ code to operate with Lua end.
   *
   * @see IMPLEMENT_LUA_TABLE_BIND_SPECIALIZATION and
   * DECLARE_LUA_TABLE_BIND_SPECIALIZATION macros.
   */
  class LuaTable {
    
    /**
     * @brief Implements a helper for left hand side operator[] of LuaTable
     * class.
     *
     * This helper allow to use LuaTable in the following way:
     *
     * @code
     * table["foo"] = bar;
     * bar = table["foo"].get<BarType>();
     * bar = table["foo"].opt<BarType>(DefaultFooValue);
     * @endcode
     */
    class LHSAccessorByString {
    public:
      LHSAccessorByString(LuaTable *table, const char * const name) :
        table(table),name(name) {}
      LHSAccessorByString(const LHSAccessorByString &other) :
        table(other.table),name(other.name) {}
      template<typename T>
      LuaTable &operator=(T value) {
        return table->put<T>(name, value);
      }
      template<typename T>
      T get() const {
        return table->get<T>(name);
      }
      template<typename T>
      T opt(const T def_value) const {
        return table->opt<T>(name, def_value);
      }
      template<typename T>
      bool is() const {
        return table->checkNilOrType<T>(name);
      }
    private:
      LuaTable *table;
      const char * const name;
    };

    /**
     * @brief Implements a helper for left hand side operator[] of LuaTable
     * class.
     *
     * This helper allow to use LuaTable in the following way:
     *
     * @code
     * table[1] = bar;
     * bar = table[1].get<BarType>();
     * bar = table[1].opt<BarType>(DefaultValue);
     * @endcode
     */
    class LHSAccessorByInteger {
    public:
      LHSAccessorByInteger(LuaTable *table, int n) :
        table(table),n(n) {}
      LHSAccessorByInteger(const LHSAccessorByInteger &other) :
        table(other.table),n(other.n) {}
      template<typename T>
      LuaTable &operator=(T value) {
        return table->put<T>(n, value);
      }
      template<typename T>
      T get() const {
        return table->get<T>(n);
      }
      template<typename T>
      T opt(const T def_value) const {
        return table->opt<T>(n, def_value);
      }
      template<typename T>
      bool is() const {
        return table->checkNilOrType<T>(n);
      }
    private:
      LuaTable *table;
      const int n;
    };

    /**
     * @brief Implements a helper for right hand side operator[] of LuaTable
     * class.
     *
     * This helper allow to use LuaTable in the following way:
     *
     * @code
     * bar = table["foo"].get<BarType>();
     * bar = table["foo"].opt<BarType>(DefaultFooValue);
     * @endcode
     */
    class RHSAccessorByString {
    public:
      RHSAccessorByString(const LuaTable *table, const char * const name) :
        table(table),name(name) {}
      RHSAccessorByString(const RHSAccessorByString &other) :
        table(other.table),name(other.name) {}
      template<typename T>
      T get() const {
        return table->get<T>(name);
      }
      template<typename T>
      T opt(const T def_value) const {
        return table->opt<T>(name, def_value);
      }
      template<typename T>
      bool is() const {
        return table->checkNilOrType<T>(name);
      }
    private:
      const LuaTable *table;
      const char * const name;
    };

    /**
     * @brief Implements a helper for right hand side operator[] of LuaTable
     * class.
     *
     * This helper allow to use LuaTable in the following way:
     *
     * @code
     * bar = table[1].get<BarType>();
     * bar = table[1].opt<BarType>(DefaultFooValue);
     * @endcode
     */
    class RHSAccessorByInteger {
    public:
      RHSAccessorByInteger(const LuaTable *table, int n) :
        table(table),n(n) {}
      RHSAccessorByInteger(const RHSAccessorByInteger &other) :
        table(other.table),n(other.n) {}
      template<typename T>
      T get() const {
        return table->get<T>(n);
      }
      template<typename T>
      T opt(const T def_value) const {
        return table->opt<T>(n, def_value);
      }
      template<typename T>
      bool is() const {
        return table->checkNilOrType<T>(n);
      }
    private:
      const LuaTable *table;
      const int n;
    };
    
  public:
    
    /// Constructor for a new LuaTable in the registry.
    LuaTable(lua_State *L = Base::getGlobalLuaState());

    /// Constructor for a new LuaTable in the registry containing C array data.
    template<typename T>
    LuaTable(const T *vec, size_t len,
             lua_State *L = Base::getGlobalLuaState()) {
      lua_newtable(L);
      init(L, -1);
      for (unsigned int i=0; i<len; ++i) {
        pushInto<T>(L, vec[i]);
        lua_rawseti(L, -2, i+1);
      }
      lua_pop(L, 1);
    }
    
    /// Constructor for a LuaTable in a given Lua stack position.
    LuaTable(lua_State *L, int n);
    
    /// Copy constructor.
    LuaTable(const LuaTable &other);
    
    /// Destructor.
    ~LuaTable();
    
    /// Indicates if the LuaTable is empty.
    bool empty() const;
    
    /// Executes length operator in Lua
    size_t length() const;
    
    /// Copy operator.
    LuaTable &operator=(const LuaTable &other);

    /// See RHSAccessorByString
    RHSAccessorByString operator[](const char *name) const {
      return RHSAccessorByString(this, name);
    }

    /// See RHSAccessorByString
    RHSAccessorByString operator[](const string &name) const {
      return RHSAccessorByString(this, name.c_str());
    }

    /// See RHSAccessorByInteger
    RHSAccessorByInteger operator[](int n) const {
      return RHSAccessorByInteger(this, n);
    }

    /// See LHSAccessorByString
    LHSAccessorByString operator[](const char *name) {
      return LHSAccessorByString(this, name);
    }

    /// See LHSAccessorByString
    LHSAccessorByString operator[](const string &name) {
      return LHSAccessorByString(this, name.c_str());
    }
    
    /// See LHSAccessorByInteger
    LHSAccessorByInteger operator[](int n) {
      return LHSAccessorByInteger(this, n);
    }
    
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
    T opt(const string &name, const T def_value) const {
      return opt<T>(name.c_str(), def_value);
    }

    /// Puts a new value into the table, using the given key name.
    template<typename T>
    LuaTable &put(const char *name, T value) {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      pushInto(L, value);
      lua_setfield(L, -2, name);
      lua_pop(L, 1);
      return *this;
    }
    
    /// Puts a new value into the table, using the given key number.
    template<typename T>
    LuaTable &put(int n, T value) {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushnumber(L, n);
      pushInto(L, value);
      lua_settable(L, -3);
      lua_pop(L, 1);
      return *this;
    }

    /// Puts a new value into the table, using the given key name.
    template<typename T>
    LuaTable &put(const char *name, size_t len, T value) {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushlstring(L, name, len);
      pushInto(L, value);
      lua_settable(L, -3);
      lua_pop(L, 1);
      return *this;
    }

    /// Checks if the field at the given key name is nil.    
    bool checkNil(const char *name) const {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_getfield(L, -1, name);
      bool ret =  lua_isnil(L, -1);
      lua_pop(L, 2);
      return ret;
    }

    /// Checks if the field at the given key number is nil.
    bool checkNil(int n) const {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushnumber(L, n);
      lua_gettable(L, -2);
      bool ret =  lua_isnil(L, -1);
      lua_pop(L, 2);
      return ret;
    }

    /// Checks if the field at the given key name is nil.    
    bool checkNil(const char *name, size_t len) const {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushlstring(L, name, len);
      lua_gettable(L, -2);
      bool ret =  lua_isnil(L, -1);
      lua_pop(L, 2);
      return ret;
    }

    /// Checks if the field at the given key name is of the given type (a nil
    /// value will be taken as true).
    template<typename T>
    bool checkNilOrType(const char *name) const {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_getfield(L, -1, name);
      bool ret = lua_isnil(L, -1) || checkType<T>(L, -1);
      lua_pop(L, 2);
      return ret;
    }

    /// Checks if the field at the given key number is of the given type (a nil
    /// value will be taken as true).
    template<typename T>
    bool checkNilOrType(int n) const {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushnumber(L, n);
      lua_gettable(L, -2);
      bool ret = lua_isnil(L, -1) || checkType<T>(L, -1);
      lua_pop(L, 2);
      return ret;
    }

    /// Checks if the field at the given key name is of the given type (a nil
    /// value will be taken as true).
    template<typename T>
    bool checkNilOrType(const char *name, size_t len) const {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushlstring(L, name, len);
      lua_gettable(L, -2);
      bool ret = lua_isnil(L, -1) || checkType<T>(L, -1);
      lua_pop(L, 2);
      return ret;
    }

    /// Returns the value stored at the given key name field.    
    template<typename T>
    T get(const char *name) const {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_getfield(L, -1, name);
      if (lua_isnil(L,-1)) ERROR_EXIT1(128, "Unable to find field %s\n", name);
      if (!checkType<T>(L, -1)) ERROR_EXIT(128, "Incorrect type\n");
      T v = convertTo<T>(L, -1);
      lua_pop(L,2);
      return v;
    }

    /// Returns the value stored at the given key number field.
    template<typename T>
    T get(int n) const {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushnumber(L, n);
      lua_gettable(L, -2);
      if (lua_isnil(L,-1)) ERROR_EXIT1(128, "Unable to find field %d\n", n);
      if (!checkType<T>(L, -1)) ERROR_EXIT(128, "Incorrect type\n");
      T v = convertTo<T>(L, -1);
      lua_pop(L,2);
      return v;
    }

    /// Returns the value stored at the given key name field.
    template<typename T>
    T get(const char *name, size_t len) const {
      if (!checkAndPushRef()) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushlstring(L, name, len);
      lua_gettable(L, -2);
      if (lua_isnil(L,-1)) ERROR_EXIT1(128, "Unable to find field %s\n", name);
      if (!checkType<T>(L, -1)) ERROR_EXIT(128, "Incorrect type\n");
      T v = convertTo<T>(L, -1);
      lua_pop(L,2);
      return v;
    }

    /// Returns the value stored at the given key name field. In case the field
    /// is empty, it returns the given def_value argument.    
    template<typename T>
    T opt(const char *name, const T def_value) const {
      if (!checkAndPushRef()) {
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

    /// Returns the value stored at the given key number field. In case the field
    /// is empty, it returns the given def_value argument.    
    template<typename T>
    T opt(int n, const T def_value) const {
      if (!checkAndPushRef()) {
        lua_pop(L, 1);
        return def_value;
      }
      else {
        lua_pushnumber(L, n);
        lua_gettable(L, -2);
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

    /// Returns the value stored at the given key name field. In case the field
    /// is empty, it returns the given def_value argument.    
    template<typename T>
    T opt(const char *name, size_t len, const T def_value) const {
      if (!checkAndPushRef()) {
        lua_pop(L, 1);
        return def_value;
      }
      else {
        lua_pushlstring(L, name, len);
        lua_gettable(L, -2);
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
    bool checkAndPushRef() const;
    
  };
  
  // Basic data types specializations.
  template<> char LuaTable::convertTo<char>(lua_State *L, int idx);
  template<> uint32_t LuaTable::convertTo<uint32_t>(lua_State *L, int idx);
  template<> int32_t LuaTable::convertTo<int32_t>(lua_State *L, int idx);
  template<> float LuaTable::convertTo<float>(lua_State *L, int idx);
  template<> double LuaTable::convertTo<double>(lua_State *L, int idx);
  template<> bool LuaTable::convertTo<bool>(lua_State *L, int idx);
  template<> const char *LuaTable::convertTo<const char *>(lua_State *L, int idx);
  template<> string LuaTable::convertTo<string>(lua_State *L, int idx);
  template<> LuaTable LuaTable::convertTo<LuaTable>(lua_State *L, int idx);

  template<> void LuaTable::pushInto<char>(lua_State *L, char value);
  template<> void LuaTable::pushInto<uint32_t>(lua_State *L, uint32_t value);
  template<> void LuaTable::pushInto<int32_t>(lua_State *L, int32_t value);
  template<> void LuaTable::pushInto<float>(lua_State *L, float value);
  template<> void LuaTable::pushInto<double>(lua_State *L, double value);
  template<> void LuaTable::pushInto<bool>(lua_State *L, bool value);
  template<> void LuaTable::pushInto<string>(lua_State *L, string value);
  template<> void LuaTable::pushInto<const string &>(lua_State *L,
                                                     const string &value);
  template<> void LuaTable::pushInto<const char *>(lua_State *L,
                                                   const char *value);
  template<> void LuaTable::pushInto<LuaTable>(lua_State *L, LuaTable value);

  template<> bool LuaTable::checkType<char>(lua_State *L, int idx);
  template<> bool LuaTable::checkType<uint32_t>(lua_State *L, int idx);
  template<> bool LuaTable::checkType<int32_t>(lua_State *L, int idx);
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
