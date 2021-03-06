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
#include "is_same.h"
#include "mystring.h"
#include "referenced.h"
#include "remove_pointer.h"
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
    template<> void LuaTable::pushInto<type>(lua_State *L, SharedPtr<type> value); \
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
    template<> void LuaTable::pushInto<type>(lua_State *L, SharedPtr<type> value) { \
      lua_push##type(L, value.get());                                   \
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
   * table[rng] = bar3;
   * bar1 = table["foo"].get<BarType>();
   * bar1 = table["foo"].opt<BarType>(DefaultFooValue);
   * bar2 = table[1].opt<BarType>(DefaultFooValue);
   * bar3 = table[rng].opt<BarType>(DefaultFooValue);
   * @endcode
   *
   * @note Keys can be strings, integers or referenced pointers.
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
     * @brief Implements a helper for left hand side operator[] of LuaTable
     * class.
     *
     * This helper allow to use LuaTable in the following way:
     *
     * @code
     * table[foo] = bar;
     * bar = table[foo].get<BarType>();
     * bar = table[foo].opt<BarType>(DefaultFooValue);
     * table["foo"] = bar;
     * bar = table["foo"].get<BarType>();
     * bar = table["foo"].opt<BarType>(DefaultFooValue);
     * @endcode
     */
    template<typename K>
    class LHSAccessorByPointer {
    public:
      LHSAccessorByPointer(LuaTable *table, K *key) :
        table(table),key(key) {}
      LHSAccessorByPointer(const LHSAccessorByPointer<K> &other) :
        table(other.table),key(other.key) {}
      template<typename T>
      LuaTable &operator=(T value) {
        return table->put<T>(key, value);
      }
      template<typename T>
      T get() const {
        return table->get<T>(key);
      }
      template<typename T>
      T opt(const T def_value) const {
        return table->opt<T>(key, def_value);
      }
      template<typename T>
      T opt(AprilUtils::SharedPtr<typename remove_pointer<T>::type> def_value) const {
        return table->opt<T>(key, def_value.get());
      }
      template<typename T>
      bool is() const {
        return table->checkNilOrType<T>(key);
      }
    private:
      LuaTable *table;
      K * key;
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

    /**
     * @brief Implements a helper for right hand side operator[] of LuaTable
     * class.
     *
     * This helper allow to use LuaTable in the following way:
     *
     * @code
     * bar = table[foo].get<BarType>();
     * bar = table[foo].opt<BarType>(DefaultFooValue);
     * bar = table["foo"].get<BarType>();
     * bar = table["foo"].opt<BarType>(DefaultFooValue);
     * @endcode
     */
    template<typename K>
    class RHSAccessorByPointer {
    public:
      RHSAccessorByPointer(const LuaTable *table, K *key) :
        table(table),key(key) {}
      RHSAccessorByPointer(const RHSAccessorByPointer<K> &other) :
        table(other.table),key(other.key) {}
      template<typename T>
      T get() const {
        return table->get<T>(key);
      }
      template<typename T>
      T opt(const T def_value) const {
        return table->opt<T>(key, def_value);
      }
      template<typename T>
      T opt(AprilUtils::SharedPtr<typename remove_pointer<T>::type> def_value) const {
        return table->opt<T>(key, def_value.get());
      }
      template<typename T>
      bool is() const {
        return table->checkNilOrType<T>(key);
      }
    private:
      const LuaTable *table;
      K * key;
    };


    ////////////////////////////////////////////////////////////////////////
  public:
    
    /// Constructor for a new LuaTable in the registry.
    LuaTable(lua_State *L = Base::getAndCheckGlobalLuaState());

    /// Constructor for a new LuaTable in the registry containing C array data.
    template<typename T>
    LuaTable(const T *vec, size_t len,
             lua_State *L = Base::getAndCheckGlobalLuaState()) {
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
    
    /**
     * @brief Pushes into stack the table and takes it as an absolute index.
     *
     * Be careful, the stack will be modified because a reference to the table
     * will be pushed on top of it. This method allows to perform a large number
     * of table read/write operations in an efficient way. See the example:
     *
     * @code
     * LuaTable t(L,-1);
     * t.lock();
     * double sum;
     * for (int i=1; i<=1000000; ++i) sum += t[i].get<double>();
     * t.unlock();
     * @endcode
     *
     * @note If LuaTable has been locked before, this method does nothing and
     * returns false. Otherwise it returns true.
     *
     * @note It is necessary to call unlock() method every time you do a lock().
     */
    bool lock() {
      if (absindex == 0) {
        int abspos;
        int pos = checkAndGetRef(abspos);
        if (pos == 0) ERROR_EXIT(128, "Found empty table\n");
        assert(pos == -1);
        absindex = abspos;
        return true;
      }
      return false;
    }
    
    /**
     * @brief Removes from stack the absolute index taken by lock() method. Be
     * careful, the stack will be modified.
     *
     * @see lock() method.
     */
    bool unlock() {
      if (absindex > 0) {
        lua_remove(L, absindex);
        absindex = 0;
        return true;
      }
      return false;
    }
    
    /// Indicates if the LuaTable is empty.
    bool empty() const;
    
    /// Executes length operator in Lua
    size_t length() const;
    
    /// Copy operator.
    LuaTable &operator=(const LuaTable &other);

    /// See RHSAccessorByPointer
    RHSAccessorByPointer<const char> operator[](const string &name) const {
      return RHSAccessorByPointer<const char>(this, name.c_str());
    }

    /// See RHSAccessorByInteger
    RHSAccessorByInteger operator[](int n) const {
      return RHSAccessorByInteger(this, n);
    }

    /// See RHSAccessorByPointer
    template<typename K>
    RHSAccessorByPointer<K> operator[](K *key) const {
      return RHSAccessorByPointer<K>(this, key);
    }

    /// See LHSAccessorByPointer
    LHSAccessorByPointer<const char> operator[](const string &name) {
      return LHSAccessorByPointer<const char>(this, name.c_str());
    }
    
    /// See LHSAccessorByInteger
    LHSAccessorByInteger operator[](int n) {
      return LHSAccessorByInteger(this, n);
    }

    /// See LHSAccessorByPointer
    template<typename K>
    LHSAccessorByPointer<K> operator[](K *key) {
      return LHSAccessorByPointer<K>(this, key);
    }
    
    /// Returns a C++ string with the Lua representation of the table.
    string toLuaString(bool binary=true);
    
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

    /// Puts a new value into the table, using the given key number.
    template<typename T>
    LuaTable &put(int n, T value) {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushnumber(L, n);
      pushInto(L, value);
      lua_settable(L, abspos);
      popRef(pos);
      return *this;
    }

    /// Puts a new value into the table, using the given key pointer or string.
    template<typename T, typename K>
    LuaTable &put(K *key, T value) {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      pushInto(L, key);
      pushInto(L, value);
      lua_settable(L, abspos);
      popRef(pos);
      return *this;
    }

    /// Puts a new value into the table, using the given key name.
    template<typename T>
    LuaTable &put(const char *name, size_t len, T value) {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushlstring(L, name, len);
      pushInto(L, value);
      lua_settable(L, abspos);
      popRef(pos);
      return *this;
    }
    
    /// Checks if the field at the given key number is nil.
    bool checkNil(int n) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushnumber(L, n);
      lua_gettable(L, abspos);
      bool ret =  lua_isnil(L, -1);
      popRef(pos, 1);
      return ret;
    }

    /// Checks if the field at the given key pointer or string is nil.
    template<typename K>
    bool checkNil(K *key) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      pushInto(L, key);
      lua_gettable(L, abspos);
      bool ret = lua_isnil(L, -1);
      popRef(pos, 1);
      return ret;
    }

    /// Checks if the field at the given key name is nil.    
    bool checkNil(const char *name, size_t len) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushlstring(L, name, len);
      lua_gettable(L, abspos);
      bool ret =  lua_isnil(L, -1);
      popRef(pos, 1);
      return ret;
    }

    /// Checks if the field at the given key number is of the given type (a nil
    /// value will be taken as true).
    template<typename T>
    bool checkNilOrType(int n) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushnumber(L, n);
      lua_gettable(L, abspos);
      bool ret = lua_isnil(L, -1) || checkType<T>(L, -1);
      popRef(pos, 1);
      return ret;
    }

    /// Checks if the field at the given key pointer or string is of the given
    /// type (a nil value will be taken as true).
    template<typename T, typename K>
    bool checkNilOrType(K *key) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      pushInto(L, key);
      lua_gettable(L, abspos);
      bool ret = lua_isnil(L, -1) || checkType<T>(L, -1);
      popRef(pos, 1);
      return ret;
    }
    
    /// Checks if the field at the given key name is of the given type (a nil
    /// value will be taken as true).
    template<typename T>
    bool checkNilOrType(const char *name, size_t len) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushlstring(L, name, len);
      lua_gettable(L, abspos);
      bool ret = lua_isnil(L, -1) || checkType<T>(L, -1);
      popRef(pos, 1);
      return ret;
    }

    /// Returns the value stored at the given key number field.
    template<typename T>
    T get(int n) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushnumber(L, n);
      lua_gettable(L, abspos);
      if (lua_isnil(L,-1)) ERROR_EXIT1(128, "Unable to find field %d\n", n);
      if (!checkType<T>(L, -1)) ERROR_EXIT(128, "Incorrect type\n");
      T v = convertTo<T>(L, -1);
      popRef(pos, 1);
      return v;
    }

    /// Returns the value stored at the given key pointer or string field.
    template<typename T, typename K>
    T get(K *key) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      pushInto(L, key);
      lua_gettable(L, abspos);
      if (lua_isnil(L,-1)) {
        if (is_same<K*, const char*>::value ||
            is_same<K*, char*>::value) { // string case, resolved at compilation
          ERROR_EXIT1(128, "Unable to find field %s\n", key);
        }
        else { // referenced pointer case
          ERROR_EXIT1(128, "Unable to find field %p\n", key);
        }
      }
      if (!checkType<T>(L, -1)) ERROR_EXIT(128, "Incorrect type\n");
      T v = convertTo<T>(L, -1);
      popRef(pos, 1);
      return v;
    }
    
    /// Returns the value stored at the given key name field.
    template<typename T>
    T get(const char *name, size_t len) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) ERROR_EXIT(128, "Invalid reference\n");
      lua_pushlstring(L, name, len);
      lua_gettable(L, abspos);
      if (lua_isnil(L,-1)) ERROR_EXIT1(128, "Unable to find field %s\n", name);
      if (!checkType<T>(L, -1)) ERROR_EXIT(128, "Incorrect type\n");
      T v = convertTo<T>(L, -1);
      popRef(pos, 1);
      return v;
    }

    /// Returns the value stored at the given key number field. In case the field
    /// is empty, it returns the given def_value argument.    
    template<typename T>
    T opt(int n, const T def_value) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) {
        popRef(pos);
        return def_value;
      }
      else {
        lua_pushnumber(L, n);
        lua_gettable(L, abspos);
        T v(def_value);
        if (!lua_isnil(L,-1)) {
          if (!checkType<T>(L, -1)) ERROR_EXIT(128, "Incorrect type\n");
          v = convertTo<T>(L, -1);
        }
        popRef(pos, 1);
        return v;
      }
      // return T();
    }

    /// Returns the value stored at the given key pointer or string field. In
    /// case the field is empty, it returns the given def_value argument.
    template<typename T, typename K>
    T opt(K *key, const T def_value) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) {
        popRef(pos);
        return def_value;
      }
      else {
        pushInto(L, key);
        lua_gettable(L, abspos);
        T v(def_value);
        if (!lua_isnil(L,-1)) {
          if (!checkType<T>(L, -1)) ERROR_EXIT(128, "Incorrect type\n");
          v = convertTo<T>(L, -1);
        }
        popRef(pos, 1);
        return v;
      }
      // return T();
    }

    /// Returns the value stored at the given key name field. In case the field
    /// is empty, it returns the given def_value argument.    
    template<typename T>
    T opt(const char *name, size_t len, const T def_value) const {
      int pos, abspos;
      if ((pos=checkAndGetRef(abspos)) == 0) {
        popRef(pos);
        return def_value;
      }
      else {
        lua_pushlstring(L, name, len);
        lua_gettable(L, abspos);
        T v(def_value);
        if (!lua_isnil(L,-1)) {
          if (!checkType<T>(L, -1)) ERROR_EXIT(128, "Incorrect type\n");
          v = convertTo<T>(L, -1);
        }
        popRef(pos, 1);
        return v;
      }
      // return T();
    }

    /// Returns a userdata interpreted as the given type T avoiding type check
    template<typename T>
    static T lua_rawgetudata(lua_State *L, int n) {
      T *pre_obj = static_cast<T*>(lua_touserdata(L,n));
      T obj = 0;
      if (pre_obj) obj = (*pre_obj);
      return obj;
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

    /// Pushes a SharedPtr instance into the Lua stack.
    template<typename T>
    static void pushInto(lua_State *L, SharedPtr<T> value) {
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
    /// Absolute index, only valid when LuaTable is in locked state (absindex>0).
    int absindex;

    /// Auxiliary method to simplify constructors.
    void init(lua_State *L, int i);
    
    /// Checks the value of ref and absindex and returns a valid position to the stack.
    int checkAndGetRef(int &abspos) const;
    
    /// Receives the position returned by checkAndGetRef() and performs operations needed to end using it.
    void popRef(int pos, int extra=0) const;
    
    
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
