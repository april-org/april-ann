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
#ifndef GENERIC_OPTIONS_H
#define GENERIC_OPTIONS_H

#include "hash_table.h"
#include "mystring.h"

namespace april_utils {
  
#define METHODS(method, type)                                           \
  virtual GenericOptions *put##method(const char *name, type value) = 0; \
  virtual type get##method(const char *name) const = 0;                 \
  virtual type getOptional##method(const char *name, type const value) const = 0

  /**
   * @brief Class which contains pairs key,value of standard types.
   *
   * The key is always a string and the value can be one of the following:
   * double, float, char, const char *, int32_t, uint32_t, int64_t, uint64_t,
   * bool.
   *
   * Three methods are available for each type:
   *
   * - The put method as putDouble() which receives a <tt>const char * key</tt>
   *   and a <tt>double value</tt> and inserts into the set the pair
   *   (key,value).
   *
   * - The get method as getDouble() which receives a <tt>const char * key</tt>
   *   and returns a @c double value. In case the given @c key doesn't exists,
   *   this method throws an error.
   *
   * - The getOptional method as getOptionalDouble() which receives a <tt>const
   *   char *key</tt>, an optional value <tt>double opt</tt>, and returns a @c
   *   double. In case the given @c key doesn't exists, this method returns the
   *   given @c opt value.
   *
   * @note The types can be binded to Lua, allowing to have the same generic
   * interface for data coming from Lua or from C/C++.
   */
  class GenericOptions {
  public:
    
    /// Constructor.
    GenericOptions() { }
    /// Destructor.
    virtual ~GenericOptions() { }
    
    METHODS(Double, double);
    METHODS(Float, float);
    METHODS(Char, char);
    METHODS(String, const char *);
    METHODS(Int32, int32_t);
    METHODS(UInt32, uint32_t);
    METHODS(Int64, int64_t);
    METHODS(UInt64, uint64_t);
    METHODS(Boolean, bool);
    
  };
#undef METHODS
  
#define METHODS(method, type)                                           \
  virtual GenericOptions *put##method(const char *name, type value);    \
  virtual type get##method(const char *name) const;                     \
  virtual type getOptional##method(const char *name, type const value) const

  /**
   * @brief Specialization of GenericOptions for passing data stored at a hash
   * table in C++.
   *
   * This class uses a april_utils::hash table where (key,value) pairs will be
   * stored. To allow multiple value types, a @c union has been declared, and an
   * enum HashTableOptions::ValueTpyes allows to indicate which type has been
   * stored.
   */  
  class HashTableOptions : public GenericOptions {
  public:
    
    HashTableOptions() : GenericOptions() { }
    virtual ~HashTableOptions() { }
    
    METHODS(Double, double);
    METHODS(Float, float);
    METHODS(Char, char);
    METHODS(String, const char *);
    METHODS(Int32, int32_t);
    METHODS(UInt32, uint32_t);
    METHODS(Int64, int64_t);
    METHODS(UInt64, uint64_t);
    METHODS(Boolean, bool);
    
  private:

    /// Enumeration with the types which can be stored in the table.
    enum ValueTypes {
      DOUBLE, FLOAT, CHAR, STRING, INT32, UINT32, INT64, UINT64, BOOL, NUM_TYPES
    };
    
    /// Value part of the table elements, it is a union of all the possible
    /// values and a ValueTypes instance indicating which one it is.
    struct Value {
      union {
        float flt;
        double dbl; 
        char chr;
        int32_t i32;
        uint32_t u32;
        int64_t i64;
        uint64_t u64;
        bool bl;
      };
      april_utils::string str; // can't be in the union
      ValueTypes type;
    };
    
    typedef april_utils::hash<april_utils::string, Value> HashTable;
    
    /// The april_utils::hash with the key,value pairs.
    HashTable dict;
      
  };

  /**
   * @brief Specialization of GenericOptions for passing data stored at a Lua
   * table.
   *
   * This class uses Lua references to copy a reference to a Lua table in the
   * registry. It can receive a Lua table at the constructor, or if not given,
   * a new empty table will be allocated.
   */
  class LuaTableOptions : public GenericOptions {
  public:
    
    /// Constructor from a new allocated Lua table.
    LuaTableOptions(lua_State *L);
    /// Constructor from a table allocated at Lua stack position @c i.
    LuaTableOptions(lua_State *L, int i);
    /// Destructor, de-references the Lua table for garbage collection.
    virtual ~LuaTableOptions();
    
    METHODS(Double, double);
    METHODS(Float, float);
    METHODS(Char, char);
    METHODS(String, const char *);
    METHODS(Int32, int32_t);
    METHODS(UInt32, uint32_t);
    METHODS(Int64, int64_t);
    METHODS(UInt64, uint64_t);
    METHODS(Boolean, bool);
    
  private:
    
    /// The lua_State where the table is allocated.
    lua_State *L;
    /// The reference in the registry where the table can be retrieved.
    int ref;
    
    /// Auxiliary method to simplify constructors.
    void init(lua_State *L, int i);
  };

#undef METHODS

} // namespace april_utils

#endif // GENERIC_OPTIONS_H
