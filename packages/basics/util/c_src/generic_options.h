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
  
  class GenericOptions {
  public:
    
    GenericOptions() { }
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
    
    enum ValueTypes {
      DOUBLE, FLOAT, CHAR, STRING, INT32, UINT32, INT64, UINT64, BOOL, NUM_TYPES
    };
    
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
    
    HashTable dict;
      
  };

  class LuaTableOptions : public GenericOptions {
  public:
    
    LuaTableOptions(lua_State *L);
    LuaTableOptions(lua_State *L, int i);
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
    
    lua_State *L;
    int ref;
    
    void init(lua_State *L, int i);
  };

#undef METHODS

} // namespace april_utils

#endif // GENERIC_OPTIONS_H
