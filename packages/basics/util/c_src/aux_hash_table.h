/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#ifndef AUX_HASH_TABLE_H
#define AUX_HASH_TABLE_H

#include <stdint.h>
#include <cstring> // strcmp
#include <cassert>
#include "error_print.h"
#include "constString.h"
#include "pair.h"

namespace april_utils {

  //----------------------------------------------------------------------
  template<typename T> struct default_hash_function
  {
    static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    long int operator()(const T& key) const {
      unsigned int resul = 1;
      unsigned int length = sizeof(T);
      const unsigned char *r = reinterpret_cast<const unsigned char *>(&key);
      for (unsigned int i=0; i<length; i++) {
        resul = (resul+(unsigned int)(r[i]))*cte_hash;
      }
      return resul;
    }
  };

  template<typename T> struct default_equality_comparison_function
  {
    bool operator()(const T& i1, const T& i2) const {
      return i1 == i2;
    }
  };

  //----------------------------------------------------------------------
  // Hash function for ints (operator == is OK)
  //
  template<> struct default_hash_function<int> {
    static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    long int operator()(const int &i) const {
      return cte_hash*i;
    }
  };

  //----------------------------------------------------------------------
  // Hash function for unsigned ints (operator == is OK)
  //
  template<> struct default_hash_function<unsigned int> {
    static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    long int operator()(const unsigned int &i) const {
      return cte_hash*i;
    }
  };

  //----------------------------------------------------------------------
  // Hash function for pair<int,int> (operator == is OK)
  //
  typedef april_utils::pair<int,int> int_pair;

  template<> struct default_hash_function<int_pair> {
    static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    long int operator()(const int_pair &i) const {
      return ((cte_hash*i.first) + i.second) * cte_hash;
    }
  };


  //----------------------------------------------------------------------
  // Hash function for pair<unsigned int,unsigned int> (operator == is OK)
  //
  typedef april_utils::pair<unsigned int,unsigned int> uint_pair;

  template<> struct default_hash_function<uint_pair> {
    static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    long int operator()(const uint_pair &i) const {
      return ((cte_hash*i.first) + i.second) * cte_hash;
    }
  };

  //----------------------------------------------------------------------
  // Hash function for pair<int16_t,int16_t> (operator == is OK)
  //
  typedef april_utils::pair<int16_t,int16_t> int16_pair;

  template<> struct default_hash_function<int16_pair> {
    static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    long int operator()(const int16_pair &i) const {
      return ((cte_hash*i.first) + i.second) * cte_hash;
    }
  };

  //----------------------------------------------------------------------
  // Hash function for constString (operator == is OK)
  //
  template <> struct default_hash_function<constString> {
    static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    long int operator()(const constString &cs) const {
      unsigned int resul = 1;    
      const char *r = (const char*)cs; // we are using operator const char *()
      for (int l=cs.len(); l>0; l--,r++)
        resul = (resul+(unsigned int)(*r))*cte_hash;
      return resul;
    }
  };

  //----------------------------------------------------------------------
  // Hash function for generic pointers (operator == is OK)
  //
  template <typename T> struct default_hash_function<T *> {
    static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    long int operator()(T* p) const {
      // arquitectura 32 bits
      if (sizeof(void*) == 4) {
        union {
          uint32_t   i;
          void   *p;
        } avoid_strict_aliasing;
        avoid_strict_aliasing.p = p;
        return avoid_strict_aliasing.i * cte_hash;
      }
      // arquitectura 64 bits
      else if (sizeof(void*) == 8) {
        union {
          uint32_t   i[2]; /* usamos 2 punteros de 32 para poder hacer
                              hashing del puntero de 64 */
          void   *p;
        } avoid_strict_aliasing;
        avoid_strict_aliasing.p = p;
        return ((avoid_strict_aliasing.i[0] * cte_hash +
              avoid_strict_aliasing.i[1]) * cte_hash);
      }
      else {
        ERROR_PRINT("parsers_pool_hash_function(): sizeof(void*) tiene un tamaño desconocido.\n");
      }
      return 0;
    }
  };

  //----------------------------------------------------------------------
  // c-style strings (const char *) need also a comparison function
  //
  template <> struct default_equality_comparison_function<char *> {
    bool operator()(const char* s1, const char* s2) const {
      return strcmp(s1, s2) == 0;
    }
  };

  template <> struct default_hash_function<char *> {
    static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
    long int operator()(const char* s1) const {
      unsigned int resul = 1;
      for (const char *r = s1; *r != '\0'; r++)
        resul = (resul+(unsigned int)(*r))*cte_hash;
      return resul;
    }
  };

  //----------------------------------------------------------------------
  // template para hash de "cosas" que tengan el metodo size,
  // elementos que son tipos básicos y acceso a ellos via operator[]
  template<typename T>
    struct hash_restricted_vector {
      static const unsigned int cte_hash  = 2654435769U; // hash Fibonacci
      long int operator()(const T& key) const {
        unsigned int resul   = 1;
        unsigned int objsize = key.size();
        for (unsigned int i=0; i<objsize; i++)
          resul = (resul+(unsigned int)(key[i]))*cte_hash;
        return resul;
      }
    };

} // closes namespace

#endif

