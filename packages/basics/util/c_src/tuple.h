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
#ifndef TUPLE_H
#define TUPLE_H

#include "april_assert.h"
#include <cstddef>

namespace AprilUtils {

  template<typename T> class tuple {

    T *vec;
    size_t tuple_size;

    public:
    typedef T         value_type;
    typedef T*        pointer;
    typedef T&        reference;
    typedef const T&  const_reference;
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    size_type  size() const { return tuple_size; }

    tuple() {
      tuple_size = 0;
      vec        = 0;
    }
    
    tuple(int tuple_size) {
      april_assert(tuple_size >= 0);
      this->tuple_size = tuple_size;
      vec = new T[tuple_size];
    }

    tuple(const tuple &l) {
      tuple_size = l.tuple_size;
      vec = new T[tuple_size];
      for (unsigned int i=0; i<tuple_size; i++)
	vec[i] = l.vec[i];
    }

    tuple &operator=(const tuple &l) {
      if (l != *this) {
        if (tuple_size != l.tuple_size) {
          tuple_size = l.tuple_size;
          delete[] vec;
          vec = new T[tuple_size];
        }
        for (unsigned int i=0; i<tuple_size; i++)
	  vec[i] = l.vec[i];
      }
      return *this;
    }

    ~tuple() {
      delete[] vec;
    }

    reference operator[](size_type n) {
      return vec[n];
    }

    const_reference operator[](size_type n) const {
      return vec[n];
    }

    bool operator== (const tuple &l) const {
      if (tuple_size != l.tuple_size) return false;
      for (unsigned int i=0; i<tuple_size; i++) {
	if (!(vec[i] == l.vec[i])) return false;
      }
      return true;
    }

    bool operator!= (const tuple &l) const {
      return !(*this == l);
    }

  };  

} // namespace AprilUtils

#endif

