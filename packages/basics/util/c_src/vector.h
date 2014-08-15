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
#ifndef VECTOR_H
#define VECTOR_H

#include <stddef.h>
#include "swap.h"

namespace april_utils {

  // Be careful! It's not std::vector :)
  template<typename T> class vector {
  protected:

    T *vec;
    size_t used_size,vector_size;

  public:
    typedef T         value_type;
    typedef T*        pointer;
    typedef T&        reference;
    typedef const T&  const_reference;
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;
    typedef T*        iterator;
    typedef const T*  const_iterator;


    size_type  size()     const { return used_size;  }
    size_type  capacity() const { return vector_size; }
    size_type  max_size() const { return size_type(-1); }
    bool       empty()    const { return (used_size==0); }
    
    pointer    release() {
      pointer tmp = vec;
      used_size = vector_size = 0;
      vec = 0;
      return tmp;
    }

    vector(size_type n=0) { // Creates a vector with n elements.
      used_size   = n;
      vector_size = n;
      vec         = 0;
      if (used_size > 0) {
        vec         = new T[vector_size];
      }
    }

    vector(size_type n, const_reference t) { // Creates a vector with n copies of t.
      used_size   = n;
      vector_size = n;
      vec         = 0;
      if (used_size > 0) {
        vec         = new T[vector_size];
        for (unsigned int i=0; i<used_size; i++)
	  vec[i] = t;
      }
    }

    vector(const vector &l) {
      vector_size = l.used_size;
      used_size   = l.used_size;
      if (vector_size) vec = new T[vector_size];
      else vec=0;
      for (unsigned int i=0; i<l.used_size; i++)
	vec[i] = l.vec[i];
    }

    // copy range [first,last)
    vector(const_iterator first, const_iterator last) {
      vector_size = used_size = last - first;
      if (vector_size) vec = new T[vector_size];
      else vec=0;
      for (unsigned int i=0; i<used_size; i++, first++)
	vec[i] = *first;
    }

    vector &operator=(const vector &l) {
      if (&l != this) {
        used_size = l.used_size;
        if (vector_size < l.used_size) {
          vector_size = l.used_size;
          if (vec)         delete[] vec;
          if (vector_size) vec = new T[vector_size];
          else vec=0;
        }
        for (unsigned int i=0; i<l.used_size; i++)
	  vec[i] = l.vec[i];
      }
      return *this;
    }

    ~vector() {
      if (vec) {
	delete[] vec;
      }
    }

    reference operator[](size_type n) {
      return vec[n];
    }

    const_reference operator[](size_type n) const {
      return vec[n];
    }

    void reserve(size_type n) {
      // Si n > vector_size -> reservar vector mas grande.
      // used_size no cambia.
      if (n > vector_size) {
        vector_size = n;
        T *old_vec  = vec;
        vec         = new T[vector_size];
        for (unsigned int i=0; i<used_size; i++)
          vec[i] = old_vec[i];
        if (old_vec) delete[] old_vec;
      }
    }


    void push_back(const_reference t) {
      if (used_size >= vector_size) {
        size_type old_vector_size = vector_size;
        vector_size = (vector_size == 0) ? 1 : 2*vector_size;
        T *old_vec  = vec;
        vec         = new T[vector_size];
        for (unsigned int i=0; i<old_vector_size; i++)
          vec[i] = old_vec[i];
        if (old_vec) delete[] old_vec;
      }
      vec[used_size] = t;
      used_size++;
    }

    void pop_back() {
      if (used_size > 0)
        used_size--;
    }

    reference back() { return vec[used_size-1]; }
    const_reference back() const { return vec[used_size-1]; }

    void resize(size_type n) {
      reserve(n);
      used_size = n;
    }

    void clear() { used_size = 0; }

    void swap(vector &other) {
      april_utils::swap(vec,         other.vec);
      april_utils::swap(vector_size, other.vector_size);
      april_utils::swap(used_size,   other.used_size);
    }

    iterator begin() { return vec; }
    const_iterator begin() const { return vec; }
    iterator end() { return vec+used_size; }
    const_iterator end() const { return vec+used_size; }

  };
} // namespace april_utils

#endif

