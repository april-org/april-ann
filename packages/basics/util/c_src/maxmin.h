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
#ifndef MAX_MIN_H
#define MAX_MIN_H

#include <cassert>

namespace april_utils {
  template<typename T>
  T max(T a, T b) {
    return (a<b) ? b : a;
  }
  
  template<typename T>
  T min(T a, T b) {
    return (a<b) ? a : b;
  }

  // argmax, argmin, max, min for arrays
  template<typename T>
  int argmax(const T *v, int size) {
    assert(size > 0);
    int p = 0;
    for (int i=1; i<size; ++i)
      if (v[p]<v[i])
	p = i;
    return p;
  }
  
  template<typename T>
  int argmin(const T *v, int size) {
    assert(size > 0);
    int p = 0;
    for (int i=1; i<size; ++i)
      if (v[i]<v[p])
	p = i;
    return p;
  }

  template<typename T>
  T max(const T *v, int size) {
      return v[argmax(v, size)];
  }

  template<typename T>
  T min(const T *v, int size) {
      return v[argmin(v, size)];
  }

  // argmax, argmin, max, min for containers
  template<typename Iterator>
  Iterator argmax(Iterator begin, Iterator end) {
      Iterator max_iter = begin;
      for (Iterator i=begin; i!=end; ++i) {
          if (*max_iter < *i) {
              max_iter = i;
          }
      }
      return max_iter;
  }
  
  template<typename Iterator>
  Iterator argmin(Iterator begin, Iterator end) {
      Iterator min_iter = begin;
      for (Iterator i=begin; i!=end; ++i) {
          if (*i < *min_iter) {
              min_iter = i;
          }
      }
      return min_iter;
  }

  template<typename T>
  typename T::iterator argmax(T& container) {
      return argmax(container.begin(), container.end());
  }
  
  template<typename T>
  typename T::iterator argmin(T& container) {
      return argmin(container.begin(), container.end());
  }

  template<typename T>
  typename T::value_type max(T& container) {
      return *april_utils::argmax(container.begin(), container.end());
  }
  
  template<typename T>
  typename T::value_type min(T& container) {
      return *april_utils::argmin(container.begin(), container.end());
  }

}

#endif //MAX_MIN_H
