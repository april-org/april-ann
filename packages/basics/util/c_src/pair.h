/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#ifndef PAIR_H
#define PAIR_H

#include "null_types.h"

namespace april_utils {

  template <typename T1, typename T2>
  struct pair {
    T1 first;
    T2 second;

    typedef T1 first_type;
    typedef T2 second_type;

    // Constructors
    pair() : first(T1()), second(T2()) {}
    pair(const T1& first, const T2& second):
      first(first), second(second) {}
    pair(const pair &other):
      first(other.first), second(other.second) {}

    pair& operator=(const pair& other) {
      if (&other != this) {
	first = other.first;
	second = other.second;
      }
      return (*this);
    }
  };


  template<typename T1, typename T2>
  bool operator== (const pair<T1,T2>& x, const pair<T1,T2>& y) {
    return ((x.first == y.first) && (x.second == y.second));
  }

  template<typename T1, typename T2>
  bool operator< (const pair<T1,T2>& x, const pair<T1,T2>& y) {
    return (( x.first <  y.first) || 
	    ((x.first == y.first) && (x.second < y.second)));
  }

  //----------------------------------------------------------------------
  // especializacion para NullType
  //----------------------------------------------------------------------

  template <typename T1>
  struct pair<T1, NullType> {
    T1 first;
    static NullType second;

    typedef T1 first_type;
    typedef NullType second_type;

    // Constructors
    pair() {}
    pair(const T1& first, const NullType&):
      first(first) {}
    pair(const pair &other):
      first(other.first) {}
    
    pair& operator=(const pair& other) {
      if (&other != this) {
	first = other.first;
      }
      return (*this);
    }
  };
  
  template<typename T1>
    bool operator== (const pair<T1,NullType>& x, const pair<T1,NullType>& y) {
    return (x.first == y.first);
  }

  template<typename T1>
    bool operator< (const pair<T1,NullType>& x, const pair<T1,NullType>& y) {
    return (x.first < y.first);
  }

  template<typename T1> NullType pair<T1, NullType>::second;

  template<typename first, typename second> inline
    pair<first, second> make_pair(const first& _X, const second& _Y)
    {
      return pair<first, second>(_X,_Y);
    }
}

#endif // PAIR_H
