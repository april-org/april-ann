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
#ifndef KNN_POINT_H
#define KNN_POINT_H

#include "matrix.h"

namespace KNN {
  
  /// Class which defines a sample (a point in the space) for the KNN algorithm
  /// and KDTree class. This class depends on a memory pointer which depends on
  /// a Matrix. To retain the Matrix, the KDTree class has the pointer to the
  /// Matrix object.
  template<typename T>
  class Point {
    const T *base;
    int stride, id;
  public:
    /// Constant iterator over the point components
    class const_iterator {
      friend class Point;
      const T *ptr;
      int stride;
      const_iterator(const T *ptr, int stride) : ptr(ptr), stride(stride) { }
    public:
      const_iterator() : ptr(0) { }
      const_iterator(const const_iterator &other) :
	ptr(other.ptr), stride(other.stride) {
      }
      ~const_iterator() { }
      const_iterator &operator=(const const_iterator &other) {
	ptr    = other->ptr;
	stride = other->stride;
	return *this;
      }
      const_iterator &operator++() { ptr+=stride; return *this; }
      const_iterator &operator--() { ptr-=stride; return *this; }
      const_iterator &operator+=(int v) { ptr+=stride*v; return *this; }
      const_iterator &operator-=(int v) { ptr-=stride*v; return *this; }
      const T &operator*() const { return *ptr; }
      const T *operator->() const { return ptr; }
      bool operator==(const const_iterator &other) const {
	return other.ptr == ptr;
      }
      bool operator!=(const const_iterator &other) const {
	return other.ptr != ptr;
      }
      bool operator<(const const_iterator &other) const {
	return other.ptr < ptr;
      }
      bool operator<=(const const_iterator &other) const {
	return other.ptr <= ptr;
      }
      bool operator>(const const_iterator &other) const {
	return other.ptr > ptr;
      }
      bool operator>=(const const_iterator &other) const {
	return other.ptr >= ptr;
      }
    };
    //
    Point() : base(0) {
    }
    Point(const Point<T> &other) :
      base(other.base), stride(other.stride), id(other.id) { }
    Point(const T *base, int stride, int id) :
      base(base), stride(stride), id(id) { }
    Point(const Basics::Matrix<T> *m, int row, int id) :id(id) {
      april_assert(m->getNumDim() == 2);
      int coords[2] = { row, 0 };
      base = m->getRawDataAccess()->getPPALForRead()+m->computeRawPos(coords);
      stride = m->getStrideSize(1);
    }
    ~Point() { }
    Point &operator=(const Point<T> &other) {
      base   = other.base;
      stride = other.stride;
      id     = other.id;
      return *this;
    }
    bool operator==(const Point<T> &other) const {
      return other.base == base;
    }
    bool operator!=(const Point<T> &other) const {
      return other.base != base;
    }
    const T &operator[](int pos) const { return base[stride*pos]; }
    const_iterator begin() const { return const_iterator(base, stride); }
    const_iterator end(const int D) const { return const_iterator(base+D*stride, stride); } 
    int getId() const { return id; }
    // WARNING: this function must to be redefined if the type T is not a plain
    // type (like a Complex number).
    double dist(const Point &other, const int D) const {
      const T *this_ptr  = base;
      const T *other_ptr = other.base;
      double d = 0.0;
      for (int i=0; i<D; ++i, this_ptr+=stride, other_ptr+=other.stride) {
	double diff = (*this_ptr) - (*other_ptr);
	d += diff * diff;
      }
      return d;
    }
  };
}
#endif // KNN_POINT_H
