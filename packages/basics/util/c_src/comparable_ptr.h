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
#ifndef COMPARABLE_PTR_H
#define COMPARABLE_PTR_H

namespace AprilUtils {

  template<typename T>
  class ComparablePtr {
    T *ptr;
  public:
  ComparablePtr(T *p=0) : ptr(p) {}
    ~ComparablePtr() {}
    bool operator<(const ComparablePtr &b) const {
      return *(ptr) < *(b.ptr);
    }
    bool operator>(const ComparablePtr &b) const {
      return *(ptr) > *(b.ptr);
    }
    bool operator==(const ComparablePtr &b) const {
      return *(ptr) == *(b.ptr);
    }
    bool operator!=(const ComparablePtr &b) const {
      return *(ptr) != *(b.ptr);
    }

    T& operator *() const { // dereference
      return *ptr;
    }
    
    T* operator ->() const {
      return ptr;
    }
    
  };

}

#endif // COMPARABLE_PTR_H

