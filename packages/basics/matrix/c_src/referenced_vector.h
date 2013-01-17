/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Francisco Zamora-Martinez
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
#ifndef REFERENCED_VECTOR_H
#define REFERENCED_VECTOR_H

#include <stdint.h>
#include "referenced.h"
#include "vector.h"

namespace april_utils {

  template<typename T> 
  class ReferencedVector : public Referenced, public vector<T> {
  public:

    ReferencedVector(size_t n=0) :
      Referenced(), vector<T>(n) { }

    // Destructor...
    virtual ~ReferencedVector() {
    }

    T* release_internal_vector() {
      // :'( he perdido una hora para averiguar que peta si no se pone el this->
      // http://www.cplusplus.com/forum/general/6798
      T *returned = this->vec;
      this->used_size = this->vector_size = 0;
      this->vec = 0;
      return returned;
    }

  };

  typedef ReferencedVector<float> ReferencedVectorFloat;
  typedef ReferencedVector<uint32_t> ReferencedVectorUint;

} // namespace

#endif
