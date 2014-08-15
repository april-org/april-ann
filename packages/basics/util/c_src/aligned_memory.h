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
#ifndef ALIGNED_MEMORY_H
#define ALIGNED_MEMORY_H

#ifndef NO_MM_MALLOC
extern "C" {
#include <mm_malloc.h>
}

#define VECTOR_ALIGNMENT 16

namespace april_utils {
  template<typename T>
  inline
  T* aligned_malloc(size_t nmemb) {
    return (T*)_mm_malloc(sizeof(T)*nmemb,VECTOR_ALIGNMENT);
  }
  
  template<typename T>
  inline
  void aligned_free(T *ptr) {
    _mm_free(ptr);
  }
} // namespace april_utils

#else

#include <cstdlib>

namespace april_utils {
  template<typename T>
  inline
  T* aligned_malloc(size_t nmemb) {
    return (T*)malloc(sizeof(T)*nmemb);
  }
  
  template<typename T>
  inline
  void aligned_free(T *ptr) {
    free(ptr);
  }
} // namespace april_utils

#endif

#endif // ALIGNED_MEMORY_H
