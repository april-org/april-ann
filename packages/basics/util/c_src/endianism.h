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
#ifndef ENDIANISM_H
#define ENDIANISM_H

#include "swap.h"

namespace april_utils {

  template<typename T>  
  void swap_bytes_in_place(T &data) {
    size_t sz   = sizeof(T);
    char *ptr   = reinterpret_cast<char*>(&data);
    size_t last = sz-1;
    const size_t half = sz/2;
    for (size_t i=0; i<half; ++i, --last)
      swap(ptr[i], ptr[last]);
  }

  template<typename T>  
  T swap_bytes(T data) {
    T copy_data = data;
    size_t sz   = sizeof(T);
    char *ptr   = reinterpret_cast<char*>(&copy_data);
    size_t last = sz-1;
    const size_t half = sz/2;
    for (size_t i=0; i<half; ++i, --last)
      swap(ptr[i], ptr[last]);
    return copy_data;
  }

}

#endif // ENDIANISM_H
