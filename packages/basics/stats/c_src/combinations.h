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
#ifndef COMBINATIONS_H
#define COMBINATIONS_H

#include "error_print.h"
#include "vector.h"

namespace Stats {

  class Combinations {
    // it is forbidden to instantiate this class, it is a static class
    Combinations() {}
  
    static april_utils::vector<unsigned int> pascal_triangle;
    
    static void reserve(unsigned int n) {
      if (pascal_triangle.size() <= n) {
        unsigned int oldn = pascal_triangle.size();
        pascal_triangle.resize( (n+1)<<1 );
        for (unsigned int i=oldn; i<pascal_triangle.size(); ++i)
          pascal_triangle[i] = 0;
      }
    }
    
    static unsigned int privateGet(unsigned int n, unsigned int k) {
      // frontier problems
      if (n <= 1 || k == 0) return 1;
      unsigned int pos = ( ( (n)*(n-1) ) >> 1 ) + (k-1) - 1;
      reserve(pos);
      unsigned int &v = pascal_triangle[pos];
      if (v == 0) {
        // general problem
        if (k<n) v = privateGet(n-1, k-1) + privateGet(n-1, k);
        else     v = privateGet(n-1, k-1);
      }
      return v;
    }
    
    public:
  
    static unsigned int get(unsigned int n, unsigned int k) {
      if (k>n) ERROR_EXIT(256, "k must be less or equal than n\n");
      return privateGet(n, k);
    }

  };
  
}

#endif // COMBINATIONS_H
