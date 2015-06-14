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

#include <cmath>

extern "C" {
#include <stdint.h>
}

#include "cmath_overloads.h"
#include "error_print.h"
#include "maxmin.h"

namespace Stats {
  
  class Combinations {
    // it is forbidden to instantiate this class, it is a static class
    Combinations() {}
  public:
    
    /// Returns the logarithm of binomial coefficient of n items in groups of k
    static long double lget(uint32_t n, uint32_t k) {
      if (k>n) ERROR_EXIT(256, "k must be less or equal than n\n");
      // base case
      if (k == 0 || k == n) return 0.0;
      // general case for large numbers, use lgammal function
      long double a = lgammal(n + 1u);
      long double b = lgammal(k + 1u);
      long double c = lgammal(n - k + 1u);
      long double result = a - b - c;
      return result;
    }

    /// Returns the binomial coefficient of n items in groups of k
    static uint32_t get(uint32_t n, uint32_t k) {
      // general case for large numbers, use lgammal function
      if (n > 20u) {
        long double result = roundl(expl(lget(n, k)));
        if (result > static_cast<long double>(AprilMath::Limits<uint32_t>::max())) {
          ERROR_EXIT(256, "Overflow in 32 bits unsigned integer, use logarithmic method\n");
        }
        return static_cast<uint32_t>(result);
      }
      else {
        if (k>n) ERROR_EXIT(256, "k must be less or equal than n\n");
        // base case
        if (k == 0 || k == n) return 1u;
        // general case for short numbers, use iteration
        else {
          // take into account triangle symmetry
          k = AprilUtils::min(k, n - k);
          uint32_t result = 1u;
          for (uint32_t i=0u; i<k; ++i) {
            result = result * (n - i) / (i + 1);
          }
          return result;
        }
      }
    }
  };
}

#endif // COMBINATIONS_H
