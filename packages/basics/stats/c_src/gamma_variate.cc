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
#include <cmath>
#include "april_assert.h"
#include "gamma_variate.h"

using Basics::MTRand;

namespace Stats {
  // a=location, b=scale, c=shape
  // Modification of: http://ftp.arl.mil/random/random.pdf
  double gammaVariate(MTRand *rng, const double a, const double b,
                      const double c) {
    double y;
    april_assert( b > 0.0 && c > 0.0 );
    const double A = 1. / sqrt( 2.0 * c - 1.0 );
    const double B = c - log( 4.0 );
    const double Q = c + 1.0 / A;
    const double T = 4.5;
    const double D = 1.0 + log( T );
    const double C = 1.0 + c / M_E;
    if ( c < 1.0 ) {
      while ( true ) {
        const double p = C * rng->rand();
        if ( p > 1.0 ) {
          y = -log( ( C - p ) / c );
          if ( rng->rand() <= pow( y, c - 1.0 ) ) break;
        }
        else {
          y = pow( p, 1.0 / c );
          if ( rng->rand() <= exp( -y ) ) break;
        }
      }
    }
    else if ( c == 1.0 ) {
      // exponential variate
      y = -log(rng->randDblExc());
    }
    else {
      while ( true ) {
        const double p1 = rng->rand();
        const double p2 = rng->rand();
        const double v = A * log( p1 / ( 1.0 - p1 ) );
        y = c * exp( v );
        const double z = p1 * p1 * p2;
        const double w = B + Q * v - y;
        if ( w + D - T * z >= 0.0 || w >= log( z ) ) break;
      }
    }
    if (a == 0.0) {
      if (b == 1.0) return y;
      else return b*y;
    }
    else {
      if (b == 1.0) return a + y;
      else return a + b*y;
    }
  }
}
