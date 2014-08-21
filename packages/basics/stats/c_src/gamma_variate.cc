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

using basics::MTRand;

namespace Stats {
  // Modification of: http://ftp.arl.mil/random/random.pdf
  double gammaVariate(MTRand *rng, double a, double b, double c) {
    double result;
    april_assert( b > 0.0f && c > 0.0f );
    const double A = 1. / sqrt( 2.0f * c - 1.0f );
    const double B = c - log( 4.0f );
    const double Q = c + 1.0f / A;
    const double T = 4.5f;
    const double D = 1.0f + log( T );
    const double C = 1.0f + c / M_E;
    if ( c < 1.0f ) {
      while ( true ) {
        double p = C * rng->rand();
        if ( p > 1.0f ) {
          double y = -log( ( C - p ) / c );
          if ( rng->rand() <= pow( y, c - 1.0f ) ) {
            result = a + b * y;
            break;
          }
        }
        else {
          double y = pow( p, 1.0f / c );
          if ( rng->rand() <= exp( -y ) ) {
            result = a + b * y;
            break;
          }
        }
      }
    }
    else if ( c == 1.0f ) {
      // exponential variate
      result = -log(rng->randDblExc());
    }
    else {
      while ( true ) {
        double p1 = rng->rand();
        double p2 = rng->rand();
        double v = A * log( p1 / ( 1.0f - p1 ) );
        double y = c * exp( v );
        double z = p1 * p1 * p2;
        double w = B + Q * v - y;
        if ( w + D - T * z >= 0.0f || w >= log( z ) ) {
          result = a + b * y;
          break;
        }
      }
    }
    return result;
  }
}
