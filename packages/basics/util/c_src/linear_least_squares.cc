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
// f(x) = a + b*x
//
//      n*sum(xi*yi) - (sum(xi)*sum(yi))
// b = --------------------------------
//      n*sum(xi^2) - (sum(xi))^2
//
// a = sum(yi)/n - b*(sum(xi)/n)

#include "linear_least_squares.h"

void least_squares(double x[], double y[], int numPoints, double &a, double &b) {
  double sum_xi=0, sum_yi=0, sum_xi_2=0, sum_xi_yi=0;
  for (int i=0; i<numPoints; ++i) {
    sum_xi    += x[i];
    sum_yi    += y[i];
    sum_xi_2  += x[i]*x[i];
    sum_xi_yi += x[i]*y[i];
  }
  b = (numPoints*sum_xi_yi - sum_xi*sum_yi)/(numPoints*sum_xi_2 - sum_xi*sum_xi);
  a = (sum_yi/numPoints) - b*(sum_xi/numPoints);
}

