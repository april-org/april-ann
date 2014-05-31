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
#include "buffer_list.h"
#include "uniform_distribution.h"
#include "utilMatrixFloat.h"

namespace Stats {

#define CHECK_MATRIX(a,b) do {                          \
    if ( (b) != 0 ) {                                   \
      if ( ! (a)->sameDim((b)) ) {                      \
        ERROR_EXIT(128, "Incorrect matrix sizes\n");    \
      }                                                 \
    }                                                   \
  } while(0)

  UniformDistribution::UniformDistribution(MatrixFloat *low,
                                           MatrixFloat *high) :
    StatisticalDistributionBase(),
    low(low), high(high) {
    IncRef(low);
    IncRef(high);
    CHECK_MATRIX(low, high);
    if (low->getNumDim() != 1)
      ERROR_EXIT(128, "Expected one-dimensional low and high matrices\n");
    MatrixFloat::const_iterator low_it(low->begin());
    MatrixFloat::const_iterator high_it(high->begin());
    while(low_it != low->end()) {
      if (*high_it < *low_it)
        ERROR_EXIT(128, "High must be always higher than low matrix\n");
      ++low_it;
      ++high_it;
    }
  }

  UniformDistribution::~UniformDistribution() {
    DecRef(low);
    DecRef(high);
  }

  MatrixFloat *UniformDistribution::sample(MTRand *rng,
                                           MatrixFloat *result) const {
    if (result == 0) result = low->cloneOnlyDims();
    else CHECK_MATRIX(low, result);
    MatrixFloat::const_iterator low_it(low->begin()), high_it(high->begin());
    for (MatrixFloat::iterator it(result->begin()); it != result->end();
         ++it, ++low_it, ++high_it) {
      float s = *high_it - *low_it;
      *it = rng->rand(s) + *low_it;
    }
    return result;
  }

  MatrixFloat *UniformDistribution::pdf(const MatrixFloat *x,
                                        MatrixFloat *result) const {
    if (result == 0) result = low->cloneOnlyDims();
    else CHECK_MATRIX(low, result);
    CHECK_MATRIX(low, x);
    MatrixFloat::const_iterator low_it(low->begin()), high_it(high->begin());
    MatrixFloat::const_iterator x_it(x->begin());
    for (MatrixFloat::iterator result_it(result->begin());
         result_it != result->end();
         ++result_it, ++low_it, ++high_it, ++x_it) {
      if (*low_it <= *x_it && *x_it <= *high_it)
        *result_it = 1.0f / (*high_it - *low_it);
      else *result_it = 0.0f;
    }
    return result;
  }

  MatrixFloat *UniformDistribution::cdf(const MatrixFloat *x,
                                        MatrixFloat *result) const {
    if (result == 0) result = low->cloneOnlyDims();
    else CHECK_MATRIX(low, result);
    CHECK_MATRIX(low, x);
    MatrixFloat::const_iterator low_it(low->begin()), high_it(high->begin());
    MatrixFloat::const_iterator x_it(x->begin());
    for (MatrixFloat::iterator result_it(result->begin());
         result_it != result->end();
         ++result_it, ++low_it, ++high_it, ++x_it) {
      if (*x_it < *low_it)
        *result_it = 0.0f;
      else if (*low_it <= *x_it && *x_it < *high_it)
        *result_it = (*x_it - *low_it) / (*high_it - *low_it);
      else
        *result_it = 1.0f;
    }
    return result;
  }

  StatisticalDistributionBase *UniformDistribution::clone() {
    return new UniformDistribution(low->clone(), high->clone());
  }

  MatrixFloatSet *UniformDistribution::getParams() const {
    MatrixFloatSet *dict = new MatrixFloatSet();
    dict->insert("low",  low);
    dict->insert("high", high);
    return dict;
  }
  
  char *UniformDistribution::toLuaString(bool is_ascii) const {
    buffer_list buffer;
    char *low_str, *high_str;
    int len;
    low_str = writeMatrixFloatToString(low, is_ascii, len);
    high_str = writeMatrixFloatToString(high, is_ascii, len);
    buffer.printf("stats.dist.uniform(matrix.fromString[[%s]], matrix.fromString[[%s]])",
                  low_str, high_str);
    delete[] low_str;
    delete[] high_str;
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
