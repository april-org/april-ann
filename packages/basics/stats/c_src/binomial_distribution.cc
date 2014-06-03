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
#include <ctgmath> // FIXME: this library needs c++11 extensions
#include "buffer_list.h"
#include "error_print.h"
#include "binomial_distribution.h"
#include "gamma_variate.h"
#include "utilMatrixFloat.h"

namespace Stats {
  
  BinomialDistribution::BinomialDistribution(MatrixFloat *n, MatrixFloat *p) :
    StatisticalDistributionBase(1),
    n(n), p(p) {
    if (n->getNumDim() != 1 || p->getNumDim() != 1 ||
        n->getDimSize(0) != 1 || p->getDimSize(0) != 1)
      ERROR_EXIT(128, "Expected n,p one-dimensional matrices with size 1\n");
    updateParams();
  }

  BinomialDistribution::~BinomialDistribution() {
  }
  
  void BinomialDistribution::updateParams() {
    nf = (*n)(0);
    pf  = (*p)(0);
    if (!(nf > 0.0f) || !(pf > 0.0f))
      ERROR_EXIT(128, "Binomial distribution needs > 0 n and p params\n");
  }
  
  void BinomialDistribution::privateSample(MTRand *rng,
                                       MatrixFloat *result) {
    // generation via gamma variate
    for (MatrixFloat::iterator result_it(result->begin());
         result_it != result->end(); ++result_it) {
      double y1 = gammaVariate(rng, 0.0f, 1.0f, nf);
      double y2 = gammaVariate(rng, 0.0f, 1.0f, pf);
      *result_it = static_cast<float>( y1 / (y1 + y2) );
    }
  }
  
  void BinomialDistribution::privateLogpdf(const MatrixFloat *x,
                                       MatrixFloat *result) {
    float min,max;
    x->minAndMax(min,max);
    if (min < 0.0f || max > 1.0f)
      ERROR_EXIT(128, "Binomial dist. is only defined in range [0,1]\n");
    MatrixFloat::const_iterator x_it(x->begin());
    MatrixFloat::iterator result_it(result->begin());
    while(x_it != x->end()) {
      log_float vx  = log_float::from_float(*x_it).raise_to(nf - 1.0f);
      log_float v1x = log_float::from_float(1.0f - *x_it).raise_to(pf - 1.0f);
      log_float r;
      if (vx <= log_float::zero() || v1x <= log_float::zero())
        r = log_float::zero();
      *result_it = r.log();
        ++x_it;
      ++result_it;
    }
  }

  void BinomialDistribution::privateLogcdf(const MatrixFloat *x,
                                       MatrixFloat *result) {
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(result);
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
  }

  StatisticalDistributionBase *BinomialDistribution::clone() {
    return new BinomialDistribution(n->clone(), p->clone());
  }
  
  MatrixFloatSet *BinomialDistribution::getParams() {
    MatrixFloatSet *dict = new MatrixFloatSet();
    dict->insert("n", n);
    dict->insert("p", p);
    return dict;
  }
  
  char *BinomialDistribution::toLuaString(bool is_ascii) const {
    UNUSED_VARIABLE(is_ascii);
    buffer_list buffer;
    int len;
    buffer.printf("stats.dist.binomial(%g, %g)", nf, pf);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  
}
