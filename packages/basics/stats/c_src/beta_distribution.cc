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
#include "beta_distribution.h"
#include "gamma_variate.h"
#include "utilMatrixFloat.h"

namespace Stats {
  
  BetaDistribution::BetaDistribution(float alpha, float beta) :
    StatisticalDistributionBase(1),
    alpha(alpha), beta(beta) {
    updateParams();
  }

  BetaDistribution::~BetaDistribution() {
  }
  
  void BetaDistribution::updateParams() {
    if (!(alpha > 0.0f) || !(beta > 0.0f))
      ERROR_EXIT(128, "Beta distribution needs > 0 alpha and beta params\n");
    log_float gamma_a(lgamma(alpha));
    log_float gamma_b(lgamma(beta));
    log_float gamma_ab(lgamma(alpha + beta));
    Bab = ( gamma_a * gamma_b / gamma_ab );
  }
  
  void BetaDistribution::privateSample(MTRand *rng,
                                       MatrixFloat *result) {
    // generation via gamma variate
    for (MatrixFloat::iterator result_it(result->begin());
         result_it != result->end(); ++result_it) {
      double y1 = gammaVariate(rng, 0.0f, 1.0f, alpha);
      double y2 = gammaVariate(rng, 0.0f, 1.0f, beta);
      *result_it = static_cast<float>( y1 / (y1 + y2) );
    }
  }
  
  void BetaDistribution::privateLogpdf(const MatrixFloat *x,
                                       MatrixFloat *result) {
    float min,max;
    x->minAndMax(min,max);
    if (min < 0.0f || max > 1.0f)
      ERROR_EXIT(128, "Beta dist. is only defined in range [0,1]\n");
    MatrixFloat::const_iterator x_it(x->begin());
    MatrixFloat::iterator result_it(result->begin());
    while(x_it != x->end()) {
      log_float vx  = log_float::from_float(*x_it).raise_to(alpha - 1.0f);
      log_float v1x = log_float::from_float(1.0f - *x_it).raise_to(beta - 1.0f);
      log_float r;
      if (vx <= log_float::zero() || v1x <= log_float::zero())
        r = log_float::zero();
      else
        r = vx * v1x / Bab;
      *result_it = r.log();
        ++x_it;
      ++result_it;
    }
  }

  void BetaDistribution::privateLogcdf(const MatrixFloat *x,
                                       MatrixFloat *result) {
    UNUSED_VARIABLE(x);
    UNUSED_VARIABLE(result);
    ERROR_EXIT(128, "NOT IMPLEMENTED\n");
  }

  StatisticalDistributionBase *BetaDistribution::clone() {
    return new BetaDistribution(alpha, beta);
  }
  
  MatrixFloatSet *BetaDistribution::getParams() {
    ERROR_EXIT(256, "NOT IMPLEMENTED\n");
    return 0;
  }
  
  char *BetaDistribution::toLuaString(bool is_ascii) const {
    UNUSED_VARIABLE(is_ascii);
    buffer_list buffer;
    int len;
    buffer.printf("stats.dist.beta(%g, %g)", alpha, beta);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  
}
