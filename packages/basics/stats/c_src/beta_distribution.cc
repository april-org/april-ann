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

using april_utils::buffer_list;
using april_utils::log_float;
using basics::MatrixFloat;
using basics::MTRand;

namespace Stats {
  
  BetaDistribution::BetaDistribution(MatrixFloat *alpha, MatrixFloat *beta) :
    StatisticalDistributionBase(1),
    alpha(alpha), beta(beta) {
    if (alpha->getNumDim() != 1 || beta->getNumDim() != 1 ||
        alpha->getDimSize(0) != 1 || beta->getDimSize(0) != 1)
      ERROR_EXIT(128, "Expected alpha,beta one-dimensional matrices with size 1\n");
    updateParams();
  }

  BetaDistribution::~BetaDistribution() {
  }
  
  void BetaDistribution::updateParams() {
    alphaf = (*alpha)(0);
    betaf  = (*beta)(0);
    if (!(alphaf > 0.0f) || !(betaf > 0.0f))
      ERROR_EXIT(128, "Beta distribution needs > 0 alpha and beta params\n");
    log_float gamma_a(lgamma(alphaf));
    log_float gamma_b(lgamma(betaf));
    log_float gamma_ab(lgamma(alphaf + betaf));
    Bab = ( gamma_a * gamma_b / gamma_ab );
  }
  
  void BetaDistribution::privateSample(MTRand *rng,
                                       MatrixFloat *result) {
    // generation via gamma variate
    for (MatrixFloat::iterator result_it(result->begin());
         result_it != result->end(); ++result_it) {
      double y1 = gammaVariate(rng, 0.0f, 1.0f, alphaf);
      double y2 = gammaVariate(rng, 0.0f, 1.0f, betaf);
      *result_it = static_cast<float>( y1 / (y1 + y2) );
    }
  }
  
  void BetaDistribution::privateLogpdf(const MatrixFloat *x,
                                       MatrixFloat *result) {
    MatrixFloat::const_iterator x_it(x->begin());
    MatrixFloat::iterator result_it(result->begin());
    while(x_it != x->end()) {
      if (*x_it < 0.0f || *x_it > 1.0f)
        *result_it = log_float::zero().log();
      else {
        log_float vx  = log_float::from_float(*x_it).raise_to(alphaf - 1.0f);
        log_float v1x = log_float::from_float(1.0f - *x_it).raise_to(betaf - 1.0f);
        log_float r;
        if (vx <= log_float::zero() || v1x <= log_float::zero())
          r = log_float::zero();
        else
          r = vx * v1x / Bab;
        *result_it = r.log();
      }
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
    return new BetaDistribution(alpha->clone(), beta->clone());
  }
  
  char *BetaDistribution::toLuaString(bool is_ascii) const {
    UNUSED_VARIABLE(is_ascii);
    buffer_list buffer;
    buffer.printf("stats.dist.beta(%g, %g)", alphaf, betaf);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  
}
