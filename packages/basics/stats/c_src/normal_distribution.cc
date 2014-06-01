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
#define _USE_MATH_DEFINES
#include <cmath>
#include "error_print.h"
#include "normal_distribution.h"

namespace Stats {
  
  const float M_2PI = M_PI*2.0f;
  
  GeneralNormalDistribution::GeneralNormalDistribution(MatrixFloat *mean,
                                                       MatrixFloat *cov) :
    mean(mean), cov(cov), inv_cov(0) {
    if (mean->getNumDim() != 1)
      ERROR_EXIT(128, "Expected one-dimensional mean matrix\n");
    if (cov->getNumDim() != 2 || cov->getDimSize(0) != cov->getDimSize(1))
      ERROR_EXIT(128, "Expected squared bi-dimensional cov matrix\n");
    if (mean->getDimSize(0) != cov->getDimSize(0))
      ERROR_EXIT(128, "Expected mean and cov matrix with same size\n");
    IncRef(mean);
    IncRef(cov);
    updateCov();
    diff = mean->cloneOnlyDims();
    IncRef(diff);
    mult = mean->cloneOnlyDims();
    IncRef(mult);
  }

  GeneralNormalDistribution::~GeneralNormalDistribution() {
    DecRef(mean);
    DecRef(cov);
    DecRef(diff);
    DecRef(mult);
  }
  
  void GeneralNormalDistribution::updateCov() {
    // K = 1 / sqrtf( 2*pi^k * |cov| )
    if (inv_cov != 0) DecRef(inv_cov);
    inv_cov = cov->inv();
    IncRef(inv_cov);
    if (inv_cov->getMajorOrder() != cov->getMajorOrder())
      AssignRef(inv_cov, inv_cov->clone(cov->getMajorOrder()));
    cov_det = cov->logDeterminant(cov_det_sign);
    log_float KM_2PI = log_float::from_float(M_2PI).
      raise_to(static_cast<float>(mean->getDimSize(0)));
    log_float denom = (KM_2PI * cov_det).raise_to(0.5f);
    K = log_float::one() / denom;
  }
  
  MatrixFloat *GeneralNormalDistribution::sample(MTRand *rng,
                                                 MatrixFloat *result) {
    
  }
  
  log_float GeneralNormalDistribution::logpdf(const MatrixFloat *x) {
    /*
      diff->copy(x);
      diff->axpy(-1.0, mean);
      mult->gemv(CblasNoTrans, 1.0f, inv_cov, diff, 0.0f);
      result->ger(-0.5f, diff, mult);
      result->scalarAdd( K.log() );
      return result;
    */
  }

  log_float GeneralNormalDistribution::logcdf(const MatrixFloat *x) {
  }

  StatisticalDistributionBase *GeneralNormalDistribution::clone() {
  }
  
  MatrixFloatSet *GeneralNormalDistribution::getParams() {
  }
  
  char *GeneralNormalDistribution::toLuaString(bool is_ascii) const {
  }
  
}
