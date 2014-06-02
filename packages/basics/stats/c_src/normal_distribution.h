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
#ifndef NORMAL_DISTRIBUTION_H
#define NORMAL_DISTRIBUTION_H

#include "sparse_matrixFloat.h"
#include "statistical_distribution.h"

namespace Stats {

  class GeneralNormalDistribution : public StatisticalDistributionBase {
    MatrixFloat *mean, *cov, *inv_cov, *L;
    log_float cov_det, K;
    float cov_det_sign;

  protected:
    virtual void privateSample(MTRand *rng, MatrixFloat *result);
    virtual void privateLogpdf(const MatrixFloat *x, MatrixFloat *result);
    virtual void privateLogcdf(const MatrixFloat *x, MatrixFloat *result);
    
  public:
    GeneralNormalDistribution(MatrixFloat *mean, MatrixFloat *cov);
    virtual ~GeneralNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    virtual MatrixFloatSet *getParams();
    virtual char *toLuaString(bool is_ascii) const;
    virtual void updateParams();
  };

  class DiagonalNormalDistribution : public StatisticalDistributionBase {
    MatrixFloat *mean;
    log_float cov_det, K;
    float cov_det_sign;
    SparseMatrixFloat *cov, *inv_cov, *L;

  protected:
    virtual void privateSample(MTRand *rng, MatrixFloat *result);
    virtual void privateLogpdf(const MatrixFloat *x, MatrixFloat *result);
    virtual void privateLogcdf(const MatrixFloat *x, MatrixFloat *result);
    
  public:
    DiagonalNormalDistribution(MatrixFloat *mean, SparseMatrixFloat *cov);
    virtual ~DiagonalNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    virtual MatrixFloatSet *getParams();
    virtual char *toLuaString(bool is_ascii) const;
    virtual void updateParams();
  };

}

#endif // NORMAL_DISTRIBUTION_H
