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

  /// Normal distribution with general covariance matrix
  class GeneralNormalDistribution : public StatisticalDistributionBase {
  protected:
    basics::MatrixFloat *mean, *cov, *inv_cov, *L;
    april_utils::log_float cov_det, K;
    float cov_det_sign;

    void updateParams();
    virtual void privateSample(basics::MTRand *rng,
                               basics::MatrixFloat *result);
    virtual void privateLogpdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogcdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogpdfDerivative(const basics::MatrixFloat *x,
                                         basics::MatrixFloat *result);
    
  public:
    GeneralNormalDistribution(basics::MatrixFloat *mean,
                              basics::MatrixFloat *cov);
    virtual ~GeneralNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    virtual char *toLuaString(bool is_ascii) const;
  };

  /// Normal distribution with diagonal covariance sparse matrix
  class DiagonalNormalDistribution : public StatisticalDistributionBase {
  protected:
    basics::MatrixFloat *mean;
    april_utils::log_float cov_det, K;
    float cov_det_sign;
    basics::SparseMatrixFloat *cov, *inv_cov, *L;

    void updateParams();
    virtual void privateSample(basics::MTRand *rng,
                               basics::MatrixFloat *result);
    virtual void privateLogpdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogcdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogpdfDerivative(const basics::MatrixFloat *x,
                                         basics::MatrixFloat *result);
    
  public:
    DiagonalNormalDistribution(basics::MatrixFloat *mean,
                               basics::SparseMatrixFloat *cov);
    virtual ~DiagonalNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    virtual char *toLuaString(bool is_ascii) const;
  };

  /// Normal distribution with mean=0 and var=1
  class StandardNormalDistribution : public StatisticalDistributionBase {
  protected:
    april_utils::log_float K;
    
    virtual void privateSample(basics::MTRand *rng,
                               basics::MatrixFloat *result);
    virtual void privateLogpdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogcdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogpdfDerivative(const basics::MatrixFloat *x,
                                         basics::MatrixFloat *result);
    
  public:
    StandardNormalDistribution();
    virtual ~StandardNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    virtual char *toLuaString(bool is_ascii) const;
  };
  

  /// Log-Normal distribution with general covariance matrix
  class GeneralLogNormalDistribution : public GeneralNormalDistribution {
    basics::MatrixFloat *location;
  
  protected:
    virtual void privateSample(basics::MTRand *rng,
                               basics::MatrixFloat *result);
    virtual void privateLogpdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogcdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogpdfDerivative(const basics::MatrixFloat *x,
                                         basics::MatrixFloat *result);
    
  public:
    GeneralLogNormalDistribution(basics::MatrixFloat *mean,
                                 basics::MatrixFloat *cov,
                                 basics::MatrixFloat *location=0);
    virtual ~GeneralLogNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    virtual char *toLuaString(bool is_ascii) const;
  };

  /// Log-LogNormal distribution with diagonal covariance sparse matrix
  class DiagonalLogNormalDistribution : public DiagonalNormalDistribution {
    basics::MatrixFloat *location;

  protected:
    virtual void privateSample(basics::MTRand *rng,
                               basics::MatrixFloat *result);
    virtual void privateLogpdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogcdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogpdfDerivative(const basics::MatrixFloat *x,
                                         basics::MatrixFloat *result);
    
  public:
    DiagonalLogNormalDistribution(basics::MatrixFloat *mean,
                                  basics::SparseMatrixFloat *cov,
                                  basics::MatrixFloat *location=0);
    virtual ~DiagonalLogNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    virtual char *toLuaString(bool is_ascii) const;
  };

}

#endif // NORMAL_DISTRIBUTION_H
