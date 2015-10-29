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

#include "matrixFloat.h"
#include "smart_ptr.h"
#include "sparse_matrixFloat.h"
#include "statistical_distribution.h"

namespace Stats {

  /// Normal distribution with general covariance matrix
  class GeneralNormalDistribution : public StatisticalDistributionBase {
  protected:
    AprilUtils::SharedPtr<Basics::MatrixFloat> mean, cov, inv_cov, L;
    AprilUtils::log_float cov_det, K;
    float cov_det_sign;

    void updateParams();
    virtual void privateSample(Basics::MTRand *rng,
                               Basics::MatrixFloat *result);
    virtual void privateLogpdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogcdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogpdfDerivative(const Basics::MatrixFloat *x,
                                         Basics::MatrixFloat *result);
    
  public:
    GeneralNormalDistribution(Basics::MatrixFloat *mean,
                              Basics::MatrixFloat *cov);
    virtual ~GeneralNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    
    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };
  
  /// Normal distribution with diagonal covariance sparse matrix
  class DiagonalNormalDistribution : public StatisticalDistributionBase {
  protected:
    AprilUtils::SharedPtr<Basics::MatrixFloat> mean;
    AprilUtils::log_float cov_det, K;
    float cov_det_sign;
    AprilUtils::SharedPtr<Basics::SparseMatrixFloat> cov, inv_cov, L;

    void updateParams();
    virtual void privateSample(Basics::MTRand *rng,
                               Basics::MatrixFloat *result);
    virtual void privateLogpdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogcdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogpdfDerivative(const Basics::MatrixFloat *x,
                                         Basics::MatrixFloat *result);
    
  public:
    DiagonalNormalDistribution(Basics::MatrixFloat *mean,
                               Basics::SparseMatrixFloat *cov);
    virtual ~DiagonalNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    
    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };

  /// Normal distribution with mean=0 and var=1
  class StandardNormalDistribution : public StatisticalDistributionBase {
  protected:
    AprilUtils::log_float K;
    
    virtual void privateSample(Basics::MTRand *rng,
                               Basics::MatrixFloat *result);
    virtual void privateLogpdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogcdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogpdfDerivative(const Basics::MatrixFloat *x,
                                         Basics::MatrixFloat *result);
    
  public:
    StandardNormalDistribution();
    virtual ~StandardNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    
    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };
  

  /// Log-Normal distribution with general covariance matrix
  class GeneralLogNormalDistribution : public GeneralNormalDistribution {
    AprilUtils::SharedPtr<Basics::MatrixFloat> location;
  
  protected:
    virtual void privateSample(Basics::MTRand *rng,
                               Basics::MatrixFloat *result);
    virtual void privateLogpdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogcdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogpdfDerivative(const Basics::MatrixFloat *x,
                                         Basics::MatrixFloat *result);
    
  public:
    GeneralLogNormalDistribution(Basics::MatrixFloat *mean,
                                 Basics::MatrixFloat *cov,
                                 Basics::MatrixFloat *location=0);
    virtual ~GeneralLogNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    
    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };

  /// Log-LogNormal distribution with diagonal covariance sparse matrix
  class DiagonalLogNormalDistribution : public DiagonalNormalDistribution {
    AprilUtils::SharedPtr<Basics::MatrixFloat> location;

  protected:
    virtual void privateSample(Basics::MTRand *rng,
                               Basics::MatrixFloat *result);
    virtual void privateLogpdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogcdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogpdfDerivative(const Basics::MatrixFloat *x,
                                         Basics::MatrixFloat *result);
    
  public:
    DiagonalLogNormalDistribution(Basics::MatrixFloat *mean,
                                  Basics::SparseMatrixFloat *cov,
                                  Basics::MatrixFloat *location=0);
    virtual ~DiagonalLogNormalDistribution();
    virtual StatisticalDistributionBase *clone();
    
    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };

}

#endif // NORMAL_DISTRIBUTION_H
