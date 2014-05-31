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
#ifndef STATISTICAL_DISTRIBUTION_H
#define STATISTICAL_DISTRIBUTION_H

#include "matrixFloat.h"
#include "matrixFloatSet.h"
#include "MersenneTwister.h"
#include "referenced.h"

namespace Stats {
  
  class StatisticalDistributionBase : public Referenced {
  public:
    StatisticalDistributionBase() : Referenced() {}
    virtual ~StatisticalDistributionBase() {}
    virtual MatrixFloat *sample(MTRand *rng, MatrixFloat *result=0) const = 0;
    virtual MatrixFloat *pdf(const MatrixFloat *x, MatrixFloat *result=0) const = 0;
    virtual MatrixFloat *cdf(const MatrixFloat *x, MatrixFloat *result=0) const = 0;
    virtual StatisticalDistributionBase *clone() = 0;
    virtual MatrixFloatSet *getParams() const = 0;
    virtual char *toLuaString(bool is_ascii) const = 0;
  };
  
  ////////////////////////////////////////////////////////////////////////////
  /*

    class ExponentialDistribution : public StatisticalDistributionBase {
    MatrixFloat *lambda;
    };

    class DiagonalNormalDistribution : public StatisticalDistributionBase {
    MatrixFloat *mean;
    SparseMatrixFloat *cov;
    };

    class GeneralNormalDistribution : public StatisticalDistributionBase {
    MatrixFloat *mean, *cov;
    };

    class StudentDistribution : public StatisticalDistributionBase {
    MatrixFloat *v;
    };
  
    class GammaDistribution : public StatisticalDistributionBase {
    MatrixFloat *alpha, *beta;
    };
  
    class BetaDistribution : public StatisticalDistributionBase {
    MatrixFloat *alpha, *beta;
    };

    class BinomialDistribution : public StatisticalDistributionBase {
    MatrixFloat *p;
    };

    class MultinomialDistribution : public StatisticalDistributionBase {
    MatrixFloat *p;
    };

  */
  
}

#endif // STATISTICAL_DISTRIBUTION_H
