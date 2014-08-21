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
#ifndef EXPONENTIAL_DISTRIBUTION_H
#define EXPONENTIAL_DISTRIBUTION_H

#include "logbase.h"
#include "statistical_distribution.h"

namespace Stats {

  class ExponentialDistribution : public StatisticalDistributionBase {
    basics::MatrixFloat *lambda, *inv_lambda;
    april_utils::log_float lambda_prod;
    
    void updateParams();

  protected:
    virtual void privateSample(basics::MTRand *rng,
                               basics::MatrixFloat *result);
    virtual void privateLogpdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogcdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    
  public:
    ExponentialDistribution(basics::MatrixFloat *lambda);
    virtual ~ExponentialDistribution();
    virtual StatisticalDistributionBase *clone();
    virtual char *toLuaString(bool is_ascii) const;
  };
  
}

#endif // EXPONENTIAL_DISTRIBUTION_H
