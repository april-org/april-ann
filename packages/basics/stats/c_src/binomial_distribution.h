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
#ifndef BINOMIAL_DISTRIBUTION_H
#define BINOMIAL_DISTRIBUTION_H

#include "logbase.h"
#include "statistical_distribution.h"

namespace Stats {

  class BinomialDistribution : public StatisticalDistributionBase {
    basics::MatrixFloat *n, *p;
    int ni;
    float pf, pfm1;
    april_utils::log_float lpf, lpfm1;
    
    void updateParams();
    april_utils::log_float computeDensity(int k);

  protected:
    virtual void privateSample(basics::MTRand *rng,
                               basics::MatrixFloat *result);
    virtual void privateLogpdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    virtual void privateLogcdf(const basics::MatrixFloat *x,
                               basics::MatrixFloat *result);
    
  public:
    BinomialDistribution(basics::MatrixFloat *alpha,
                         basics::MatrixFloat *binomial);
    virtual ~BinomialDistribution();
    virtual StatisticalDistributionBase *clone();
    virtual char *toLuaString(bool is_ascii) const;
  };
  
}

#endif // BINOMIAL_DISTRIBUTION_H
