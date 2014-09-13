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
#ifndef BETA_DISTRIBUTION_H
#define BETA_DISTRIBUTION_H

#include "logbase.h"
#include "statistical_distribution.h"

namespace Stats {

  class BetaDistribution : public StatisticalDistributionBase {
    Basics::MatrixFloat *alpha, *beta;
    float alphaf, betaf;
    AprilUtils::log_float Bab;

    void updateParams();
    
  protected:
    virtual void privateSample(Basics::MTRand *rng, Basics::MatrixFloat *result);
    virtual void privateLogpdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogcdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    
  public:
    BetaDistribution(Basics::MatrixFloat *alpha, Basics::MatrixFloat *beta);
    virtual ~BetaDistribution();
    virtual StatisticalDistributionBase *clone();
    virtual char *toLuaString(bool is_ascii) const;
  };
  
}

#endif // BETA_DISTRIBUTION_H
