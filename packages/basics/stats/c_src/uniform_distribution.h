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
#ifndef UNIFORM_DISTRIBUTION_H
#define UNIFORM_DISTRIBUTION_H

#include "statistical_distribution.h"

namespace Stats {
  
  class UniformDistribution : public StatisticalDistributionBase {
    MatrixFloat *low, *high;
  public:
    UniformDistribution(MatrixFloat *low, MatrixFloat *high);
    virtual ~UniformDistribution();
    virtual MatrixFloat *sample(MTRand *rng, MatrixFloat *result=0);
    virtual log_float logpdf(const MatrixFloat *x);
    virtual log_float logcdf(const MatrixFloat *x);
    virtual StatisticalDistributionBase *clone();
    virtual MatrixFloatSet *getParams();
    virtual char *toLuaString(bool is_ascii) const;
  };
  
}

#endif // UNIFORM_DISTRIBUTION_H
