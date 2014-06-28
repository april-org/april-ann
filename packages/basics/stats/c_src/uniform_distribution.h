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

#include "unused_variable.h"
#include "statistical_distribution.h"

namespace Stats {
  
  class UniformDistribution : public StatisticalDistributionBase {
    MatrixFloat *low, *high, *diff;
    
    void updateParams();

  protected:
    virtual void privateSample(MTRand *rng, MatrixFloat *result);
    virtual void privateLogpdf(const MatrixFloat *x, MatrixFloat *result);
    virtual void privateLogcdf(const MatrixFloat *x, MatrixFloat *result);
    virtual void privateLogpdfDerivative(const MatrixFloat *x,
                                         MatrixFloat *result) {
      UNUSED_VARIABLE(x);
      result->zeros();
    }

  public:
    UniformDistribution(MatrixFloat *low, MatrixFloat *high);
    virtual ~UniformDistribution();
    virtual StatisticalDistributionBase *clone();
    virtual char *toLuaString(bool is_ascii) const;
  };
  
}

#endif // UNIFORM_DISTRIBUTION_H
