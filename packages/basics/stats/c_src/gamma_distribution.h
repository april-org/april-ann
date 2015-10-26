/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2015, Francisco Zamora-Martinez
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
#ifndef GAMMA_DISTRIBUTION_H
#define GAMMA_DISTRIBUTION_H

#include "logbase.h"
#include "smart_ptr.h"
#include "statistical_distribution.h"

namespace Stats {

  class GammaDistribution : public StatisticalDistributionBase {
    /// A positive real number. The scale parameter.
    AprilUtils::SharedPtr<Basics::MatrixFloat> alpha;
    /// A positive real number. The rate paremater.
    AprilUtils::SharedPtr<Basics::MatrixFloat> beta;
    
    float alphaf; ///< The scale parameter.
    float betaf;  /// The rate parameter.
    AprilUtils::log_float CTE;
    
    void updateParams();
    
  protected:
    virtual void privateSample(Basics::MTRand *rng, Basics::MatrixFloat *result);
    virtual void privateLogpdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    virtual void privateLogcdf(const Basics::MatrixFloat *x,
                               Basics::MatrixFloat *result);
    
  public:
    GammaDistribution(Basics::MatrixFloat *alpha, Basics::MatrixFloat *beta);
    virtual ~GammaDistribution();
    virtual StatisticalDistributionBase *clone();

    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };
  
}

#endif // GAMMA_DISTRIBUTION_H
