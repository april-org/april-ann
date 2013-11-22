/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#ifndef STOCHASTICCOMPONENT_H
#define STOCHASTICCOMPONENT_H

#include "april_assert.h"
#include "ann_component.h"
#include "MersenneTwister.h"

namespace ANN {
  
  class StochasticANNComponent : public ANNComponent {
  protected:
    MTRand *random;
  public:
    StochasticANNComponent(MTRand *random,
			   const char *name=0,
			   const char *weights=0,
			   unsigned int input_size=0,
			   unsigned int output_size=0) :
      ANNComponent(name, weights, input_size, output_size),
      random(random) {
      april_assert(random != 0 && "Needs a random object\n");
      IncRef(random);
    }
    virtual ~StochasticANNComponent() {
      DecRef(random);
    }
    
    void setRandom(MTRand *random) {
      DecRef(this->random);
      this->random = random;
      IncRef(this->random);
    }
    
    MTRand *getRandom() { return random; }
    
    const MTRand *getRandom() const { return random; }
  };
}

#endif // STOCHASTICCOMPONENT_H
