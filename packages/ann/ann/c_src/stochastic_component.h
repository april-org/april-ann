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
#include "vector.h"

namespace ANN {

  /// This class is a base class for stochastic components. It contains a random
  /// MTRand object and other properties in order to produce random numbers. The
  /// sequence of random numbers could be stored to repeat it several times. For
  /// this purpose, the reset method receives an iteration number. So, in the
  /// first iteration (number 0), the object stores the sequence of random
  /// objects. In the following iterations, this sequence is repeated.
  class StochasticANNComponent : public ANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(StochasticANNComponent);
    
    enum StochasticState {
      NORMAL     = 0,
      KEEP       = 1,
      FROZEN     = 2
    };
    StochasticState stochastic_state;
    unsigned int last_reset_it;
    uint32_t *random_frozen_state;
  protected:
    Basics::MTRand   *random;
  public:
    StochasticANNComponent(Basics::MTRand *random,
			   const char *name=0,
			   const char *weights=0,
			   unsigned int input_size=0,
			   unsigned int output_size=0) :
      ANNComponent(name, weights, input_size, output_size),
      stochastic_state(NORMAL),
      last_reset_it(0),
      random(random) {
      april_assert(random != 0 && "Needs a random object\n");
      IncRef(random);
      random_frozen_state = new uint32_t[Basics::MTRand::SAVE];
    }
    virtual ~StochasticANNComponent() {
      DecRef(random);
      delete[] random_frozen_state;
    }

    virtual Basics::Token *doForward(Basics::Token* input,
                                     bool during_training) {
      if (!during_training) stochastic_state = NORMAL;
      return input;
    }
    
    virtual void reset(unsigned int it=0) {
      if (it == 0) {
	// first iteration, goes to KEEP state
	switch(stochastic_state) {
	case FROZEN:
	case NORMAL:
	  stochastic_state = KEEP;
	  random->save(random_frozen_state);
	  break;
	default:
	  ;
	}
      }
      else if (it != last_reset_it) {
	// iteration change, reinitialize the store sequence position pointer
	// and goes to FROZEN state
	stochastic_state = FROZEN;
	random->load(random_frozen_state);
      }
      last_reset_it = it;
    }
    
    /// Method to restore previously serialized random object
    virtual void setRandom(Basics::MTRand *random) {
      DecRef(this->random);
      this->random = random;
      IncRef(this->random);
      last_reset_it = 0;
    }
    
    /// Method to serialize the underlying random object
    virtual Basics::MTRand *getRandom() { return random; }
    
    /// Method to serialize the underlying random object
    virtual const Basics::MTRand *getRandom() const { return random; }
  };
}

#endif // STOCHASTICCOMPONENT_H
