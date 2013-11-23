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
    enum StochasticState {
      NORMAL     = 0,
      KEEP       = 1,
      FROZEN     = 2
    };
    StochasticState stochastic_state;
    unsigned int last_reset_it;
    MTRand *random;
    april_utils::vector<double> stored_sequence;
    unsigned int stored_pos;
  public:
    StochasticANNComponent(MTRand *random,
			   const char *name=0,
			   const char *weights=0,
			   unsigned int input_size=0,
			   unsigned int output_size=0) :
      ANNComponent(name, weights, input_size, output_size),
      stochastic_state(NORMAL),
      last_reset_it(0),
      random(random),
      stored_pos(0) {
      april_assert(random != 0 && "Needs a random object\n");
      IncRef(random);
    }
    virtual ~StochasticANNComponent() {
      DecRef(random);
    }

    virtual Token *doForward(Token* input, bool during_training) {
      if (!during_training) stochastic_state = NORMAL;
      return input;
    }
    
    virtual void reset(unsigned int it=0) {
      if (it == 0) {
	// first iteration, goes to KEEP state, where the sequence of numbers is
	// stored
	switch(stochastic_state) {
	case FROZEN:
	case NORMAL:
	  stochastic_state = KEEP;
	  stored_pos       = 0;
	  stored_sequence.clear();
	  break;
	default:
	  ;
	}
      }
      else if (it != last_reset_it) {
	// iteration change, reinitialize the store sequence position pointer
	// and goes to FROZEN state
	stochastic_state = FROZEN;
	stored_pos       = 0;
      }
      last_reset_it = it;
    }
    
    /// Method to restore previously serialized random object
    virtual void setRandom(MTRand *random) {
      DecRef(this->random);
      this->random = random;
      IncRef(this->random);
    }
    
    /// Method to serialize the underlying random object
    virtual MTRand *getRandom() { return random; }
    
    /// Method to serialize the underlying random object
    virtual const MTRand *getRandom() const { return random; }
    
#define SAMPLE(value,FUNC) do {						\
      if (stored_sequence.size() > 10000000)				\
	ERROR_PRINT("WARNING!!! stochastec state sequence too large, "	\
		    "please, check that you are passing iteration "	\
		    "number at reset(...) method\n");			\
      switch(stochastic_state) {					\
      case NORMAL:							\
	(value) = (FUNC);						\
	break;								\
      case KEEP:							\
	(value) = (FUNC);						\
	stored_sequence.push_back( (value) );				\
	break;								\
      case FROZEN:							\
	if (stored_pos < stored_sequence.size())			\
	  (value) = stored_sequence[stored_pos];			\
	else {								\
	  (value) = (FUNC);						\
	  stored_sequence.push_back( (value) );				\
	}								\
	++stored_pos;							\
	break;								\
      default:								\
	;								\
      }									\
    } while(0)
    
    /// Uniform distribution random sampling function
    double rand() {
       double value=0.0f;
       SAMPLE(value, random->rand());
       return value;
    }
    
    /// Normal distribution random sampling function
    double randNorm( const double& mean=0.0, const double &variance = 0.0 ) {
      double value=0.0f;
      SAMPLE(value, random->randNorm(mean, variance));
      return value;
    }
#undef SAMPLE
  };
}

#endif // STOCHASTICCOMPONENT_H
