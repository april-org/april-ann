/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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
#include "activations_action.h"

namespace ANN {

  MTRand *rnd = new MTRand();
  
  ActivationsAction::ActivationsAction(const ANNConfiguration &conf,
				       ActivationUnits     *units,
				       ActivationFunction *act_func) :
    Action(conf), units(units), act_func(act_func) {
    IncRef(units);
    IncRef(act_func);
  }
  
  ActivationsAction::~ActivationsAction() {
    DecRef(units);
    DecRef(act_func);
  }
  
  void ActivationsAction::doForward(bool during_training) {
    act_func->applyActivation(units->getPtr(),
			      units->size(),
			      conf,
			      conf.use_cuda_flag);
    if (during_training && units->getType() != OUTPUTS_TYPE) {
      // dropout 50%
      FloatGPUMirroredMemoryBlock *unit_ptr = units->getPtr();
      float min = act_func->getMinimum();
      if (min > -10.0f) {
	if (units->getType() == INPUTS_TYPE)
	  units->drop_factor = 0.3;
	units->drop_factor = 0.5;
	unsigned int size = units->numNeurons();
	int *unit_order = new int[size];
	for (unsigned int i=0; i<size; ++i)
	  unit_order[i] = static_cast<int>(i);
	float *unit = unit_ptr->getPPALForReadAndWrite();
	for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	  rnd->shuffle(size, unit_order);
	  for (unsigned int i=0; i<size*units->drop_factor; ++i)
	    unit[b + unit_order[i]*conf.max_bunch_size] = min;
	}
	delete[] unit_order;
      }
    }
  }
  
  void ActivationsAction::doBackprop() {
    act_func->multiplyDerivatives(units->getPtr(),
				  units->getErrorVectorPtr(),
				  units->size(),
				  conf,
				  conf.use_cuda_flag,
				  (units->getType() == OUTPUTS_TYPE));
  }

  void ActivationsAction::doUpdate() {
  }
  
  Action *ActivationsAction::clone(hash<void*,void*> &clone_dict,
				   const ANNConfiguration &conf) {
    return new ActivationsAction(conf,
				 static_cast<ActivationUnits*>(clone_dict[units]),
				 act_func->clone());
  }
}
