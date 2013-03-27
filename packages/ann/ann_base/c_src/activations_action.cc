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
#include "trainsuper.h"
#include "activations_action.h"
#include "wrapper.h"

namespace ANN {

  ActivationsAction::ActivationsAction(const ANNConfiguration &conf,
				       ActivationUnits        *units,
				       ActivationFunction     *act_func) :
    Action(conf), units(units), act_func(act_func),
    dropout(0.0f) {
    IncRef(units);
    IncRef(act_func);
    dropout_mask = new FloatGPUMirroredMemoryBlock(units->numNeurons()*conf.max_bunch_size);
    units_order_permutation = new int[units->numNeurons()];
  }
  
  ActivationsAction::~ActivationsAction() {
    DecRef(units);
    DecRef(act_func);
    delete dropout_mask;
    delete[] units_order_permutation;
  }
  
  void ActivationsAction::doForward(bool during_training) {
    act_func->applyActivation(units->getPtr(),
			      units->size(),
			      conf,
			      conf.use_cuda_flag);
    if (dropout > 0.0f) {
      if (during_training) {
	unsigned int size   = units->numNeurons();
	float *mask_ptr     = dropout_mask->getPPALForWrite();
	float dropout_value = act_func->getDropoutValue();
	/*
	// initialize permutation vector
	for (unsigned int i=0; i<size; ++i)
	units_order_permutation[i] = static_cast<int>(i);
	// auxiliar variables
	// configure dropout mask for each bunch of activations
	for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	conf.rnd.shuffle(size, units_order_permutation);
	unsigned int i;
	// first size*dropout units are masked
	for (i=0; i<size*dropout; ++i)
	mask_ptr[b + units_order_permutation[i]*conf.max_bunch_size] = 0.0f;
	// the rest are unmasked
	for (; i<size; ++i)
	mask_ptr[b + units_order_permutation[i]*conf.max_bunch_size] = 1.0f;
	}
	*/
	for (unsigned int i=0; i<size*conf.max_bunch_size; ++i)
	  if (conf.rnd.rand() < dropout) mask_ptr[i] = 0.0f;
	  else mask_ptr[i] = 1.0f;
	// apply mask
       	applyMask(units->getPtr(), dropout_mask, dropout_value,
		  units->size(), conf, conf.use_cuda_flag);
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
    ActivationsAction *action = new
      ActivationsAction(conf,
			static_cast<ActivationUnits*>(clone_dict[units]),
			act_func->clone());
    action->dropout            = dropout;
    action->units->drop_factor = dropout;
    return action;
  }
  
  void ActivationsAction::setOption(const char *name, double value) {
    if (strcmp(name, "dropout") == 0) {
      if (units->getType() != INPUTS_TYPE &&
	  units->getType() != OUTPUTS_TYPE) {
	dropout = value;
	units->drop_factor = dropout;
      }
    }
  }
  
  double ActivationsAction::getOption(const char *name) {
    mGetOption("dropout", dropout);
    ERROR_EXIT(140, "The option to be get does not exist.\n");
  }
  
  bool ActivationsAction::hasOption(const char *name) {
    mHasOption("dropout");
    return false;
  }
}
