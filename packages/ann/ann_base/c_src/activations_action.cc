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
  
  void ActivationsAction::doForward() {
    act_func->applyActivation(units->getPtr(),
			      units->size(),
			      conf,
			      conf.use_cuda_flag);
  }
  
  void ActivationsAction::doBackward() {
    act_func->multiplyDerivatives(units->getPtr(),
				  units->getErrorVectorPtr(),
				  units->size(),
				  conf,
				  conf.use_cuda_flag);
  }
  
  Action *ActivationsAction::clone(hash<void*,void*> &clone_dict,
				   const ANNConfiguration &conf) {
    return new ActivationsAction(conf,
				 static_cast<ActivationUnits*>(clone_dict[units]),
				 act_func->clone());
  }
}
