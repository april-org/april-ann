/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
#ifndef ACTIVATIONSACTION_H
#define ACTIVATIONSACTION_H 

#include "action.h"
#include "actunit.h"
#include "activation_function.h"

namespace ANN {
  class ActivationsAction : public Action {
    ActivationUnits             *units;
    ActivationFunction          *act_func;
    FloatGPUMirroredMemoryBlock *dropout_mask;
    float                        dropout;
    int                         *units_order_permutation;
  public:
    ActivationsAction(const ANNConfiguration &conf,
		      ActivationUnits    *units,
		      ActivationFunction *act_func);
    virtual ~ActivationsAction();
    virtual void doForward(bool during_training=false);
    virtual void doBackprop();
    virtual void doUpdate();
    virtual Action *clone(hash<void*,void*> &clone_dict,
			  const ANNConfiguration &conf);
    void transferFanInToConnections() { }
    void setOption(const char *name, double value);
    bool hasOption(const char *name);
    double getOption(const char *name);
  };
}

#endif // ACTIVATIONSACTION_H
