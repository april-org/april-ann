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
#ifndef SOFTPLUSACTFCOMPONENT_H
#define SOFTPLUSACTFCOMPONENT_H

#include "activation_function_component.h"
#include "ann_component.h"
#include "gpu_mirrored_memory_block.h"

namespace ANN {

  /// An abstract class that defines the basic interface that
  /// the anncomponents must fulfill.
  class SoftplusActfANNComponent : public ActivationFunctionANNComponent {
  protected:
    virtual void applyActivation(FloatGPUMirroredMemoryBlock *input_units,
				 FloatGPUMirroredMemoryBlock *output_units,
				 unsigned int size,
				 unsigned int bunch_size);
    virtual void multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
				     FloatGPUMirroredMemoryBlock *output_units,
				     FloatGPUMirroredMemoryBlock *input_errors,
				     FloatGPUMirroredMemoryBlock *output_errors,
				     unsigned int size,
				     unsigned int bunch_size);
  public:
    SoftplusActfANNComponent(const char *name);
    virtual ~SoftplusActfANNComponent();
    virtual ANNComponent *clone();

    virtual char *toLuaString();
  };
}

#endif // SOFTPLUSACTFCOMPONENT_H
