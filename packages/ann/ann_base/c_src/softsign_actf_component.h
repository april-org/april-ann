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
#ifndef SOFTSIGNACTFCOMPONENT_H
#define SOFTSIGNACTFCOMPONENT_H

#include "activation_function_component.h"
#include "token_memory_block.h"
#include "ann_component.h"
#include "gpu_mirrored_memory_block.h"

namespace ANN {

  /// An abstract class that defines the basic interface that
  /// the anncomponents must fulfill.
  class SoftsignActfANNComponent : public ActivationFunctionANNComponent {
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
    SoftsignActfANNComponent(const char *name);
    virtual ~SoftsignActfANNComponent();
    virtual ANNComponent *clone();

    virtual char *toLuaString();
  };
}

#endif // SOFTSIGNACTFCOMPONENT_H
