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
#ifndef HARDTANHACTFCOMPONENT_H
#define HARDTANHACTFCOMPONENT_H

#include "activation_function_component.h"
#include "ann_component.h"
#include "gpu_mirrored_memory_block.h"

namespace ANN {

  /// Component which computes the HardTanh activation function. It is useful
  /// for deep ANNs and CNNs.
  class HardtanhActfANNComponent : public ActivationFunctionANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(HardtanhActfANNComponent);

    float inf, sup;
  protected:
    virtual void applyActivation(april_math::FloatGPUMirroredMemoryBlock *input_units,
				 april_math::FloatGPUMirroredMemoryBlock *output_units,
				 unsigned int size,
				 unsigned int bunch_size);
    virtual void multiplyDerivatives(april_math::FloatGPUMirroredMemoryBlock *input_units,
				     april_math::FloatGPUMirroredMemoryBlock *output_units,
				     april_math::FloatGPUMirroredMemoryBlock *input_errors,
				     april_math::FloatGPUMirroredMemoryBlock *output_errors,
				     unsigned int size,
				     unsigned int bunch_size);
  public:
    HardtanhActfANNComponent(const char *name, float inf=-1.0f, float sup=1.0f);
    virtual ~HardtanhActfANNComponent();
    virtual ANNComponent *clone();

    virtual char *toLuaString();
  };
}

#endif // HARDTANHACTFCOMPONENT_H
