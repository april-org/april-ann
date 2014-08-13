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
#ifndef ACTFCOMPONENT_H
#define ACTFCOMPONENT_H

#include "token_matrix.h"
#include "ann_component.h"
#include "gpu_mirrored_memory_block.h"
#include "MersenneTwister.h"

namespace ANN {

  /// An abstract class that defines the basic interface that
  /// the anncomponents must fulfill.
  class ActivationFunctionANNComponent : public ANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(ActivationFunctionANNComponent);
    basics::TokenMatrixFloat *input, *output, *error_input, *error_output;
  protected:
    virtual void applyActivation(april_math::FloatGPUMirroredMemoryBlock *input_units,
				 april_math::FloatGPUMirroredMemoryBlock *output_units,
				 unsigned int size,
				 unsigned int bunch_size) = 0;
    virtual void multiplyDerivatives(april_math::FloatGPUMirroredMemoryBlock *input_units,
				     april_math::FloatGPUMirroredMemoryBlock *output_units,
				     april_math::FloatGPUMirroredMemoryBlock *input_errors,
				     april_math::FloatGPUMirroredMemoryBlock *output_errors,
				     unsigned int size,
				     unsigned int bunch_size) = 0;
  public:
    ActivationFunctionANNComponent(const char *name=0);
    virtual ~ActivationFunctionANNComponent();
    
    virtual basics::Token *getInput() { return input; }
    virtual basics::Token *getOutput() { return output; }
    virtual basics::Token *getErrorInput() { return error_input; }
    virtual basics::Token *getErrorOutput() { return error_output; }
    
    virtual basics::Token *doForward(basics::Token* input, bool during_training);
    
    virtual basics::Token *doBackprop(basics::Token *input_error);

    virtual void reset(unsigned int it=0);
    
    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       basics::MatrixFloatSet *weights_dict,
		       april_utils::hash<april_utils::string,ANNComponent*> &components_dict);

  };
}

#endif // ACTFCOMPONENT_H
