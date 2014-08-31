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
    Basics::TokenMatrixFloat *input, *output, *error_input, *error_output;
  protected:
    virtual void applyActivation(AprilMath::FloatGPUMirroredMemoryBlock *input_units,
				 AprilMath::FloatGPUMirroredMemoryBlock *output_units,
				 unsigned int size,
				 unsigned int bunch_size) = 0;
    virtual void multiplyDerivatives(AprilMath::FloatGPUMirroredMemoryBlock *input_units,
				     AprilMath::FloatGPUMirroredMemoryBlock *output_units,
				     AprilMath::FloatGPUMirroredMemoryBlock *input_errors,
				     AprilMath::FloatGPUMirroredMemoryBlock *output_errors,
				     unsigned int size,
				     unsigned int bunch_size) = 0;
  public:
    ActivationFunctionANNComponent(const char *name=0);
    virtual ~ActivationFunctionANNComponent();
    
    virtual Basics::Token *getInput() { return input; }
    virtual Basics::Token *getOutput() { return output; }
    virtual Basics::Token *getErrorInput() { return error_input; }
    virtual Basics::Token *getErrorOutput() { return error_output; }
    
    virtual Basics::Token *doForward(Basics::Token* input, bool during_training);
    
    virtual Basics::Token *doBackprop(Basics::Token *input_error);

    virtual void reset(unsigned int it=0);
    
    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       Basics::MatrixFloatSet *weights_dict,
		       AprilUtils::hash<AprilUtils::string,ANNComponent*> &components_dict);

  };
}

#endif // ACTFCOMPONENT_H
