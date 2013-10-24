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
    TokenMatrixFloat *input, *output, *error_input, *error_output;
    // for dropout
    float                        dropout_factor;
    FloatGPUMirroredMemoryBlock *dropout_mask;
    int                         *units_order_permutation;
    static MTRand                dropout_random;
    static int                   dropout_seed;
  protected:
    virtual void applyActivation(FloatGPUMirroredMemoryBlock *input_units,
				 FloatGPUMirroredMemoryBlock *output_units,
				 unsigned int size,
				 unsigned int bunch_size) = 0;
    virtual void multiplyDerivatives(FloatGPUMirroredMemoryBlock *input_units,
				     FloatGPUMirroredMemoryBlock *output_units,
				     FloatGPUMirroredMemoryBlock *input_errors,
				     FloatGPUMirroredMemoryBlock *output_errors,
				     unsigned int size,
				     unsigned int bunch_size) = 0;
  public:
    ActivationFunctionANNComponent(const char *name=0);
    virtual ~ActivationFunctionANNComponent();
    
    virtual Token *getInput() { return input; }
    virtual Token *getOutput() { return output; }
    virtual Token *getErrorInput() { return error_input; }
    virtual Token *getErrorOutput() { return error_output; }
    
    virtual Token *doForward(Token* input, bool during_training);
    
    virtual Token *doBackprop(Token *input_error);

    virtual void reset();
    
    virtual void setOption(const char *name, double value);

    virtual bool hasOption(const char *name);
    
    virtual double getOption(const char *name);
    
    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);

  };
}

#endif // ACTFCOMPONENT_H
