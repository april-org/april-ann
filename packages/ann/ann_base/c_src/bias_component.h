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
#ifndef BIASANNCOMPONENT_H
#define BIASANNCOMPONENT_H  

#include "cblas_headers.h"
#include "ann_component.h"
#include "connection.h"
#include "token_memory_block.h"

namespace ANN {
  class BiasANNComponent : public ANNComponent {
    TokenMemoryBlock *input, *output, *error;
    Connections *bias_vector;
    unsigned int bunch_size;
    
    /// learning parameters
    float learning_rate, momentum;
    
    void
    computeBPUpdateOnPrevVectors(FloatGPUMirroredMemoryBlock *prev_weights_mat_ptr,
				 FloatGPUMirroredMemoryBlock *input,
				 const unsigned int input_shift,
				 FloatGPUMirroredMemoryBlock *input_error,
				 const unsigned int input_error_shift,
				 float beta);
    
  public:
    BiasANNComponent(const char *name=0, const char *weights_name=0);
    virtual ~BiasANNComponent();
    virtual Token *getInput() { return input; }
    virtual Token *getOutput() { return output; }
    virtual Token *getErrorInput() { return error; }
    virtual Token *getErrorOutput() { return error; }
    virtual Token *doForward(Token* input, bool during_training);
    virtual Token *doBackprop(Token *input_error);
    virtual void   doUpdate();
    virtual void   reset();
    virtual ANNComponent *clone();
    virtual void setOption(const char *name, double value);
    virtual bool hasOption(const char *name);
    virtual double getOption(const char *name);
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);
    virtual void copyWeights(hash<string,Connections*> &weights_dict);
  };
}

#endif // BIASANNCOMPONENT_H
