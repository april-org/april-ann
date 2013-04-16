/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2013, Salvador España-Boquera, Adrian Palacios Corella, Francisco
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
#include "bias_component.h"  

void
computeBPUpdateOnPrevVectors(FloatGPUMirroredMemoryBlock *prev_weights_mat_ptr,
			     FloatGPUMirroredMemoryBlock *input,
			     const unsigned int input_shift,
			     FloatGPUMirroredMemoryBlock *input_error,
			     const unsigned int input_error_shift,
			     float beta);

BiasBiasANNComponent::BiasANNComponent(const char *name, const char *weights_name,
				       unsigned int size = 0) :
  ANNComponent(name, weights_name, input_size, output_size), 
  learning_rate(-1.0f), momentum(0.0f), bias_vector(0) { }

BiasANNComponent::~BiasANNComponent() { }

Token *doForward(Token* _input, bool during_training) {
  assert(bias_vector != 0);
  // error checking
  if ( (_input == 0) ||
       (_input->getTokenCode() != table_of_token_codes::token_mem_block))
    ERROR_EXIT(129,"Incorrect input Token type, expected token_mem_block!\n");
  // change current input by new input
  if (input) DecRef(input);
  input = _input->convertTo<TokenMemoryBlock*>();
  IncRef(input);
  // compute current bunch
  unsigned int bunch_size = input->getUsedSize() / input_size;
  this->bunch_size = bunch_size;
  // and resize the output to fit the bunch
  output->resize(bunch_size * output_size);
  // get memory blocks for tokens and weights
  FloatGPUMirroredMemoryBlock *input_ptr       = input->getMemBlock();
  FloatGPUMirroredMemoryBlock *output_ptr      = output->getMemBlock();
  FloatGPUMirroredMemoryBlock *bias_vector_ptr = bias_vector->getPtr();
  doScopyLoop(output_size,
	      bias_vector_ptr, 1,
	      output_ptr, conf.max_bunch_size,
	      conf.cur_bunch_size, 1,
	      conf.use_cuda_flag);

  doSaxpyLoop(output_size, 1.0f,
	      input_ptr, 0, bunch_size,
	      output_ptr, 0, bunch_size,
}

virtual Token *doBackprop(Token *input_error);
    virtual void  *doUpdate();
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
    virtual void copyComponents(hash<string,ANNComponent*> &weights_dict);
    virtual ANNComponent *getComponent(string &name) = 0;
  };
