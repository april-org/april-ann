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

namespace ANN {

  BiasBiasANNComponent::BiasANNComponent(const char *name,
					 const char *weights_name,
					 unsigned int size = 0) :
    ANNComponent(name, weights_name, size, size), 
    input(0), output(new TokenMemoryBlock()), error(0),
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
    // linear transfer of input to output
    doScopyLoop(output_size,
		input_ptr, 1,
		output_ptr, 1,
		bunch_size, bunch_size,
		GlobalConf::use_cuda);
    // addition of bias vector at output
    doSaxpyLoop(output_size, 1.0f,
		bias_vector_ptr, 1,
		output_ptr, bunch_size,
		bunch_size, bunch_size,
		GlobalConf::use_cuda);
    return output;
  }

  /// In BiasANNComponent this method is a by-pass
  Token *BiasANNComponent::doBackprop(Token *input_error)
  {
    if (error != 0) DecRef(error);
    error = input_error;
    IncRef(error);
    return error;
  }

  void BiasANNComponent::doUpdate() {
    assert(learning_rate > 0.0f &&
	   "Learning rate needs to be fixed with setOption method!!!");
    // Foces bias_vector to update internal counts for a update step
    bias_vector->beginUpdate();
  
    FloatGPUMirroredMemoryBlock *bias_ptr      = bias_vector->getPtr();
    FloatGPUMirroredMemoryBlock *prev_bias_ptr = bias_vector->getPrevPtr();
    FloatGPUMirroredMemoryBlock *input_error   = error->getMemBlock();
  
    // Momentum computation
    if (bias_vector->isFirstUpdateCall()) {
      if (momentum > 0.0f) {
	// prev_w[i,j] = momentum * (w[i,j] - prev_w[i,j])
	bias_vector->computeMomentumOnPrevVector(momentum, GlobalConf::use_cuda);
	bias_vector->computeWeightDecayOnPrevVector(1.0f,  GlobalConf::use_cuda);
      }
      else bias_vector->copyToPrevVector(GlobalConf::use_cuda);
    } // if (bias_vector->needsToComputeMomentum()) {
  
    // update learning rule:
    // PREV_W = alpha * ERRORS + PREV_W
    const unsigned int references = bias_vector->getNumReferences();
    // prev_w[i,j] = -learning_rate*1/sqrt(N*bsize) * ERROR_INPUT[j] + prev_w[i,j]
    const float norm_learn_rate =
      -(1.0f/sqrtf(static_cast<float>(references*conf.cur_bunch_size))) *
      learning_rate;
  
    // bias update: prev_bias[j] = prev_bias[j] + \sum_b norm_learn_rate * ERROR_INPUT[b,j]
    doSaxpyLoop(output_size,
		norm_learn_rate,
		input_error, bunch_size,
		prev_bias_ptr, 1,
		bunch_size, 1,
		GlobalConf::use_cuda);

    // If necessary, update counts, swap vectors, and other stuff
    bias_vector->endUpdate();
  }

  void reset() {
    if (output != 0) doVectorSetToZero(output->getMemBlock(),
				       output->getMaxSize(),
				       0, 0, GlobalConf::use_cuda);
    if (input) DecRef(input); input = 0;
    if (error != 0) DecRef(error); error = 0;
  }

  ANNComponent *BiasANNComponent::clone() {
    ANNComponent *component  = new BiasANNComponent(name,weights_name,input_size);
    component->learning_rate = learning_rate;
    component->momentum      = momentum;
    return component;
  }

  void BiasANNComponent::setOption(const char *name, double value) {
    mSetOption("learning_rate", learning_rate);
    mSetOption("momentum", momentum);
  }

  bool BiasANNComponent::hasOption(const char *name) {
    mHasOption("learning_rate");
    mHasOption("momentum");
    return false;
  }

  double BiasANNComponent::getOption(const char *name) {
    mGetOption("learning_rate", learning_rate);
    mGetOption("momentum", momentum);
    ERROR_EXIT(140, "The option to be get does not exist.\n");
  }

  void BiasANNComponent::build(unsigned int input_size,
			       unsigned int output_size,
			       hash<string,Connections*> &weights_dict,
			       hash<string,ANNComponent*> &components_dict) {
    unsigned int weights_input_size  = 1;
    unsigned int weights_output_size = output_size;
    ////////////////////////////////////////////////////////////////////
    if (input_size == 0)  this->input_size  = input_size;
    if (output_size == 0) this->output_size = output_size;
    if (this->input_size != input_size)
      ERROR_EXIT2(129, "Incorrect input size, expected %d, found %d\n",
		  this->input_size, input_size);
    if (this->output_size != output_size)
      ERROR_EXIT2(129, "Incorrect output size, expected %d, found %d\n",
		  this->output_size, output_size);
    ////////////////////////////////////////////////////////////////////
    ANNComponent *&component = components_dict[name];
    if (component != 0) ERROR_EXIT(102, "Non unique component name found: %s\n",
				   name.c_str());
    component = this;
    ////////////////////////////////////////////////////////////////////
    if (bias_vector != 0) DecRef(bias_vector);
    Connections *&w = weights_dict[weights_name];
    if (w != 0) {
      bias_vector = w;
      if (!bias_vector->checkInputOutputSizes(weights_input_size,
					      weights_output_size))
	ERROR_EXIT2(256,"The weights matrix input/output sizes are not correct, "
		    "expected %d,%d.\n",
		    weights_input_size, weights_output_size);
    }
    else {
      bias_vector = new Connections(weights_input_size,
				    weights_output_size);
      w = bias_vector;
    }
    // TODO: compute fan-in
    // outputs->increaseFanIn(inputs->numNeurons());
    bias_vector->countReference();
    IncRef(bias_vector);  
  }

  void copyWeights(hash<string,Connections*> &weights_dict) {
    if (bias_vector == 0)
      ERROR_EXIT(100, "Component not built, impossible execute copyWeights\n");
    Connections *&w = weights_dict[weights_name];
    if (w != 0 && w != bias_vector)
      ERROR_EXIT1(101, "Weights dictionary contains %s weights name which is "
		  "not shared with bias_vector attribute\n",
		  weights_name.c_str());
    else if (w == 0) w = bias_vector;
  }
}
