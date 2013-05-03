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
#include "wrapper.h"

namespace ANN {

  BiasANNComponent::BiasANNComponent(const char *name,
				     const char *weights_name) :
    ANNComponent(name, weights_name, 0, 0),
    input(0), output(0), error(0),
    bias_vector(0), learning_rate(-1.0f), momentum(0.0f) {
    if (weights_name == 0) generateDefaultWeightsName();
  }

  BiasANNComponent::~BiasANNComponent() {
    if (bias_vector) DecRef(bias_vector);
    if (input) DecRef(input);
    if (error) DecRef(error);
    if (output) DecRef(output);
  }

  Token *BiasANNComponent::doForward(Token* _input, bool during_training) {
    assert(bias_vector != 0);
    // error checking
    if ( (_input == 0) ||
	 (_input->getTokenCode() != table_of_token_codes::token_mem_block))
      ERROR_EXIT(129,"Incorrect input Token type, expected token_mem_block!\n");
    // change current input by new input
    AssignRef(input,_input->convertTo<TokenMemoryBlock*>());
    // compute current bunch
    unsigned int bunch_size = input->getUsedSize() / input_size;
    if (input->getUsedSize() % input_size != 0)
      ERROR_EXIT2(128, "Input memory block (size %d) is not multiple of %d\n",
		  input->getUsedSize(), input_size);
    this->bunch_size = bunch_size;
    // new output to fit the bunch
    AssignRef(output,new TokenMemoryBlock(input->getUsedSize()));
    // get memory blocks for tokens and weights
    FloatGPUMirroredMemoryBlock *input_ptr       = input->getMemBlock();
    FloatGPUMirroredMemoryBlock *output_ptr      = output->getMemBlock();
    FloatGPUMirroredMemoryBlock *bias_vector_ptr = bias_vector->getPtr();
    // linear transfer of input to output
    doScopy(output_size*bunch_size,
	    input_ptr, 0, 1,
	    output_ptr, 0, 1,
	    use_cuda);
    // addition of bias vector at output
    doSaxpyLoop(output_size, 1.0f,
		bias_vector_ptr, 1,
		output_ptr, bunch_size,
		bunch_size,
		0, 1,
		use_cuda);
    return output;
  }

  /// In BiasANNComponent this method is a by-pass
  Token *BiasANNComponent::doBackprop(Token *_error_input)
  {
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_mem_block))
      ERROR_EXIT(129,"Incorrect input error Token type, expected token_mem_block!\n");
    // change current input by new input
    AssignRef(error,_error_input->convertTo<TokenMemoryBlock*>());
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
	bias_vector->computeMomentumOnPrevVector(momentum, use_cuda);
	bias_vector->computeWeightDecayOnPrevVector(1.0f,  use_cuda);
      }
      else bias_vector->copyToPrevVector(use_cuda);
    } // if (bias_vector->needsToComputeMomentum()) {
  
    // update learning rule:
    // PREV_W = alpha * ERRORS + PREV_W
    const unsigned int references = bias_vector->getNumReferences();
    // prev_w[i,j] = -learning_rate*1/sqrt(N*bsize) * ERROR_INPUT[j] + prev_w[i,j]
    const float norm_learn_rate =
      -(1.0f/sqrtf(static_cast<float>(references*bunch_size))) *
      learning_rate;
  
    // bias update: prev_bias[j] = prev_bias[j] + \sum_b norm_learn_rate * ERROR_INPUT[b,j]
    doSaxpyLoop(output_size,
		norm_learn_rate,
		input_error, bunch_size,
		prev_bias_ptr, 1,
		bunch_size,
		1, 0,
		use_cuda);

    // If necessary, update counts, swap vectors, and other stuff
    bias_vector->endUpdate();
  }

  void BiasANNComponent::reset() {
    if (input)  DecRef(input);
    if (error)  DecRef(error);
    if (output) DecRef(output);
    input  = 0;
    error  = 0;
    output = 0;
  }

  ANNComponent *BiasANNComponent::clone() {
    BiasANNComponent *component = new BiasANNComponent(name.c_str(),
						       weights_name.c_str());
    component->input_size    = input_size;
    component->output_size   = output_size;
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
    return ANNComponent::getOption(name);
  }

  void BiasANNComponent::build(unsigned int _input_size,
			       unsigned int _output_size,
			       hash<string,Connections*> &weights_dict,
			       hash<string,ANNComponent*> &components_dict) {
    bool rebuild = (getIsBuilt() == true);
    ANNComponent::build(_input_size, _output_size, weights_dict, components_dict);
    //
    if (input_size == 0 || output_size == 0)
      ERROR_EXIT(141, "Impossible to compute input/output "
		 "sizes for this component\n");
    if (input_size != output_size)
      ERROR_EXIT(142, "BiasANNComponent input/output sizes must be equal\n");
    unsigned int weights_input_size  = 1;
    unsigned int weights_output_size = output_size;
    ////////////////////////////////////////////////////////////////////
    Connections *&w = weights_dict[weights_name];
    if (w != 0) {
      bias_vector = w;
      if (!bias_vector->checkInputOutputSizes(weights_input_size,
					      weights_output_size))
	ERROR_EXIT2(256,"The weights matrix input/output sizes are not correct, "
		    "expected %d inputs and %d outputs.\n",
		    weights_input_size, weights_output_size);
      IncRef(bias_vector);
    }
    else {
      if (bias_vector == 0) {
	bias_vector = new Connections(weights_input_size,
				      weights_output_size);
	IncRef(bias_vector);
      }
      w = bias_vector;
    }
    if (!rebuild) bias_vector->countReference();
  }

  void BiasANNComponent::copyWeights(hash<string,Connections*> &weights_dict) {
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
