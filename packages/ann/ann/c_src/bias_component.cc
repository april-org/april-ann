/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
#include "unused_variable.h"

namespace ANN {

  BiasANNComponent::BiasANNComponent(unsigned int size,
				     const char *name,
				     const char *weights_name) :
    ANNComponent(name, weights_name, size, size),
    input(0), output(0), error(0),
    bias_vector(0), num_updates_from_last_prune(0),
    learning_rate(-1.0f), momentum(0.0f) {
    if (weights_name == 0) generateDefaultWeightsName(this->weights_name, "b");
  }

  BiasANNComponent::~BiasANNComponent() {
    if (bias_vector) DecRef(bias_vector);
    if (input) DecRef(input);
    if (error) DecRef(error);
    if (output) DecRef(output);
  }

  void BiasANNComponent::computeBP(MatrixFloat *weights_mat,
				   MatrixFloat *input_error_mat,
				   float alpha) {
    unsigned int bunch_size = input_error_mat->getDimSize(0);
    // bias update: prev_bias[j] = prev_bias[j] + \sum_b norm_learn_rate * ERROR_INPUT[b,j]
    if (bunch_size == 1) weights_mat->axpy(alpha, input_error_mat);
    else doAxpyLoop(output_size,
		    alpha,
		    input_error_mat->getRawDataAccess(),
		    input_error_mat->getStrideSize(1),
		    0,
		    weights_mat->getRawDataAccess(),
		    weights_mat->getStrideSize(0),
		    0,
		    bunch_size,
		    input_error_mat->getStrideSize(0), 0,
		    use_cuda);
  }

  Token *BiasANNComponent::doForward(Token* _input, bool during_training) {
    UNUSED_VARIABLE(during_training);
    if (bias_vector == 0) ERROR_EXIT1(129, "Not built component %s\n",
				      name.c_str());
    // error checking
    if ( (_input == 0) ||
	 (_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(input,_input->convertTo<TokenMatrixFloat*>());
    MatrixFloat *input_mat = input->getMatrix();
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    ASSERT_MATRIX(input_mat);
    april_assert(input_mat->getDimSize(1) == static_cast<int>(input_size));
    unsigned int bunch_size = input_mat->getDimSize(0);
    // linear transfer of input to output
    MatrixFloat *output_mat = input_mat->clone();
    AssignRef(output,new TokenMatrixFloat(output_mat));
    // bias
    MatrixFloat *bias_ptr = bias_vector->getPtr();
    if (bunch_size == 1) output_mat->axpy(1.0f, bias_ptr);
    else {
      // addition of bias vector at output
      doAxpyLoop(output_size, 1.0f,
		 bias_ptr->getRawDataAccess(), bias_ptr->getStrideSize(0), 0,
		 output_mat->getRawDataAccess(), output_mat->getStrideSize(1), 0,
		 bunch_size,
		 0, output_mat->getStrideSize(0),
		 use_cuda);
    }
    return output;
  }

  /// In BiasANNComponent this method is a by-pass
  Token *BiasANNComponent::doBackprop(Token *_error_input)
  {
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input error Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(error,_error_input->convertTo<TokenMatrixFloat*>());
#ifdef USE_CUDA
    error->getMatrix()->setUseCuda(use_cuda);
#endif
    return error;
  }

  void BiasANNComponent::doUpdate() {
    april_assert(learning_rate > 0.0f &&
	   "Learning rate needs to be fixed with setOption method!!!");
    // Foces bias_vector to update internal counts for a update step
    bias_vector->beginUpdate();
  
    MatrixFloat *bias_ptr        = bias_vector->getPtr();
    MatrixFloat *prev_bias_ptr   = bias_vector->getPrevPtr();
    MatrixFloat *input_error_mat = error->getMatrix();
    ASSERT_MATRIX(input_error_mat);
    april_assert(input_error_mat->getDimSize(1) == static_cast<int>(input_size));
    unsigned int bunch_size = input_error_mat->getDimSize(0);
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
    april_assert(references > 0 && "Found 0 references of bias vector");
    // prev_w[i,j] = -learning_rate*1/sqrt(N*bsize) * ERROR_INPUT[j] + prev_w[i,j]
    const float norm_learn_rate =
      -(1.0f/sqrtf(static_cast<float>(references*bunch_size))) *
      learning_rate;

    computeBP(prev_bias_ptr, input_error_mat, norm_learn_rate);
    
    // If necessary, update counts, swap vectors, and other stuff
    if (bias_vector->endUpdate()) {
      ++num_updates_from_last_prune;
      if (num_updates_from_last_prune > MAX_UPDATES_WITHOUT_PRUNE) {
	num_updates_from_last_prune = 0;
	bias_vector->pruneSubnormalAndCheckNormal();
      }
    }
  }
  
  void BiasANNComponent::reset() {
    if (input)  DecRef(input);
    if (error)  DecRef(error);
    if (output) DecRef(output);
    input  = 0;
    error  = 0;
    output = 0;
  }

  void BiasANNComponent::computeGradients(MatrixFloat*& weight_grads) {
    if (weight_grads == 0) {
      weight_grads = bias_vector->getPtr()->cloneOnlyDims();
      weight_grads->zeros();
    }
    else if (!weight_grads->sameDim(bias_vector->getPtr()))
      ERROR_EXIT(128, "Incorrect weights matrix dimensions\n");
    MatrixFloat *input_error_mat = error->getMatrix();
    unsigned int bunch_size = input_error_mat->getDimSize(0);
    computeBP(weight_grads, error->getMatrix(), 1.0f/bunch_size);
  }

  ANNComponent *BiasANNComponent::clone() {
    BiasANNComponent *component = new BiasANNComponent(input_size,
						       name.c_str(),
						       weights_name.c_str());
    component->learning_rate = learning_rate;
    component->momentum      = momentum;
    return component;
  }

  void BiasANNComponent::setOption(const char *name, double value) {
    mSetOption(LEARNING_RATE_STRING, learning_rate);
    mSetOption(MOMENTUM_STRING,      momentum);
    ANNComponent::setOption(name, value);
  }

  bool BiasANNComponent::hasOption(const char *name) {
    mHasOption(LEARNING_RATE_STRING);
    mHasOption(MOMENTUM_STRING);
    return false;
  }

  double BiasANNComponent::getOption(const char *name) {
    mGetOption(LEARNING_RATE_STRING, learning_rate);
    mGetOption(MOMENTUM_STRING,      momentum);
    return ANNComponent::getOption(name);
  }

  void BiasANNComponent::build(unsigned int _input_size,
			       unsigned int _output_size,
			       hash<string,Connections*> &weights_dict,
			       hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    //
    if (input_size == 0 || output_size == 0)
      ERROR_EXIT1(141, "Impossible to compute input/output "
		  "sizes for this component [%s]\n",
		  name.c_str());
    if (input_size != output_size)
      ERROR_EXIT1(142, "BiasANNComponent input/output sizes must be equal [%s]\n",
		  name.c_str());
    unsigned int weights_input_size  = 1;
    unsigned int weights_output_size = output_size;
    ////////////////////////////////////////////////////////////////////
    Connections *&w = weights_dict[weights_name];
    // printf("%s :: %p %p\n", weights_name.c_str(), w, bias_vector);
    if (w != 0) {
      AssignRef(bias_vector, w);
      // printf("COPY OF BIAS FROM HASH %s\n", weights_name.c_str());
      if (!bias_vector->checkInputOutputSizes(weights_input_size,
					      weights_output_size))
	ERROR_EXIT3(256,"The weights matrix input/output sizes are not correct, "
		    "expected %d inputs and %d outputs. [%s]\n",
		    weights_input_size, weights_output_size,
		    name.c_str());
    }
    else {
      if (bias_vector == 0) {
	// printf("NEW OF BIAS %s\n", weights_name.c_str());
	bias_vector = new Connections(weights_input_size,
				      weights_output_size);
	IncRef(bias_vector);
      }
      // else printf("USING PREVIOUS BIAS %s\n", weights_name.c_str());
      w = bias_vector;
    }
    bias_vector->countReference();
  }

  void BiasANNComponent::copyWeights(hash<string,Connections*> &weights_dict) {
    if (bias_vector == 0)
      ERROR_EXIT1(100, "Component not built, impossible execute copyWeights [%s]\n",
		  name.c_str());
    Connections *&w = weights_dict[weights_name];
    if (w != 0 && w != bias_vector)
      ERROR_EXIT2(101, "Weights dictionary contains %s weights name which is "
		  "not shared with bias_vector attribute [%s]\n",
		  weights_name.c_str(),
		  name.c_str());
    else if (w == 0) w = bias_vector;
  }

  char *BiasANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.bias{ size=%d, name='%s', weights='%s' }",
		  input_size, name.c_str(), weights_name.c_str());
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
