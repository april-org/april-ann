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
#include "unused_variable.h"
#include "bias_component.h"  
#include "wrapper.h"
#include "unused_variable.h"

namespace ANN {

  BiasANNComponent::BiasANNComponent(unsigned int size,
				     const char *name,
				     const char *weights_name) :
    ANNComponent(name, weights_name, size, size),
    input(0), output(0), error(0),
    bias_vector(0) {
    if (weights_name == 0) generateDefaultWeightsName(this->weights_name, "b");
  }

  BiasANNComponent::~BiasANNComponent() {
    if (bias_vector) DecRef(bias_vector);
    if (input) DecRef(input);
    if (error) DecRef(error);
    if (output) DecRef(output);
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
    MatrixFloat *bias_ptr = bias_vector;
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
    //
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

  void BiasANNComponent::reset(unsigned int it) {
    UNUSED_VARIABLE(it);
    if (input)  DecRef(input);
    if (error)  DecRef(error);
    if (output) DecRef(output);
    input  = 0;
    error  = 0;
    output = 0;
    // reset shared counter
    bias_vector->resetSharedCount();
  }

  void BiasANNComponent::computeGradients(MatrixFloat*& grads_mat) {
    // count one use of the vector
    bias_vector->addToSharedCount();
    if (grads_mat == 0) {
      grads_mat = bias_vector->cloneOnlyDims();
      grads_mat->zeros();
      IncRef(grads_mat);
    }
    else if (!grads_mat->sameDim(bias_vector))
      ERROR_EXIT(128, "Incorrect weights matrix dimensions\n");
    MatrixFloat *input_error_mat = error->getMatrix();
    unsigned int bunch_size = input_error_mat->getDimSize(0);
    // bias update: prev_bias[j] = prev_bias[j] + \sum_b norm_learn_rate * ERROR_INPUT[b,j]
    if (bunch_size == 1) grads_mat->axpy(1.0f, input_error_mat);
    else doAxpyLoop(output_size,
		    1.0f,
		    input_error_mat->getRawDataAccess(),
		    input_error_mat->getStrideSize(1),
		    0,
		    grads_mat->getRawDataAccess(),
		    grads_mat->getStrideSize(0),
		    0,
		    bunch_size,
		    input_error_mat->getStrideSize(0), 0,
		    use_cuda);
  }

  ANNComponent *BiasANNComponent::clone() {
    BiasANNComponent *component = new BiasANNComponent(input_size,
						       name.c_str(),
						       weights_name.c_str());
    return component;
  }

  void BiasANNComponent::build(unsigned int _input_size,
			       unsigned int _output_size,
			       MatrixFloatSet *weights_dict,
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
    MatrixFloat *&w = (*weights_dict)[weights_name].getDense();
    // printf("%s :: %p %p\n", weights_name.c_str(), w, bias_vector);
    if (w != 0) {
      AssignRef(bias_vector, w);
      // printf("COPY OF BIAS FROM HASH %s\n", weights_name.c_str());
      if (!Connections::checkInputOutputSizes(bias_vector,
					      weights_input_size,
					      weights_output_size))
	ERROR_EXIT3(256,"The weights matrix input/output sizes are not correct, "
		    "expected %d inputs and %d outputs. [%s]\n",
		    weights_input_size, weights_output_size,
		    name.c_str());
    }
    else {
      if (bias_vector == 0) {
	// printf("NEW OF BIAS %s\n", weights_name.c_str());
	bias_vector = Connections::build(weights_input_size,
					 weights_output_size);
	IncRef(bias_vector);
      }
      // else printf("USING PREVIOUS BIAS %s\n", weights_name.c_str());
      w = bias_vector;
      IncRef(w);
    }
  }

  void BiasANNComponent::copyWeights(MatrixFloatSet *weights_dict) {
    if (bias_vector == 0)
      ERROR_EXIT1(100, "Component not built, impossible execute copyWeights [%s]\n",
		  name.c_str());
    MatrixFloat *&w = (*weights_dict)[weights_name].getDense();
    if (w != 0 && w != bias_vector)
      ERROR_EXIT2(101, "Weights dictionary contains %s weights name which is "
		  "not shared with bias_vector attribute [%s]\n",
		  weights_name.c_str(),
		  name.c_str());
    else if (w == 0) {
      w = bias_vector;
      IncRef(w);
    }
  }

  char *BiasANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.bias{ size=%d, name='%s', weights='%s' }",
		  input_size, name.c_str(), weights_name.c_str());
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
