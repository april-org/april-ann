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
#include "unused_variable.h"
#include "error_print.h"
#include "table_of_token_codes.h"
#include "dropout_component.h"
#include "dropout_kernel.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {
  
  DropoutANNComponent::DropoutANNComponent(MTRand *random,
					   float value,
					   float prob,
					   const char *name,
					   unsigned int size) :
    StochasticANNComponent(random, name, 0, size, size),
    input(0),
    output(0),
    error_input(0),
    error_output(0),
    dropout_mask(0),
    value(value),
    prob(prob),
    size(size) {
  }
  
  DropoutANNComponent::~DropoutANNComponent() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    if (dropout_mask) DecRef(dropout_mask);
  }
  
  Token *DropoutANNComponent::doForward(Token* _input, bool during_training) {
    _input = StochasticANNComponent::doForward(_input, during_training);
    // error checking
    if ( (_input == 0) ||
	 (_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(input,_input->convertTo<TokenMatrixFloat*>());
    if (prob > 0.0f) {
      MatrixFloat *input_mat  = input->getMatrix();
#ifdef USE_CUDA
      input_mat->setUseCuda(use_cuda);
#endif
      // new  output to fit the bunch
      AssignRef(output,new TokenMatrixFloat(input_mat->clone()));
      MatrixFloat *output_mat = output->getMatrix();
      // apply dropout
      if (during_training) {
	if (dropout_mask == 0 || dropout_mask->size() != input_mat->size()) {
	  if (dropout_mask) DecRef(dropout_mask);
	  dropout_mask = new MatrixFloat(1, input_mat->size());
	  IncRef(dropout_mask);
	}
	for (MatrixFloat::iterator it(dropout_mask->begin());
	     it != dropout_mask->end(); ++it) {
	  if (random->rand() < prob) *it = 0.0f;
	  else *it = 1.0f;
	}
	// apply mask
        Kernels::applyDropoutMask(output_mat, dropout_mask, value);
      }
      else {
        matScal(output_mat, 1.0f - prob);
      }
    }
    else AssignRef(output, input);
    return output;
  }

  Token *DropoutANNComponent::doBackprop(Token *_error_input) {
    if (dropout_mask && prob > 0.0f) {
      // error checking
      if ( (_error_input == 0) ||
	   (_error_input->getTokenCode() != table_of_token_codes::token_matrix))
	ERROR_EXIT1(129,"Incorrect input error Token type, expected token_matrix! [%s]\n",
		    name.c_str());
      // change current input by new input
      AssignRef(error_input,_error_input);
      TokenMatrixFloat *error_input_token_matrix = _error_input->convertTo<TokenMatrixFloat*>();
      MatrixFloat *error_input_mat = error_input_token_matrix->getMatrix();
      april_assert(error_input_mat->getNumDim() >= 2);
      if (!error_input_mat->getIsContiguous()) {
	error_input_mat = error_input_mat->clone();
	AssignRef<Token>(error_input,new TokenMatrixFloat(error_input_mat));
      }
#ifdef USE_CUDA
      error_input_mat->setUseCuda(use_cuda);
#endif
      MatrixFloat *error_output_mat = error_input_mat->clone();
      AssignRef<Token>(error_output,new TokenMatrixFloat(error_output_mat));
      if (!error_output_mat->sameDim(input->getMatrix()))
	ERROR_EXIT1(129, "Different bunches found at doForward and doBackprop [%s]\n",
		    name.c_str());
      // apply mask
      Kernels::applyDropoutMask(error_output_mat, dropout_mask, 0.0f);
    }
    else {
      AssignRef(error_input,  _error_input);
      AssignRef(error_output, _error_input);
    }
    return error_output;
  }
  
  void DropoutANNComponent::reset(unsigned int it) {
    StochasticANNComponent::reset(it);
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
  }

  ANNComponent *DropoutANNComponent::clone() {
    DropoutANNComponent *copy_component = new
      DropoutANNComponent(new MTRand(*getRandom()),
			  value, prob,
			  name.c_str(),
			  size);
    return copy_component;
  }

  void DropoutANNComponent::build(unsigned int _input_size,
				  unsigned int _output_size,
				  AprilUtils::LuaTable &weights_dict,
				  AprilUtils::LuaTable &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    if (input_size == 0) input_size = output_size;
    if (output_size == 0) output_size = input_size;
    if (size != 0 && size != input_size)
      ERROR_EXIT(128, "Size given different from size at input/output\n");
    size = input_size;
    if (input_size != output_size)
      ERROR_EXIT3(128, "Incorrect input/output sizes: input=%d output=%d [%s]\n",
		  input_size, output_size, name.c_str());
  }
  
  char *DropoutANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.dropout{ name='%s',size=%d,random=random(),value=%g,prob=%g}",
		  name.c_str(), size, // random->toLuaString(),
		  value, prob);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
