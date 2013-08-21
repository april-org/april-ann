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
#include "error_print.h"
#include "table_of_token_codes.h"
#include "salt_and_pepper_component.h"
#include "wrapper.h"

namespace ANN {
  
  SaltAndPepperANNComponent::SaltAndPepperANNComponent(MTRand *random,
						       float zero,
						       float one,
						       float prob,
						       const char *name,
						       unsigned int size) :
    ANNComponent(name, 0, size, size),
    random(random),
    input(0),
    output(0),
    error_input(0),
    error_output(0),
    zero(zero),
    one(one),
    prob(prob) {
    IncRef(random);
  }
  
  SaltAndPepperANNComponent::~SaltAndPepperANNComponent() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    DecRef(random);
  }
  
  Token *SaltAndPepperANNComponent::doForward(Token* _input, bool during_training) {
    // error checking
    if ( (_input == 0) ||
	 (_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(input,_input->convertTo<TokenMatrixFloat*>());
#ifdef USE_CUDA
    input->getMatrix()->setUseCuda(use_cuda);
#endif
    // new  output to fit the bunch
    AssignRef(output,new TokenMatrixFloat(input->getMatrix()->clone()));
    MatrixFloat *output_mat = output->getMatrix();
    april_assert(output_mat->getMajorOrder() == CblasColMajor);
    for (MatrixFloat::col_major_iterator it(output_mat->begin());
	 it != output_mat->end();
	 ++it) {
      float p = random->rand();
      if (p < prob) {
	if (p < prob * 0.5f) *it = zero;
	else *it = one;
      }
    }
#ifdef USE_CUDA
    output->getMatrix()->setUseCuda(use_cuda);
#endif
    return output;
  }

  Token *SaltAndPepperANNComponent::doBackprop(Token *_error_input) {
    AssignRef(error_input,  _error_input);
    AssignRef(error_output, _error_input);
    return error_output;
  }
  
  void SaltAndPepperANNComponent::reset() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
  }

  ANNComponent *SaltAndPepperANNComponent::clone() {
    SaltAndPepperANNComponent *copy_component = new
      SaltAndPepperANNComponent(new MTRand(*random),
				zero, one, prob,
				name.c_str(),
				input_size);
    return copy_component;
  }

  void SaltAndPepperANNComponent::build(unsigned int _input_size,
					unsigned int _output_size,
					hash<string,Connections*> &weights_dict,
					hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    if (output_size == 0) output_size = input_size;
    if (input_size  == 0) input_size  = output_size;
    if (input_size != output_size)
      ERROR_EXIT3(128, "Incorrect input/output sizes: input=%d output=%d [%s]\n",
		  input_size, output_size, name.c_str());
  }
  
  char *SaltAndPepperANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.salt_and_pepper{ name='%s',size=%d }",
		  name.c_str(), input_size);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
