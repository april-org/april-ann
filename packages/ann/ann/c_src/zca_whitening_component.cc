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
#include "token_vector.h"
#include "token_matrix.h"
#include "zca_whitening_component.h"
#include "wrapper.h"
#include "utilMatrixFloat.h"

#define WEIGHTS_NAME "U"

namespace ANN {
  
  ZCAWhiteningANNComponent::ZCAWhiteningANNComponent(MatrixFloat *U,
						     MatrixFloat *S,
						     float epsilon,
						     unsigned int takeN,
						     const char *name) :
    PCAWhiteningANNComponent(U,S,epsilon,takeN,name),
    dot_product_decoder(0, WEIGHTS_NAME,
			getOutputSize(), getInputSize(),
			false)
  {
    output_size = input_size;
    matrix_set.insert(WEIGHTS_NAME, this->U);
    hash<string,ANNComponent*> components_dict;
    dot_product_decoder.build(0, 0, &matrix_set, components_dict);
  }
  
  ZCAWhiteningANNComponent::~ZCAWhiteningANNComponent() {
  }
  
  Token *ZCAWhiteningANNComponent::doForward(Token* _input,
					     bool during_training) {
    Token *rotated = PCAWhiteningANNComponent::doForward(_input, during_training);
    return dot_product_decoder.doForward(rotated, during_training);
  }
  
  Token *ZCAWhiteningANNComponent::doBackprop(Token *_error_input) {
    Token *rotated_error = PCAWhiteningANNComponent::doBackprop(_error_input);
    return dot_product_decoder.doBackprop(rotated_error);
  }
  
  ANNComponent *ZCAWhiteningANNComponent::clone() {
    ZCAWhiteningANNComponent *component = new ZCAWhiteningANNComponent(U, S,
								       epsilon,
								       0,
								       name.c_str());
    return component;
  }
  
  char *ZCAWhiteningANNComponent::toLuaString() {
    buffer_list buffer;
    char *U_str, *S_str;
    int len;
    U_str = writeMatrixFloatToString(U, false, len);
    S_str = writeMatrixFloatToString(S, false, len);
    buffer.printf("ann.components.zca_whitening{ name='%s', U=%s, S=%s, epsilon=%g, takeN=0, }",
		  name.c_str(), U_str, S_str, epsilon);
    delete[] U_str;
    delete[] S_str;
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
