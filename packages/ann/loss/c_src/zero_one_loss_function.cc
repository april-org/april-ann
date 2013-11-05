/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios, Francisco Zamora-Martinez
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
#include "token_matrix.h"
#include "zero_one_loss_function.h"
#include "wrapper.h"
#include "unused_variable.h"

namespace ANN {

  ZeroOneLossFunction::ZeroOneLossFunction(unsigned int size, float TH) :
    LossFunction(size), TH(TH) {
  }
  
  ZeroOneLossFunction::~ZeroOneLossFunction() {
  }
  
  MatrixFloat *ZeroOneLossFunction::computeLossBunch(Token *input,
						     Token *target) {
    MatrixFloat *input_mat, *target_mat;
    throwErrorAndGetMatrixFromTokens(input, target, input_mat, target_mat,
				     false);
    int N = input_mat->getDimSize(1);
    int dim = input_mat->getDimSize(0);
    MatrixFloat *loss_output = new MatrixFloat(1, &dim, CblasColMajor);
    MatrixFloat::iterator loss_output_it(loss_output->begin());
    // Two major cases:
    //    1) A two-class problem.
    //    2) A multi-class problem.
    if (N == 1) {
      // we have two-class problem solved by using one output logistic neuron
      april_assert(target_mat->getDimSize(1) == N);
      MatrixFloat::const_iterator input_it(input_mat->begin());
      MatrixFloat::const_iterator target_it(target_mat->begin());
      while(loss_output_it != loss_output->end()) {
	april_assert(input_it != input_mat->end());
	april_assert(target_it != target_mat->end());
	if (*input_it > TH) {
	  if (*target_it > 0.5f) *loss_output_it = 0.0f;
	  else *loss_output_it = 1.0f;
	}
	else if (*target_it > 0.5f) *loss_output_it = 1.0f;
	else *loss_output_it = 0.0f;
	++loss_output_it;
	++input_it;
	++target_it;
      }
    }
    else {
      // A multi-class problem
      
      // compute the max for every sliding_window
      MatrixFloat::sliding_window input_sw(input_mat);
      MatrixFloat *input_sw_mat  = 0;
      // The target could be:
      //   1) A matrix of the same size of input matrix, so the ARGMAX will be
      //      searched.
      //   2) A uni-dimensional matrix with bunch_size elements, the value of
      //      every element is the label of the class.
      if (target_mat->getDimSize(1) == N) {
	// the ARGMAX will be searched for every pattern in the bunch
	MatrixFloat::sliding_window target_sw(target_mat);
	MatrixFloat *target_sw_mat = 0;
	for (int i=0; i<dim; ++i, ++loss_output_it,
	       input_sw.next(), target_sw.next()) {
	  april_assert(!input_sw.isEnd() && !target_sw.isEnd() &&
		       loss_output_it!=loss_output->end());
	  input_sw_mat  = input_sw.getMatrix(input_sw_mat);
	  target_sw_mat = target_sw.getMatrix(target_sw_mat);
	  float input_max;
	  int input_argmax, input_argmax_rawpos;
	  float target_max;
	  int target_argmax, target_argmax_rawpos;
	  input_max  = input_sw_mat->max(input_argmax, input_argmax_rawpos);
	  target_max = target_sw_mat->max(target_argmax, target_argmax_rawpos);
	  if (input_argmax != target_argmax) *loss_output_it = 1.0f;
	  else *loss_output_it = 0.0f;
	}
	april_assert(target_sw.isEnd());
	delete target_sw_mat;
      }
      else if (target_mat->getDimSize(1) == 1) {
	// the target contains only the class label (starting at 1, instead of 0)
	MatrixFloat::const_iterator target_it(target_mat->begin());
	for (int i=0; i<dim; ++i, ++loss_output_it, ++target_it, input_sw.next()) {
	  april_assert(!input_sw.isEnd() && target_it!=target_mat->end() &&
		       loss_output_it!=loss_output->end());
	  input_sw_mat  = input_sw.getMatrix(input_sw_mat);
	  float input_max;
	  int input_argmax, input_argmax_rawpos;
	  input_max = input_sw_mat->max(input_argmax, input_argmax_rawpos);
	  if (input_argmax != *target_it - 1) *loss_output_it = 1.0f;
	  else *loss_output_it = 0.0f;
	}
	april_assert(target_it==target_mat->end());
      }
      else ERROR_EXIT2(128, "Incorrect target matrix bunch_size, found %d, "
		       "expected %d or 1\n", target_mat->getDimSize(0), dim);
      april_assert(input_sw.isEnd() && loss_output_it==loss_output->end());
      delete input_sw_mat;
    }
    return loss_output;
  }

  Token *ZeroOneLossFunction::computeGradient(Token *input, Token *target) {
    UNUSED_VARIABLE(input);
    UNUSED_VARIABLE(target);
    ERROR_EXIT(128, "NON DIFERENTIABLE LOSS FUNCTION\n");
    return 0;
  }

  char *ZeroOneLossFunction::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.loss.zero_one(%d,%f)", size, TH);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
