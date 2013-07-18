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
#include "swap.h"
#include "maxpooling_component.h"
#include "token_matrix.h"
#include "table_of_token_codes.h"

using april_utils::swap;

namespace ANN {

  ///////////////////////////////////////////
  // MaxPoolingANNComponent implementation //
  ///////////////////////////////////////////

  void MaxPoolingANNComponent::initializeArrays(const int *input_dims) {
    // input_dims[0]  => BUNCH SIZE
    // output_dims[0] => BUNCH SIZE, output_dims[1] => HIDDEN LAYER SIZE
    output_dims[0]	       = input_dims[0];
    input_window_size[0]       = input_dims[0];
    input_window_num_steps[0]  = 1;
    output_window_size[0]      = input_dims[0];
    output_window_num_steps[0] = 1;
    input_window_rewrap[0]     = input_dims[0];
    output_window_rewrap[0]    = input_dims[0];
    for (int i=1; i<=input_num_dims; ++i) {
      if (kernel_dims[i] == 0) {
	output_dims[i]             = 1;
	input_window_size[i]       = input_dims[i];
	input_window_num_steps[i]  = output_dims[i];
	output_window_num_steps[i] = output_dims[i];
      }
      else {
	output_dims[i] = (input_dims[i] - kernel_dims[i])/kernel_step[i] + 1;
	input_window_size[i]       = kernel_dims[i];
	input_window_num_steps[i]  = output_dims[i];
	output_window_num_steps[i] = output_dims[i];
      }
    }
  }
  
  MaxPoolingANNComponent::MaxPoolingANNComponent(int input_num_dims,
						 const int *_kernel_dims,
						 const int *_kernel_step,
						 const char *name) :
    ANNComponent(name, 0, 0, 0),
    input(0),
    error_input(0),
    output(0),
    error_output(0),
    argmax_raw_pos(0),
    number_input_windows(0),
    kernel_size(1),
    input_num_dims(input_num_dims),
    kernel_dims(new int[input_num_dims+1]),
    kernel_step(new int[input_num_dims+1]),
    output_dims(new int[input_num_dims+1]),
    input_window_size(new int[input_num_dims+1]),
    input_window_num_steps(new int[input_num_dims+1]),
    input_window_order_step(new int[input_num_dims+1]),
    input_window_rewrap(new int[2]),
    output_window_size(new int[input_num_dims+1]),
    output_window_step(new int[input_num_dims+1]),
    output_window_num_steps(new int[input_num_dims+1]),
    output_window_order_step(new int[input_num_dims+1]),
    output_window_rewrap(new int[2]) {
    //
    input_window_order_step[0] = 0;
    output_window_size[0] = 0;
    output_window_order_step[0] = 0;
    output_window_step[0] = 1;
    kernel_step[0] = 1;
    kernel_dims[0] = 1;
    for(int i=0; i<input_num_dims; ++i) {
      kernel_size *= _kernel_dims[i];
      kernel_dims[i+1] = _kernel_dims[i];
      kernel_step[i+1] = _kernel_step[i];
      input_window_order_step[i+1] = i+1;
      output_window_size[i+1] = 1;
      output_window_order_step[i+1] = i+1;
      output_window_step[i+1] = 1;
    }
    input_window_rewrap[0]  = 0;
    input_window_rewrap[1]  = static_cast<int>(kernel_size);
    output_window_rewrap[0] = 0;
    output_window_rewrap[1] = 1;
  }
  
  MaxPoolingANNComponent::~MaxPoolingANNComponent() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
    delete[] kernel_dims;
    delete[] kernel_step;
    delete[] output_dims;
    delete[] input_window_size;
    delete[] input_window_num_steps;
    delete[] input_window_order_step;
    delete[] output_window_size;
    delete[] output_window_step;
    delete[] output_window_num_steps;
    delete[] output_window_order_step;
    delete[] input_window_rewrap;
    delete[] output_window_rewrap;
  }
  
  // The MaxPoolingANNComponent
  Token *MaxPoolingANNComponent::doForward(Token *_input, bool during_training) {
    // error checking
    if (_input == 0) ERROR_EXIT1(129,"Null Token received! [%s]\n",
				 name.c_str());
    if (_input->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT1(129, "Incorrect token received, expected token_matrix [%s]\n",
		  name.c_str());
    AssignRef(input, _input->convertTo<TokenMatrixFloat*>());
    MatrixFloat *input_mat=input->getMatrix();
    if (input_mat->getNumDim() != input_num_dims+1)
      ERROR_EXIT3(129, "Incorrect input matrix numDims, "
		  "expected %d, found %d [%s]\n", input_num_dims+1,
		  input_mat->getNumDim(), name.c_str());
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    if (!input_mat->getIsContiguous()) {
      input_mat = input_mat->clone();
      AssignRef(input,new TokenMatrixFloat(input_mat));
    }
    const int *input_dims = input_mat->getDimPtr();
    initializeArrays(input_dims);
    MatrixFloat *output_mat;
    output_mat = new MatrixFloat(input_num_dims+1, output_dims, CblasColMajor);
    AssignRef(output, new TokenMatrixFloat(output_mat));
#ifdef USE_CUDA
    output_mat->setUseCuda(use_cuda);
#endif
    
    /////////////////////////////////////////////////////////////////////////
    
    // Prepare sliding windows to compute the convolution
    MatrixFloat::sliding_window input_sw(input_mat, input_window_size,
					 0,  // OFFSET
					 kernel_step,
					 input_window_num_steps,
					 input_window_order_step);
    MatrixFloat::sliding_window output_sw(output_mat, output_window_size,
					  0,  // OFFSET
					  output_window_step,
					  output_window_num_steps,
					  output_window_order_step);
    number_input_windows = input_sw.numWindows();
    assert(number_input_windows == output_sw.numWindows());
    if (during_training)
      AssignRef(argmax_raw_pos,
		new IntGPUMirroredMemoryBlock(input_mat->getDimSize(0)*
					      output_sw.numWindows()));
    else if (argmax_raw_pos) {
      DecRef(argmax_raw_pos);
      argmax_raw_pos = 0;
    }
    int k=0;
    // CONVOLUTION OVER number_input_windows
    while(!input_sw.isEnd() && !output_sw.isEnd()) {
      MatrixFloat *input_w  = input_sw.getMatrix();
      MatrixFloat *output_w = output_sw.getMatrix();
      IncRef(input_w);
      IncRef(output_w);
      MatrixFloat *max_sel_dim = input_w->maxSelDim(0, argmax_raw_pos, k);
      IncRef(max_sel_dim);
      MatrixFloat *max_sel_dim_rewrapped;
      max_sel_dim_rewrapped = max_sel_dim->rewrap(output_w->getDimPtr(),
						  output_w->getNumDim());
      IncRef(max_sel_dim_rewrapped);
      output_w->copy(max_sel_dim_rewrapped);
      // Next iteration
      input_sw.next();
      output_sw.next();
      // Free memory
      DecRef(max_sel_dim_rewrapped);
      DecRef(max_sel_dim);
      DecRef(input_w);
      DecRef(output_w);
      k += input_mat->getDimSize(0);
    }
    return output;
  }
  
  Token *MaxPoolingANNComponent::doBackprop(Token *_error_input) {
    // error checking
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input error Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(error_input,_error_input->convertTo<TokenMatrixFloat*>());
    MatrixFloat *error_input_mat=error_input->getMatrix();
    if (!output->getMatrix()->sameDim(error_input_mat))
      ERROR_EXIT1(129, "Incorrect dimensions at error input matrix [%s]\n",
		  name.c_str());
#ifdef USE_CUDA
    error_input_mat->setUseCuda(use_cuda);
#endif
    MatrixFloat *error_output_mat = input->getMatrix()->cloneOnlyDims();
    AssignRef(error_output, new TokenMatrixFloat(error_output_mat));
    error_output_mat->zeros();
    
    // Prepare sliding windows to compute the convolution gradient
    MatrixFloat::sliding_window error_output_sw(error_output_mat, input_window_size,
						0,  // OFFSET
						kernel_step,
						input_window_num_steps,
						input_window_order_step);
    MatrixFloat::sliding_window error_input_sw(error_input_mat, output_window_size,
					       0,  // OFFSET
					       output_window_step,
					       output_window_num_steps,
					       output_window_order_step);
    assert(argmax_raw_pos != 0);
    assert(static_cast<int>(argmax_raw_pos->getSize()) == error_input_mat->size());
    assert(error_output_sw.numWindows() == error_input_sw.numWindows());
    const int *argmax_ints = argmax_raw_pos->getPPALForRead();
    float *error_output_ptr = error_output_mat->getRawDataAccess()->getPPALForReadAndWrite();
    // CONVOLUTION GRADIENT
    while(!error_input_sw.isEnd()) {
      MatrixFloat *error_input_w = error_input_sw.getMatrix();
      
      IncRef(error_input_w);
      for (MatrixFloat::const_iterator it(error_input_w->begin());
	   it!=error_input_w->end(); ++it, ++argmax_ints) {
	(*error_output_mat)[*argmax_ints] += *it;
	/*
	  int jaja[4];
	  error_output_w->computeCoords(*argmax_ints, jaja);
	  printf ("ERROR_OUTPUT-W[");
	  for (int i=0; i<4; ++i) printf (" %d", jaja[i]);
	  printf(" ] :: %d => %g\n", *argmax_ints, *it);
	  error_output_mat->computeCoords(*argmax_ints, jaja);
	  printf ("ERROR_OUTPUT-M[");
	  for (int i=0; i<4; ++i) printf (" %d", jaja[i]);
	  printf(" ] :: %d => %g\n", *argmax_ints, *it);
	*/
      }
      // Next iteration
      error_input_sw.next();
      
      // Free memory
      DecRef(error_input_w);
    }
    return error_output;
  }
  
  void MaxPoolingANNComponent::reset() {
    if (input)          DecRef(input);
    if (error_input)    DecRef(error_input);
    if (output)         DecRef(output);
    if (error_output)   DecRef(error_output);
    if (argmax_raw_pos) DecRef(argmax_raw_pos);
    input	   = 0;
    error_input	   = 0;
    output	   = 0;
    error_output   = 0;
    argmax_raw_pos = 0;
  }
  
  ANNComponent *MaxPoolingANNComponent::clone() {
    MaxPoolingANNComponent *component = new
      MaxPoolingANNComponent(input_num_dims, kernel_dims+1, kernel_step+1,
			     name.c_str());
    component->input_size     = input_size;
    component->output_size    = output_size;
    return component;
  }
  
  char *MaxPoolingANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.max_pooling{ name='%s', kernel={",
		  name.c_str());
    for (int i=0; i<input_num_dims; ++i)
      buffer.printf("%d,", kernel_dims[i+1]);
    buffer.printf("}, step={");
    for (int i=0; i<input_num_dims; ++i)
      buffer.printf("%d,", kernel_step[i+1]);
    buffer.printf("} }");
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  //////////////////////////////////////////////////////////////////////////
}
