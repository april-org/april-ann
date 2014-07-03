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
#include "convolution_component.h"
#include "token_matrix.h"
#include "table_of_token_codes.h"

namespace ANN {

  ///////////////////////////////////////////
  // ConvolutionANNComponent implementation //
  ///////////////////////////////////////////

  void ConvolutionANNComponent::initializeArrays(const int *input_dims) {
    for (int i=1; i<input_planes_dim; ++i) {
      output_dims[i+1] = (input_dims[i] - kernel_dims[i])/kernel_step[i] + 1;
      input_window_num_steps[i]    = output_dims[i+1];
      output_window_num_steps[i+1] = output_dims[i+1];
    }
    for (int i=input_planes_dim+1; i<=input_num_dims; ++i) {
      output_dims[i] = (input_dims[i] - kernel_dims[i])/kernel_step[i] + 1;
      input_window_num_steps[i]  = output_dims[i];
      output_window_num_steps[i] = output_dims[i];
    }
    /*
      for (int i=1; i<=input_num_dims; ++i) {
      output_dims[i] = (input_dims[i] - kernel_dims[i])/kernel_step[i] + 1;
      input_window_num_steps[i]  = output_dims[i];
      output_window_num_steps[i] = output_dims[i];
      }
    */
    // input_dims[0]  => BUNCH SIZE
    // output_dims[0] => BUNCH SIZE, output_dims[1] => HIDDEN LAYER SIZE
    output_dims[0]	       = input_dims[0];
    output_dims[1]	       = hidden_size;
    input_window_size[0]       = input_dims[0];
    input_window_num_steps[0]  = 1;
    input_window_num_steps[input_planes_dim] = 1;
    output_window_size[0]      = input_dims[0];
    // AT CONSTRUCTOR: output_window_size[1] = hidden_size;
    output_window_num_steps[0] = 1;
    output_window_num_steps[1] = 1;
    input_window_rewrap[0]     = input_dims[0];
    // AT CONSTRUCTOR: input_window_rewrap[1] = kernel_size;
    output_window_rewrap[0]    = input_dims[0];
    // AT CONSTRUCTOR: output_window_rewrap[1] = hidden_size;
    if (input_dims[input_planes_dim] != kernel_dims[input_planes_dim])
      ERROR_EXIT7(128, "Input matrix dim %d must be equals to kernel dim %d, "
		  "input_dims[%d]=%d, kernel_dims[%d]=%d [%s]\n",
		  input_planes_dim, input_planes_dim,
		  input_planes_dim, input_dims[input_planes_dim],
		  input_planes_dim, kernel_dims[input_planes_dim],
		  name.c_str());
  }
  
  ConvolutionANNComponent::ConvolutionANNComponent(int input_num_dims,
						   const int *_kernel_dims,
						   const int *_kernel_step,
						   const int input_planes_dim,
						   int num_output_planes,
						   const char *name,
						   const char *weights_name) :
    VirtualMatrixANNComponent(name, weights_name, 0, 0),
    weights_matrix(0),
    input_planes_dim(input_planes_dim),
    number_input_windows(0),
    kernel_size(1),
    hidden_size(num_output_planes),
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
    setInputContiguousProperty(true);
    if (weights_name == 0) generateDefaultWeightsName("w");
    kernel_dims[0] = static_cast<int>(hidden_size);
    kernel_step[0] = 1;
    input_window_order_step[0] = 0;
    output_window_size[0] = 0;
    output_window_size[1] = static_cast<int>(hidden_size);
    output_window_order_step[0] = 0;
    output_window_step[0] = 1;
    output_window_step[1] = 1;
    for(int i=0; i<input_num_dims; ++i) {
      kernel_size *= _kernel_dims[i];
      kernel_dims[i+1] = _kernel_dims[i];
      kernel_step[i+1] = _kernel_step[i];
      input_window_order_step[i+1] = i+1;
      output_window_order_step[i+1] = i+1;
      input_window_size[i+1] = kernel_dims[i+1];
    }
    for(int i=2; i<=input_num_dims; ++i) {
      output_window_size[i] = 1;
      output_window_step[i] = 1;
    }
    input_window_rewrap[0]  = 0;
    input_window_rewrap[1]  = static_cast<int>(kernel_size);
    output_window_rewrap[0] = 0;
    output_window_rewrap[1] = static_cast<int>(hidden_size);
  }
  
  ConvolutionANNComponent::~ConvolutionANNComponent() {
    if (weights_matrix) DecRef(weights_matrix);
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
  
  MatrixFloat *ConvolutionANNComponent::
  privateDoForward(MatrixFloat *input_mat, bool during_training) {
    UNUSED_VARIABLE(during_training);
    if (weights_matrix == 0) ERROR_EXIT1(129, "Not built component %s\n",
					 name.c_str());
    MatrixFloat *weights_mat = weights_matrix;
    // error checking
    if (input_mat->getNumDim() != input_num_dims+1)
      ERROR_EXIT3(129, "Incorrect input matrix numDims, "
		  "expected %d, found %d [%s]\n", input_num_dims+1,
		  input_mat->getNumDim(), name.c_str());
    const int *input_dims = input_mat->getDimPtr();
    initializeArrays(input_dims);
    MatrixFloat *output_mat;
    output_mat = new MatrixFloat(input_num_dims+1, output_dims, CblasColMajor);
    IncRef(output_mat);
#ifdef USE_CUDA
    output_mat->setUseCuda(use_cuda);
#endif
    
    /////////////////////////////////////////////////////////////////////////

    // Prepare sliding windows to compute the convolution
    MatrixFloat::sliding_window *input_sw =
      new MatrixFloat::sliding_window(input_mat, input_window_size,
                                      0,  // OFFSET
                                      kernel_step,
                                      input_window_num_steps,
                                      input_window_order_step);
    MatrixFloat::sliding_window *output_sw =
      new MatrixFloat::sliding_window(output_mat, output_window_size,
                                      0,  // OFFSET
                                      output_window_step,
                                      output_window_num_steps,
                                      output_window_order_step);
    number_input_windows = input_sw->numWindows();
    // CONVOLUTION OVER number_input_windows
    MatrixFloat *input_w  = input_sw->getMatrix();
    MatrixFloat *output_w = output_sw->getMatrix();
    IncRef(input_w);
    IncRef(output_w);
    while(!input_sw->isEnd() && !output_sw->isEnd()) {
      // reusing the same MatrixFloat across all the possible windows
      input_sw->getMatrix(input_w);
      output_sw->getMatrix(output_w);
      MatrixFloat *input_flattened  = getRewrappedMatrix(input_w,
							 input_window_rewrap,
							 2, true);
      MatrixFloat *output_flattened = getRewrappedMatrix(output_w,
							 output_window_rewrap,
							 2, false);
      IncRef(input_flattened);
      IncRef(output_flattened);
      
      // COMPUTE MATRIX MULTIPLICATION
      output_flattened->gemm(CblasNoTrans, CblasTrans,
			     1.0f, input_flattened,
			     weights_mat,
			     0.0f);
      // COPY TO DESTINATION IF NEEDED
      if (output_w->getRawDataAccess()!=output_flattened->getRawDataAccess()) {
	// if output_w and output_flattened are pointing to different data
	// then the copy is needed, otherwise it isn't
	MatrixFloat *conv_output_rewrapped;
	conv_output_rewrapped = output_flattened->rewrap(output_w->getDimPtr(),
							 output_w->getNumDim());
	IncRef(conv_output_rewrapped);
	output_w->copy(conv_output_rewrapped);
	DecRef(conv_output_rewrapped);
      }
      
      // Next iteration
      input_sw->next();
      output_sw->next();
      
      // Free memory
      DecRef(input_flattened);
      DecRef(output_flattened);
    }
    DecRef(input_w);
    DecRef(output_w);
    delete input_sw;
    delete output_sw;
    ReleaseRef(output_mat);
    return output_mat;
  }
  
  MatrixFloat *ConvolutionANNComponent::
  privateDoBackprop(MatrixFloat *error_input_mat) {
    MatrixFloat *weights_mat = weights_matrix;
    MatrixFloat *output_mat  = getOutputMatrix();
    MatrixFloat *input_mat   = getInputMatrix();
    if (!output_mat->sameDim(error_input_mat))
      ERROR_EXIT1(129, "Incorrect dimensions at error input matrix [%s]\n",
		  name.c_str());
    MatrixFloat *error_output_mat = input_mat->cloneOnlyDims();
    IncRef(error_output_mat);
    // initialization of error_output_mat is needed because of kernel
    // overlapping
    error_output_mat->zeros();

    // Prepare sliding windows to compute the convolution gradient
    MatrixFloat::sliding_window *error_output_sw =
      new MatrixFloat::sliding_window(error_output_mat, input_window_size,
                                      0,  // OFFSET
                                      kernel_step,
                                      input_window_num_steps,
                                      input_window_order_step);
    MatrixFloat::sliding_window *error_input_sw =
      new MatrixFloat::sliding_window(error_input_mat, output_window_size,
                                      0,  // OFFSET
                                      output_window_step,
                                      output_window_num_steps,
                                      output_window_order_step);
    april_assert(error_input_sw->numWindows() == number_input_windows);
    // CONVOLUTION GRADIENT
    MatrixFloat *error_input_w  = error_input_sw->getMatrix();
    MatrixFloat *error_output_w = error_output_sw->getMatrix();
    IncRef(error_input_w);
    IncRef(error_output_w);
    while(!error_input_sw->isEnd() && !error_output_sw->isEnd()) {
      // reuse the same MatrixFloat across all possible windows
      error_input_sw->getMatrix(error_input_w);
      error_output_sw->getMatrix(error_output_w);
      MatrixFloat *error_input_flattened  = getRewrappedMatrix(error_input_w,
							       output_window_rewrap,
							       2, true);
      MatrixFloat *error_output_flattened = getRewrappedMatrix(error_output_w,
							       input_window_rewrap,
							       2, true);
      IncRef(error_input_flattened);
      IncRef(error_output_flattened);
      
      // COMPUTE MATRIX MULTIPLICATION
      error_output_flattened->gemm(CblasNoTrans, CblasNoTrans,
				   1.0f, error_input_flattened,
				   weights_mat,
				   1.0f); // accumulative operation
      // COPY TO DESTINATION IF NEEDED
      if (error_output_w->getRawDataAccess()!=error_output_flattened->getRawDataAccess()) {
	// if error_output_w and error_output_flattened are pointing to
	// different data then the copy is needed, otherwise it isn't
	MatrixFloat *conv_error_output_rewrapped;
	conv_error_output_rewrapped =
	  error_output_flattened->rewrap(error_output_w->getDimPtr(),
					 error_output_w->getNumDim());
	IncRef(conv_error_output_rewrapped);
	// COPY THE RESULT
	error_output_w->copy(conv_error_output_rewrapped);
	DecRef(conv_error_output_rewrapped);
      }
      
      // Next iteration
      error_input_sw->next();
      error_output_sw->next();
      
      // Free memory
      DecRef(error_input_flattened);
      DecRef(error_output_flattened);
    }
    DecRef(error_input_w);
    DecRef(error_output_w);
    delete error_input_sw;
    delete error_output_sw;
    ReleaseRef(error_output_mat);
    return error_output_mat;
  }
  
  void ConvolutionANNComponent::computeGradients(MatrixFloat *&grads_mat) {
    weights_matrix->addToSharedCount(number_input_windows);
    if (grads_mat == 0) {
      grads_mat = weights_matrix->cloneOnlyDims();
      grads_mat->zeros();
      IncRef(grads_mat);
    }
    MatrixFloat *input_mat       = getInputMatrix();
    MatrixFloat *error_input_mat = getErrorInputMatrix();
    // Prepare sliding windows to compute the convolution
    MatrixFloat::sliding_window input_sw(input_mat, input_window_size,
					 0,  // OFFSET
					 kernel_step,
					 input_window_num_steps,
					 input_window_order_step);
    MatrixFloat::sliding_window error_input_sw(error_input_mat, output_window_size,
					       0,  // OFFSET
					       output_window_step,
					       output_window_num_steps,
					       output_window_order_step);
    MatrixFloat *input_w       = input_sw.getMatrix();
    MatrixFloat *error_input_w = error_input_sw.getMatrix();
    IncRef(input_w);
    IncRef(error_input_w);
    while(!input_sw.isEnd() && !error_input_sw.isEnd()) {
      input_sw.getMatrix(input_w);
      error_input_sw.getMatrix(error_input_w);
      MatrixFloat *input_flattened = getRewrappedMatrix(input_w,
							input_window_rewrap,
							2, true);
      MatrixFloat *error_input_flattened = getRewrappedMatrix(error_input_w,
							      output_window_rewrap,
							      2, true);
      IncRef(input_flattened);
      IncRef(error_input_flattened);
      
      // WEIGHTS UPDATE
      grads_mat->gemm(CblasTrans, CblasNoTrans,
		      1.0f,
		      error_input_flattened, // A
		      input_flattened,       // B
		      1.0f);
      
      // Next iteration
      input_sw.next();
      error_input_sw.next();
      
      // Free memory
      DecRef(input_flattened);
      DecRef(error_input_flattened);
    }
    DecRef(input_w);
    DecRef(error_input_w);
  }

  void ConvolutionANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
    weights_matrix->resetSharedCount();
  }
  
  ANNComponent *ConvolutionANNComponent::clone() {
    ConvolutionANNComponent *component = new
      ConvolutionANNComponent(input_num_dims, kernel_dims+1, kernel_step+1,
			      input_planes_dim, hidden_size,
			      name.c_str(), weights_name.c_str());
    component->input_size     = input_size;
    component->output_size    = output_size;
    return component;
  }

  void ConvolutionANNComponent::build(unsigned int _input_size,
				     unsigned int _output_size,
				     MatrixFloatSet *weights_dict,
				     hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    //
    unsigned int weights_input_size  = kernel_size;
    unsigned int weights_output_size = hidden_size;
    ////////////////////////////////////////////////////////////////////
    MatrixFloat *&w = (*weights_dict)[weights_name];
    // printf("%s :: %p %p\n", weights_name.c_str(), w, weights_matrix);
    if (w != 0) {
      // printf("COPY OF WEIGHTS FROM HASH %s\n", weights_name.c_str());
      AssignRef(weights_matrix, w);
      if (!Connections::checkInputOutputSizes(weights_matrix,
					      weights_input_size,
					      weights_output_size))
	ERROR_EXIT3(256,"The weights matrix input/output sizes are not correct, "
		    "expected %dx%d [%s]\n",
		    weights_input_size, weights_output_size,
		    name.c_str());
    }
    else {
      if (weights_matrix == 0) {
	// printf("NEW OF WEIGHTS %s\n", weights_name.c_str());
	weights_matrix = Connections::build(weights_input_size,
					    weights_output_size);
	IncRef(weights_matrix);
      }
      // else printf("USING PREVIOUS WEIGHTS %s\n", weights_name.c_str());
      w = weights_matrix;
      IncRef(w);
    }
  }

  void ConvolutionANNComponent::copyWeights(MatrixFloatSet *weights_dict) {
    if (weights_matrix == 0)
      ERROR_EXIT1(100, "Component not built, impossible execute copyWeights [%s]\n",
		  name.c_str());
    MatrixFloat *&w = (*weights_dict)[weights_name];
    if (w != 0 && w != weights_matrix)
      ERROR_EXIT2(101, "Weights dictionary contains %s weights name which is "
		  "not shared with weights_matrix attribute [%s]\n",
		  weights_name.c_str(),
		  name.c_str());
    else if (w == 0) {
      w = weights_matrix;
      IncRef(w);
    }
  }  

  char *ConvolutionANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.convolution{ name='%s',weights='%s',"
		  "n=%d, input_planes_dim=%d, kernel={", name.c_str(), weights_name.c_str(),
		  hidden_size, input_planes_dim);
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
