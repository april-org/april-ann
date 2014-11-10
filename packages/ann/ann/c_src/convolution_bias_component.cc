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
#include "convolution_bias_component.h"
#include "token_matrix.h"
#include "table_of_token_codes.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  ////////////////////////////////////////////////
  // ConvolutionBiasANNComponent implementation //
  ////////////////////////////////////////////////

  void ConvolutionBiasANNComponent::initializeArrays(const int *input_dims) {
    // input_dims[0]  => BUNCH SIZE, input_dims[1] => HIDDEN SIZE
    window_size[0] = input_dims[0];
    for (int i=2; i<=num_dims; ++i) window_num_steps[i] = input_dims[i];
  }
  
  // A method to convert the bias vector of size Nx1 in a matrix of size
  // BUNCHxK1xK2x...xKM being Ki the kernel size at dimension i
  MatrixFloat *ConvolutionBiasANNComponent::prepareBiasBunch() {
    /*
      FIXME: When the following improvement is set, the performance of the ANN
      degenerates...
      
      if (bias_matrix != 0 && bias_matrix->sameDim(window_size, num_dims + 1))
      return bias_matrix;
    */
    // this line converts the bias matrix of Nx1 in a vector of N elements
    MatrixFloat *bias_vec = bias_vector->select(1,0);
    IncRef(bias_vec);
    // the output bias as a 2d matrix of BUNCHxN
    MatrixFloat *bias_matrix_2d = new MatrixFloat(2, window_size);
    IncRef(bias_matrix_2d);
    // first pattern is done out of the loop
    MatrixFloat *dest = bias_matrix_2d->select(0, 0);
    IncRef(dest);
    matCopy(dest,bias_vec);
    // for the rest of patterns at the bunch
    for (int b=1; b<window_size[0]; ++b) {
      // select the row b at the output bias matrix
      bias_matrix_2d->select(0, b, dest);
      matCopy(dest,bias_vec);
    }
    DecRef(dest);
    if (bias_matrix) DecRef(bias_matrix);
    // reinterpret the output bias matrix of BUNCHxN to fit with the output
    // sliding window of BUNCHxK1xK2x....xKM where Ki is the kernel size at
    // dimension i
    bias_matrix = bias_matrix_2d->rewrap(window_size, num_dims + 1);
    IncRef(bias_matrix);
    // free the memory of auxiliary matrices
    DecRef(bias_matrix_2d);
    DecRef(bias_vec);
    return bias_matrix;
  }

  ConvolutionBiasANNComponent::
  ConvolutionBiasANNComponent(int num_dims,
			      unsigned int num_output_planes,
			      const char *name,
			      const char *bias_name) :
    VirtualMatrixANNComponent(name, bias_name, 0, 0),
    bias_vector(0),
    bias_matrix(0),
    hidden_size(num_output_planes),
    num_dims(num_dims),
    window_size(new int[num_dims+1]),
    window_step(new int[num_dims+1]),
    window_num_steps(new int[num_dims+1]) {
    setInputContiguousProperty(true);
    if (bias_name == 0) generateDefaultWeightsName("b");
    window_size[1]     = static_cast<int>(hidden_size);
    window_step[0]     = 1;
    window_step[1]     = 1;
    window_num_steps[0] = 1;
    window_num_steps[1] = 1;
    for (int i=2; i<=num_dims; ++i) {
      window_size[i] = 1;
      window_step[i] = 1;
    }
  }
  
  ConvolutionBiasANNComponent::~ConvolutionBiasANNComponent() {
    if (bias_vector) DecRef(bias_vector);
    delete[] window_size;
    delete[] window_step;
    delete[] window_num_steps;
    if (bias_matrix) DecRef(bias_matrix);
  }
  
  MatrixFloat *ConvolutionBiasANNComponent::
  privateDoForward(MatrixFloat *input_mat, bool during_training) {
    UNUSED_VARIABLE(during_training);
    if (bias_vector == 0) ERROR_EXIT1(129, "Not built component %s\n",
				      name.c_str());
    if (input_mat->getNumDim() != num_dims+1)
      ERROR_EXIT3(129, "Incorrect input matrix numDims, "
		  "expected %d, found %d [%s]\n", num_dims+1,
		  input_mat->getNumDim(), name.c_str());
    const int *input_dims = input_mat->getDimPtr();
    if (input_dims[1] != static_cast<int>(hidden_size))
      ERROR_EXIT3(129,"Incorrect input dim[1] size, found %d, expected %d [%s]\n",
		  input_dims[1], hidden_size, name.c_str());
    initializeArrays(input_dims);
    MatrixFloat *output_mat;
    output_mat = input_mat->clone();
    IncRef(output_mat);
    MatrixFloat *bias_matrix = prepareBiasBunch();
    IncRef(bias_matrix);
    /////////////////////////////////////////////////////////////////////////

    // Prepare sliding windows to compute the convolution
    MatrixFloat::sliding_window *input_sw =
      new MatrixFloat::sliding_window(input_mat, window_size,
                                      0,  // OFFSET
                                      window_step,
                                      window_num_steps);
    MatrixFloat::sliding_window *output_sw =
      new MatrixFloat::sliding_window(output_mat, window_size,
                                      0,  // OFFSET
                                      window_step,
                                      window_num_steps);
    number_input_windows = input_sw->numWindows();
    // CONVOLUTION OVER number_input_windows
    MatrixFloat *input_w  = input_sw->getMatrix();
    MatrixFloat *output_w = output_sw->getMatrix();
    IncRef(input_w);
    IncRef(output_w);
    while(!input_sw->isEnd() && !output_sw->isEnd()) {
      input_sw->getMatrix(input_w);
      output_sw->getMatrix(output_w);
      // ADD BIAS
      matAxpy(output_w, 1.0f, bias_matrix);
      // Next iteration
      input_sw->next();
      output_sw->next();
    }
    // Free memory
    DecRef(input_w);
    DecRef(output_w);
    DecRef(bias_matrix);
    delete input_sw;
    delete output_sw;
    ReleaseRef(output_mat);
    return output_mat;
  }
  
  MatrixFloat *ConvolutionBiasANNComponent::
  privateDoBackprop(MatrixFloat *error_mat) {
    return error_mat;
  }
     
  void ConvolutionBiasANNComponent::computeGradients(const char *name,
                                                     AprilUtils::LuaTable &grads_mat_dict) {
    // reset shared counter
    bias_vector->addToSharedCount(number_input_windows);
    MatrixFloat *grads_mat = grads_mat_dict.opt<MatrixFloat*>(name, 0);
    if (grads_mat == 0) {
      grads_mat = bias_vector->cloneOnlyDims();
      matZeros(grads_mat);
      grads_mat_dict.put(name, grads_mat);
    }
#ifdef USE_CUDA
    grads_mat->setUseCuda(use_cuda);
#endif
    MatrixFloat *input_error_mat = getErrorInputMatrix();
    // Prepare sliding windows to compute the convolution
    MatrixFloat::sliding_window error_sw(input_error_mat, window_size,
					 0,  // OFFSET
					 window_step,
					 window_num_steps);
    unsigned int bunch_size = input_error_mat->getDimSize(0);
    april_assert(error_sw.numWindows() == number_input_windows);
    MatrixFloat *error_w = error_sw.getMatrix();
    IncRef(error_w);
    while(!error_sw.isEnd()) {
      error_sw.getMatrix(error_w);
      // BIAS UPDATE
      doAxpyLoop(hidden_size,
		 1.0f,
		 error_w->getRawDataAccess(),
		 error_w->getStrideSize(1),
		 error_w->getOffset(),
		 grads_mat->getRawDataAccess(),
		 grads_mat->getStrideSize(0),
		 0,
		 bunch_size,
		 error_w->getStrideSize(0), 0,
		 use_cuda);
      // Next iteration
      error_sw.next();
    }
    // Free memory
    DecRef(error_w);
  }

  void ConvolutionBiasANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
    // reset shared counter
    bias_vector->resetSharedCount();
  }
  
  ANNComponent *ConvolutionBiasANNComponent::clone() {
    ConvolutionBiasANNComponent *component = new
      ConvolutionBiasANNComponent(num_dims, hidden_size,
				  name.c_str(), weights_name.c_str());
    component->input_size     = input_size;
    component->output_size    = output_size;
    return component;
  }

  void ConvolutionBiasANNComponent::build(unsigned int _input_size,
					  unsigned int _output_size,
					  AprilUtils::LuaTable &weights_dict,
					  AprilUtils::LuaTable &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    ////////////////////////////////////////////////////////////////////
    MatrixFloat *b = weights_dict.opt<MatrixFloat*>(weights_name, 0);
    if (b != 0) {
      AssignRef(bias_vector, b);
      if (!Connections::checkInputOutputSizes(bias_vector,1,hidden_size))
	ERROR_EXIT2(256,"The bias vector input/output sizes are not correct, "
		    "expected 1x%d [%s]\n", hidden_size, name.c_str());
    }
    else {
      if (bias_vector == 0) {
	bias_vector = Connections::build(1, hidden_size);
	IncRef(bias_vector);
      }
      weights_dict.put(weights_name, bias_vector);
    }
  }

  void ConvolutionBiasANNComponent::copyWeights(AprilUtils::LuaTable &weights_dict) {
    if (bias_vector == 0)
      ERROR_EXIT1(100, "Component not built, impossible execute copyWeights [%s]\n",
		  name.c_str());
    MatrixFloat *b = weights_dict.opt<MatrixFloat*>(weights_name, 0);
    if (b != 0 && b != bias_vector)
      ERROR_EXIT2(101, "Weights dictionary contains %s bias name which is "
		  "not shared with bias_vector attribute [%s]\n",
		  weights_name.c_str(),
		  name.c_str());
    else if (b == 0) {
      weights_dict.put(weights_name, bias_vector);
    }
  }  

  char *ConvolutionBiasANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.convolution_bias{ name='%s',weights='%s', "
		  "n=%d, ndims=%d }", name.c_str(), weights_name.c_str(),
		  hidden_size, num_dims);
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  //////////////////////////////////////////////////////////////////////////
}
