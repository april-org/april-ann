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
#include "convolution_bias_component.h"
#include "token_matrix.h"
#include "table_of_token_codes.h"

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
    if (bias_matrix != 0 && bias_matrix->getDimSize(0) == window_size[0])
      return bias_matrix;
    // this line converts the bias matrix of Nx1 in a vector of N elements
    MatrixFloat *bias_vec = bias_vector->getPtr()->select(1,0);
    IncRef(bias_vec);
    // the output bias as a 2d matrix of BUNCHxN
    MatrixFloat *bias_matrix_2d = new MatrixFloat(2, window_size,
						  CblasColMajor);
    IncRef(bias_matrix_2d);
    // for each pattern at the bunch
    for (int b=0; b<window_size[0]; ++b) {
      // select the row b at the output bias matrix
      MatrixFloat *dest = bias_matrix_2d->select(0, b);
      IncRef(dest);
      dest->copy(bias_vec);
      DecRef(dest);
    }
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
    ANNComponent(name, bias_name, 0, 0),
    input(0),
    error(0),
    output(0),
    bias_vector(0),
    bias_matrix(0),
    num_updates_from_last_prune(0),
    hidden_size(num_output_planes),
    num_dims(num_dims),
    window_size(new int[num_dims+1]),
    window_step(new int[num_dims+1]),
    window_num_steps(new int[num_dims+1]),
    learning_rate(-1.0f),
    momentum(0.0f) {
    if (bias_name == 0) generateDefaultWeightsName(this->weights_name, "b");
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
    if (input) DecRef(input);
    if (error) DecRef(error);
    if (output) DecRef(output);
    delete[] window_size;
    delete[] window_step;
    delete[] window_num_steps;
    if (bias_matrix) DecRef(bias_matrix);
  }
  
  // The ConvolutionBiasANNComponent
  Token *ConvolutionBiasANNComponent::doForward(Token *_input,
						bool during_training) {
    if (bias_vector == 0) ERROR_EXIT1(129, "Not built component %s\n",
				      name.c_str());
    // error checking
    if (_input == 0) ERROR_EXIT1(129,"Null Token received! [%s]\n",
				 name.c_str());
    if (_input->getTokenCode() != table_of_token_codes::token_matrix)
      ERROR_EXIT1(129, "Incorrect token received, expected token_matrix [%s]\n",
		  name.c_str());
    AssignRef(input, _input->convertTo<TokenMatrixFloat*>());
    MatrixFloat *input_mat=input->getMatrix();
    if (input_mat->getNumDim() != num_dims+1)
      ERROR_EXIT3(129, "Incorrect input matrix numDims, "
		  "expected %d, found %d [%s]\n", num_dims+1,
		  input_mat->getNumDim(), name.c_str());
#ifdef USE_CUDA
    input_mat->setUseCuda(use_cuda);
#endif
    const int *input_dims = input_mat->getDimPtr();
    if (input_dims[1] != static_cast<int>(hidden_size))
      ERROR_EXIT3(129,"Incorrect input dim[1] size, found %d, expected %d [%s]\n",
		  input_dims[1], hidden_size, name.c_str());
    initializeArrays(input_dims);
    MatrixFloat *output_mat;
    output_mat = input_mat->clone();
    AssignRef(output, new TokenMatrixFloat(output_mat));
    
    MatrixFloat *bias_matrix = prepareBiasBunch();
    IncRef(bias_matrix);
    /////////////////////////////////////////////////////////////////////////

    // Prepare sliding windows to compute the convolution
    MatrixFloat::sliding_window input_sw(input_mat, window_size,
					 0,  // OFFSET
					 window_step,
					 window_num_steps);
    MatrixFloat::sliding_window output_sw(output_mat, window_size,
					  0,  // OFFSET
					  window_step,
					  window_num_steps);
    number_input_windows = input_sw.numWindows();
    // CONVOLUTION OVER number_input_windows
    while(!input_sw.isEnd() && !output_sw.isEnd()) {
      MatrixFloat *input_w  = input_sw.getMatrix();
      MatrixFloat *output_w = output_sw.getMatrix();
      IncRef(input_w);
      IncRef(output_w);
      // ADD BIAS
      output_w->axpy(1.0f, bias_matrix);
      // Next iteration
      input_sw.next();
      output_sw.next();
      // Free memory
      DecRef(input_w);
      DecRef(output_w);
    }
    DecRef(bias_matrix);
    return output;
  }
  
  Token *ConvolutionBiasANNComponent::doBackprop(Token *_error_input) {
    // error checking
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input error Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(error,_error_input->convertTo<TokenMatrixFloat*>());
    MatrixFloat *error_mat=error->getMatrix();
    if (!output->getMatrix()->sameDim(error_mat))
      ERROR_EXIT1(129, "Incorrect dimensions at error input matrix [%s]\n",
		  name.c_str());
#ifdef USE_CUDA
    error_mat->setUseCuda(use_cuda);
#endif
    return error;
  }
     
  // The ConvolutionBiasANNComponent
  void ConvolutionBiasANNComponent::doUpdate() {
    assert(learning_rate > 0.0f &&
	   "Learning rate needs to be fixed with setOption method!!!");
    
    // Foces weights_matrix to update internal counts for a backward step
    bias_vector->beginUpdate();
    
    MatrixFloat *prev_bias_ptr = bias_vector->getPrevPtr();
    MatrixFloat *error_mat     = error->getMatrix();
    
    // BIAS MOMENTUM
    if (bias_vector->isFirstUpdateCall()) {
      if (momentum > 0.0f) {
        // prev_w[i,j] = momentum * (w[i,j] - prev_w[i,j])
        bias_vector->computeMomentumOnPrevVector(momentum, use_cuda);
        bias_vector->computeWeightDecayOnPrevVector(1.0f,  use_cuda);
      }
      else bias_vector->copyToPrevVector(use_cuda);
    } // if (bias_vector->needsToComputeMomentum()) {
    
    unsigned int bunch_size = error_mat->getDimSize(0);
    // backprop learning rule:
    // PREV_W = alpha * ERRORS + PREV_W
    const unsigned int references = bias_vector->getNumReferences();
    assert(references > 0 && "Found 0 references of weights matrix");
    // prev_w[i,j] = -learning_rate*1/sqrt(N*bsize) * ERROR_INPUT[j] + prev_w[i,j]
    const float norm_learn_rate =
      -(1.0f/sqrtf(static_cast<float>(references*bunch_size*number_input_windows))) *
      learning_rate;
    
    // CONVOLUTION OVER number_input_windows
    computeBP(prev_bias_ptr, error_mat, norm_learn_rate);
    
    // Forces to update counts and swap vectors if necessary at this backward
    // step
    if (bias_vector->endUpdate()) {
      ++num_updates_from_last_prune;
      if (num_updates_from_last_prune > MAX_UPDATES_WITHOUT_PRUNE) {
	num_updates_from_last_prune = 0;
	bias_vector->pruneSubnormalAndCheckNormal();
      }
    }
  }

  void ConvolutionBiasANNComponent::computeBP(MatrixFloat *weights_mat,
					      MatrixFloat *error_mat,
					      const float alpha) {
    // Prepare sliding windows to compute the convolution
    MatrixFloat::sliding_window error_sw(error_mat, window_size,
					 0,  // OFFSET
					 window_step,
					 window_num_steps);
    unsigned int bunch_size = error_mat->getDimSize(0);
    assert(error_sw.numWindows() == number_input_windows);
    while(!error_sw.isEnd()) {
      MatrixFloat *error_w = error_sw.getMatrix();
      IncRef(error_w);
      // BIAS UPDATE
      doSaxpyLoop(hidden_size,
		  alpha,
		  error_w->getRawDataAccess(),
		  error_w->getStrideSize(1),
		  error_w->getOffset(),
		  weights_mat->getRawDataAccess(),
		  weights_mat->getStrideSize(0),
		  0,
		  bunch_size,
		  error_w->getStrideSize(0), 0,
		  use_cuda);
      // Next iteration
      error_sw.next();
      // Free memory
      DecRef(error_w);
    }
  }

  void ConvolutionBiasANNComponent::computeGradients(MatrixFloat*& weight_grads) {
    if (weight_grads == 0) {
      weight_grads = bias_vector->getPtr()->cloneOnlyDims();
      weight_grads->zeros();
    }
    MatrixFloat *input_error_mat = error->getMatrix();
    unsigned int bunch_size = input_error_mat->getDimSize(0)*number_input_windows;
    computeBP(weight_grads, error->getMatrix(), 1.0f/bunch_size);
  }

  
  void ConvolutionBiasANNComponent::reset() {
    if (input)        DecRef(input);
    if (error)        DecRef(error);
    if (output)       DecRef(output);
    input	 = 0;
    error 	 = 0;
    output	 = 0;
  }
  
  ANNComponent *ConvolutionBiasANNComponent::clone() {
    ConvolutionBiasANNComponent *component = new
      ConvolutionBiasANNComponent(num_dims, hidden_size,
				  name.c_str(), weights_name.c_str());
    component->input_size     = input_size;
    component->output_size    = output_size;
    component->learning_rate  = learning_rate;
    component->momentum       = momentum;
    return component;
  }

  void ConvolutionBiasANNComponent::setOption(const char *name, double value) {
    mSetOption(LEARNING_RATE_STRING, learning_rate);
    mSetOption(MOMENTUM_STRING,      momentum);
    ANNComponent::setOption(name, value);
  }
  
  bool ConvolutionBiasANNComponent::hasOption(const char *name) {
    mHasOption(LEARNING_RATE_STRING);
    mHasOption(MOMENTUM_STRING);
    return false;
  }
  
  double ConvolutionBiasANNComponent::getOption(const char *name) {
    mGetOption(LEARNING_RATE_STRING, learning_rate);
    mGetOption(MOMENTUM_STRING, momentum);
    return ANNComponent::getOption(name);
  }
  
  void ConvolutionBiasANNComponent::build(unsigned int _input_size,
					  unsigned int _output_size,
					  hash<string,Connections*> &weights_dict,
					  hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    ////////////////////////////////////////////////////////////////////
    Connections *&b = weights_dict[weights_name];
    if (b != 0) {
      AssignRef(bias_vector, b);
      if (!bias_vector->checkInputOutputSizes(1,hidden_size))
	ERROR_EXIT2(256,"The bias vector input/output sizes are not correct, "
		    "expected 1x%d [%s]\n", hidden_size, name.c_str());
    }
    else {
      if (bias_vector == 0) {
	bias_vector = new Connections(1, hidden_size);
	IncRef(bias_vector);
      }
      b = bias_vector;
    }
    bias_vector->countReference();
  }

  void ConvolutionBiasANNComponent::copyWeights(hash<string,Connections*> &weights_dict) {
    if (bias_vector == 0)
      ERROR_EXIT1(100, "Component not built, impossible execute copyWeights [%s]\n",
		  name.c_str());
    Connections *&b = weights_dict[weights_name];
    if (b != 0 && b != bias_vector)
      ERROR_EXIT2(101, "Weights dictionary contains %s bias name which is "
		  "not shared with bias_vector attribute [%s]\n",
		  weights_name.c_str(),
		  name.c_str());
    else if (b == 0) b = bias_vector;
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
