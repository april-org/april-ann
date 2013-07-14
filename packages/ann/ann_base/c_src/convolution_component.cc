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
#include "convolution_component.h"
#include "token_matrix.h"
#include "table_of_token_codes.h"

using april_utils::swap;

namespace ANN {

  ///////////////////////////////////////////
  // ConvolutionANNComponent implementation //
  ///////////////////////////////////////////

  void ConvolutionANNComponent::initializeArrays(const int *input_dims) {
    // input_dims[0]  => BUNCH SIZE
    // output_dims[0] => BUNCH SIZE, output_dims[1] => HIDDEN LAYER SIZE
    bias_rewrap[0]             = input_dims[0];
    output_dims[0]	       = input_dims[0];
    output_dims[1]	       = hidden_size;
    input_window_size[0]       = input_dims[0];
    input_window_size[1]       = input_dims[1];
    input_window_num_steps[0]  = 1;
    input_window_num_steps[1]  = 1;
    output_window_size[0]      = input_dims[0];
    // AT CONSTRUCTOR: output_window_size[1] = hidden_size;
    output_window_num_steps[0] = 1;
    output_window_num_steps[1] = 1;
    input_window_rewrap[0]     = input_dims[0];
    // AT CONSTRUCTOR: input_window_rewrap[1] = kernel_size;
    output_window_rewrap[0]    = input_dims[0];
    // AT CONSTRUCTOR: output_window_rewrap[1] = hidden_size;
    if (input_dims[1] != kernel_dims[1])
      ERROR_EXIT3(128, "Input matrix dim 1 must be equals to kernel dim 1,"
		  "input_dims[1]=%d, kernel_dims[1]=%d [%s]\n",
		  input_dims[1], kernel_dims[0], name.c_str());
    for (int i=2; i<=input_num_dims; ++i) {
      output_dims[i] = (input_dims[i] - kernel_dims[i])/kernel_step[i] + 1;
      
      input_window_size[i]	 = kernel_dims[i];
      input_window_num_steps[i]  = output_dims[i];
      output_window_num_steps[i] = output_dims[i];
    }
  }
  
  MatrixFloat *ConvolutionANNComponent::prepareBiasBunch() {
    // this line converts the bias matrix of Nx1 in a vector of N elements
    MatrixFloat *bias_vec = bias_vector->getPtr()->select(1,0);
    IncRef(bias_vec);
    MatrixFloat *bias_matrix_2d = new MatrixFloat(2, output_dims,
						  CblasColMajor);
    IncRef(bias_matrix_2d);
    for (int b=0; b<output_dims[0]; ++b) {
      MatrixFloat *dest = bias_matrix_2d->select(0, b);
      IncRef(dest);
      dest->copy(bias_vec);
      DecRef(dest);
    }
    MatrixFloat *bias_matrix = bias_matrix_2d->rewrap(bias_rewrap,
						      input_num_dims + 1);
    DecRef(bias_matrix_2d);
    DecRef(bias_vec);
    return bias_matrix;
  }

  ConvolutionANNComponent::ConvolutionANNComponent(int input_num_dims,
						   const int *_kernel_dims,
						   const int *_kernel_step,
						   int num_output_planes,
						   const char *name,
						   const char *weights_name,
						   const char *bias_name) :
    ANNComponent(name, weights_name, 0, 0),
    input(0),
    error_input(0),
    output(0),
    error_output(0),
    weights_matrix(0),
    bias_vector(0),
    num_updates_from_last_prune(0),
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
    output_window_rewrap(new int[2]),
    bias_rewrap(new int[input_num_dims+1]),
    learning_rate(-1.0f),
    momentum(0.0f),
    weight_decay(0.0f),
    c_weight_decay(1.0f),
    max_norm_penalty(-1.0f) {
    if (bias_name == 0) generateDefaultWeightsName(this->bias_name, "b");
    else this->bias_name = string(bias_name);
    if (weights_name == 0) generateDefaultWeightsName(this->weights_name, "w");
    kernel_dims[0] = static_cast<int>(hidden_size);
    kernel_step[0] = 1;
    input_window_order_step[0] = 0;
    output_window_size[0] = 0;
    output_window_size[1] = static_cast<int>(hidden_size);
    output_window_order_step[0] = 0;
    output_window_step[0] = 1;
    output_window_step[1] = 1;
    bias_rewrap[0] = 0;
    bias_rewrap[1] = static_cast<int>(hidden_size);
    for(int i=0; i<input_num_dims; ++i) {
      kernel_size *= _kernel_dims[i];
      kernel_dims[i+1] = _kernel_dims[i];
      kernel_step[i+1] = _kernel_step[i];
      input_window_order_step[i+1] = i+1;
      output_window_order_step[i+1] = i+1;
    }
    for(int i=2; i<=input_num_dims; ++i) {
      output_window_size[i] = 1;
      output_window_step[i] = 1;
      bias_rewrap[i] = 1;
    }
    input_window_rewrap[0]  = 0;
    input_window_rewrap[1]  = static_cast<int>(kernel_size);
    output_window_rewrap[0] = 0;
    output_window_rewrap[1] = static_cast<int>(hidden_size);
  }
  
  ConvolutionANNComponent::~ConvolutionANNComponent() {
    if (weights_matrix) DecRef(weights_matrix);
    if (bias_vector) DecRef(bias_vector);
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
    delete[] bias_rewrap;
  }
  
  // The ConvolutionANNComponent
  Token *ConvolutionANNComponent::doForward(Token *_input, bool during_training) {
    if (weights_matrix == 0) ERROR_EXIT1(129, "Not built component %s\n",
					 name.c_str());
    MatrixFloat *weights_mat = weights_matrix->getPtr();
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
    const int *input_dims = input_mat->getDimPtr();
    initializeArrays(input_dims);
    MatrixFloat *output_mat;
    output_mat = new MatrixFloat(input_num_dims+1, output_dims, CblasColMajor);
    AssignRef(output, new TokenMatrixFloat(output_mat));
#ifdef USE_CUDA
    output_mat->setUseCuda(use_cuda);
#endif
    
    MatrixFloat *bias_matrix = prepareBiasBunch();
    IncRef(bias_matrix);
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
    // CONVOLUTION OVER number_input_windows
    while(!input_sw.isEnd() && !output_sw.isEnd()) {
      MatrixFloat *input_w  = input_sw.getMatrix();
      MatrixFloat *output_w = output_sw.getMatrix();
      IncRef(input_w);
      IncRef(output_w);
      MatrixFloat *input_flattened  = getRewrappedMatrix(input_w,
							 input_window_rewrap,
							 2);
      MatrixFloat *output_flattened = getRewrappedMatrix(output_w,
							 output_window_rewrap,
							 2);
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
      // ADD BIAS
      output_w->axpy(1.0f, bias_matrix);
      
      // Next iteration
      input_sw.next();
      output_sw.next();
      
      // Free memory
      DecRef(input_flattened);
      DecRef(output_flattened);
      DecRef(input_w);
      DecRef(output_w);
    }
    DecRef(bias_matrix);
    return output;
  }
  
  Token *ConvolutionANNComponent::doBackprop(Token *_error_input) {
    // error checking
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input error Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(error_input,_error_input->convertTo<TokenMatrixFloat*>());
    return 0;

    /*
      MatrixFloat *error_input_mat = error_input->getMatrix();
      if (! error_input_mat->sameDim(output->getMatrix()) )
      ERROR_EXIT1(129, "Different bunches found at doForward and doBackprop [%s]\n",
      name.c_str());
      // new error output to fit the bunch
      ASSERT_MATRIX(error_input_mat);
      assert(error_input_mat->getDimSize(1) == static_cast<int>(output_size));
      if (error_input_mat->getStrideSize(0) > 1) {
      error_input_mat = error_input_mat->clone();
      AssignRef(error_input,new TokenMatrixFloat(error_input_mat));
      }
      #ifdef USE_CUDA
      error_input_mat->setUseCuda(use_cuda);
      #endif
      unsigned int bunch_size = error_input_mat->getDimSize(0);
      // new output to fit the bunch
      MatrixFloat *error_output_mat;
      int dims[2] = { static_cast<int>(bunch_size),
      static_cast<int>(input_size) };
      error_output_mat = new MatrixFloat(2, dims, CblasColMajor);
      AssignRef(error_output,new TokenMatrixFloat(error_output_mat));
      #ifdef USE_CUDA
      error_output_mat->setUseCuda(use_cuda);
      #endif      
      //
      MatrixFloat *weights_mat = weights_matrix->getPtr();
      if (bunch_size > 1) {
      // C = alpha * A * B + beta * C
      error_output_mat->gemm(CblasNoTrans, transpose_weights,
      1.0f, error_input_mat,
      weights_mat,
      0.0f);
      }
      else {
      error_output_mat->gemv(NEGATE_CBLAS_TRANSPOSE(transpose_weights),
      1.0f, weights_mat,
      error_input_mat,
      0.0f);
      }
      return error_output;
    */
  }
  
  void ConvolutionANNComponent::
  computeBPUpdateOnPrevVectors(MatrixFloat *prev_weights_mat,
			       Token *input_token,
			       MatrixFloat *error_input_mat,
			       float beta) {
    /*
      assert(error_input_mat->getDimSize(1) == static_cast<int>(output_size));
      unsigned int bunch_size = error_input_mat->getDimSize(0);
      // backprop learning rule:
      // PREV_W = alpha * ERRORS + PREV_W
      const unsigned int references = weights_matrix->getNumReferences();
      assert(references > 0 && "Found 0 references of weights matrix");
      // prev_w[i,j] = -learning_rate*1/sqrt(N*bsize) * ERROR_INPUT[j] + prev_w[i,j]
      const float norm_learn_rate =
      -(1.0f/sqrtf(static_cast<float>(references*bunch_size))) *
      learning_rate;
      if (sparse_input) {
      TokenBunchVector *input_vector_token;
      input_vector_token  = input_token->convertTo<TokenBunchVector*>();
      int w_dim = (transpose_weights == CblasNoTrans) ? 1 : 0;
      for (unsigned int b=0; b<bunch_size; ++b) {
      MatrixFloat *error_input_pat_mat;
      error_input_pat_mat = error_input_mat->select(0,static_cast<int>(b));
      Token *current = (*input_vector_token)[b];
      TokenSparseVectorFloat *sparse_token;
      sparse_token = current->convertTo<TokenSparseVectorFloat*>();
      for (unsigned int k=0; k<sparse_token->size(); ++k) {
      unsigned int pos     = (*sparse_token)[k].first;
      float value          = (*sparse_token)[k].second;
      int w_index          = static_cast<int>(pos);
      if (pos >= input_size)
      ERROR_EXIT1(128, "Overflow at sparse vector input pos [%s]\n",
      name.c_str());
      MatrixFloat *w_column = prev_weights_mat->select(w_dim, w_index);
      w_column->axpy(norm_learn_rate*value, error_input_pat_mat);
      delete w_column;
      }
      delete error_input_pat_mat;
      }
      } // if sparse_input ... else
      else {
      TokenMatrixFloat *input_mem_token=input_token->convertTo<TokenMatrixFloat*>();
      MatrixFloat *input_mat=input_mem_token->getMatrix();
      if (bunch_size > 1) {
      prev_weights_mat->gemm(CblasTrans, CblasNoTrans,
      norm_learn_rate,
      (transpose_weights == CblasNoTrans)?error_input_mat:input_mat, // A
      (transpose_weights == CblasNoTrans)?input_mat:error_input_mat, // B
      beta);
      } // if bunch_size > 1 ... else
      else {
      if (beta < 1.0f)
      prev_weights_mat->scal(beta);
      prev_weights_mat->ger(norm_learn_rate,
      (transpose_weights == CblasNoTrans)?error_input_mat:input_mat,
      (transpose_weights == CblasNoTrans)?input_mat:error_input_mat);
      } // if bunch_size > 1 ... else
      } // if sparse_input ... else
    */
  }
     
  // The ConvolutionANNComponent
  void ConvolutionANNComponent::doUpdate() {
    /*
      assert(learning_rate > 0.0f &&
      "Learning rate needs to be fixed with setOption method!!!");
      
      // Foces weights_matrix to update internal counts for a backward step
      weights_matrix->beginUpdate();
      
      MatrixFloat *weights_mat      = weights_matrix->getPtr();
      MatrixFloat *prev_weights_mat = weights_matrix->getPrevPtr();
      MatrixFloat *error_input_mat  = error_input->getMatrix();
    
      float beta_parameter_for_cblas_bp = 1.0f;
      if (weights_matrix->isFirstUpdateCall()) {
      // Momentum computation
      if (momentum > 0.0f) {
      // prev_w[i,j] = momentum * (w[i,j] - prev_w[i,j])
      weights_matrix->computeMomentumOnPrevVector(momentum,
      use_cuda);
      weights_matrix->computeWeightDecayOnPrevVector(c_weight_decay,
      use_cuda);
      }
      else {
      weights_matrix->copyToPrevVector(use_cuda);
      beta_parameter_for_cblas_bp = c_weight_decay;
      }
      } // if (weights_matrix->needsToComputeMomentum()) {
      
      computeBPUpdateOnPrevVectors(prev_weights_mat,
      input,
      error_input_mat,
      beta_parameter_for_cblas_bp);
      
      // Forces to update counts and swap vectors if necessary at this backward
      // step
      if (weights_matrix->endUpdate()) {
      ++num_updates_from_last_prune;
      if (num_updates_from_last_prune > MAX_UPDATES_WITHOUT_PRUNE) {
      num_updates_from_last_prune = 0;
      weights_matrix->pruneSubnormalAndCheckNormal();
      }
      if (max_norm_penalty > 0.0)
      weights_matrix->applyMaxNormPenalty(max_norm_penalty);
      }
    */
  }
  
  void ConvolutionANNComponent::reset() {
    if (input)        DecRef(input);
    if (error_input)  DecRef(error_input);
    if (output)       DecRef(output);
    if (error_output) DecRef(error_output);
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
  }
  
  ANNComponent *ConvolutionANNComponent::clone() {
    ConvolutionANNComponent *component = new
      ConvolutionANNComponent(input_num_dims, kernel_dims+1, kernel_step+1,
			      hidden_size,
			      name.c_str(), weights_name.c_str());
    component->input_size     = input_size;
    component->output_size    = output_size;
    component->learning_rate  = learning_rate;
    component->momentum       = momentum;
    component->weight_decay   = weight_decay;
    component->c_weight_decay = c_weight_decay;
    component->max_norm_penalty = max_norm_penalty;
    return component;
  }

  void ConvolutionANNComponent::setOption(const char *name, double value) {
    mSetOption(LEARNING_RATE_STRING, learning_rate);
    mSetOption(MOMENTUM_STRING,      momentum);
    if (strcmp(WEIGHT_DECAY_STRING, name) == 0) {
      weight_decay   = static_cast<float>(value);
      c_weight_decay = 1.0f - weight_decay;
      return;
    }
    mSetOption(MAX_NORM_PENALTY_STRING, max_norm_penalty);
    ANNComponent::setOption(name, value);
  }
  
  bool ConvolutionANNComponent::hasOption(const char *name) {
    mHasOption(LEARNING_RATE_STRING);
    mHasOption(MOMENTUM_STRING);
    mHasOption(WEIGHT_DECAY_STRING);
    mHasOption(MAX_NORM_PENALTY_STRING);
    return false;
  }
  
  double ConvolutionANNComponent::getOption(const char *name) {
    mGetOption(LEARNING_RATE_STRING, learning_rate);
    mGetOption(MOMENTUM_STRING, momentum);
    // the weight decay is always fixed to 0
    mGetOption(WEIGHT_DECAY_STRING, weight_decay);
    mGetOption(MAX_NORM_PENALTY_STRING, max_norm_penalty);
    return ANNComponent::getOption(name);
  }
  
  void ConvolutionANNComponent::build(unsigned int _input_size,
				     unsigned int _output_size,
				     hash<string,Connections*> &weights_dict,
				     hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    //
    unsigned int weights_input_size  = kernel_size;
    unsigned int weights_output_size = hidden_size;
    ////////////////////////////////////////////////////////////////////
    Connections *&w = weights_dict[weights_name];
    // printf("%s :: %p %p\n", weights_name.c_str(), w, weights_matrix);
    if (w != 0) {
      // printf("COPY OF WEIGHTS FROM HASH %s\n", weights_name.c_str());
      AssignRef(weights_matrix, w);
      if (!weights_matrix->checkInputOutputSizes(weights_input_size,
						 weights_output_size))
	ERROR_EXIT3(256,"The weights matrix input/output sizes are not correct, "
		    "expected %dx%d [%s]\n",
		    weights_input_size, weights_output_size,
		    name.c_str());
    }
    else {
      if (weights_matrix == 0) {
	// printf("NEW OF WEIGHTS %s\n", weights_name.c_str());
	weights_matrix = new Connections(weights_input_size,
					 weights_output_size);
	IncRef(weights_matrix);
      }
      // else printf("USING PREVIOUS WEIGHTS %s\n", weights_name.c_str());
      w = weights_matrix;
    }
    weights_matrix->countReference();
    /////////////////////////////////////////////////////////////////
    Connections *&b = weights_dict[bias_name];
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

  void ConvolutionANNComponent::copyWeights(hash<string,Connections*> &weights_dict) {
    if (weights_matrix == 0)
      ERROR_EXIT1(100, "Component not built, impossible execute copyWeights [%s]\n",
		  name.c_str());
    Connections *&w = weights_dict[weights_name];
    if (w != 0 && w != weights_matrix)
      ERROR_EXIT2(101, "Weights dictionary contains %s weights name which is "
		  "not shared with weights_matrix attribute [%s]\n",
		  weights_name.c_str(),
		  name.c_str());
    else if (w == 0) w = weights_matrix;
    Connections *&b = weights_dict[bias_name];
    if (b != 0 && b != bias_vector)
      ERROR_EXIT2(101, "Weights dictionary contains %s bias name which is "
		  "not shared with bias_vector attribute [%s]\n",
		  bias_name.c_str(),
		  name.c_str());
    else if (b == 0) b = bias_vector;
  }  

  char *ConvolutionANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.convolution{ name='%s',weights='%s',bias='%s',"
		  "n=%d, kernel={", name.c_str(), weights_name.c_str(),
		  bias_name.c_str(), hidden_size);
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
