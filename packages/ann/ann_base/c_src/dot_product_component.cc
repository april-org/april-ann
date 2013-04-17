/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
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
#include "dot_product_action.h"
#include "trainsuper.h"

namespace ANN {

  ///////////////////////////////////////////
  // DotProductANNComponent implementation //
  ///////////////////////////////////////////
  
  DotProductANNComponent::DotProductANNComponent(const char *name,
						 const char *weights_name,
						 unsigned int input_size,
						 unsigned int output_size,
						 bool transpose_weights) :
    ANNComponent(input_size, output_size, name, weights_name),
    input(0), output(new TokenMemoryBlock()),
    error_input(0), error_output(new TokenMemoryBlock()),
    weights_matrix(0),
    learning_rate(-1.0f),
    momentum(0.0f),
    weight_decay(0.0f),
    c_weight_decay(1.0f),
    neuron_squared_length_upper_bound(-1.0f) {
    IncRef(output);
    IncRef(error_output);
    this->transpose_weights = (transpose_weights) ? CblasTrans : CblasNoTrans;
  }
  
  DotProductANNComponent::~DotProductANNComponent() {
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    DecRef(output);
    DecRef(error_output);
    if (weights_matrix) DecRef(weights_matrix);
  }
  
  // The DotProductANNComponent
  Token *DotProductANNComponent::doForward(Token *_input, bool during_training) {
    assert(weights_matrix != 0);
    // error checking
    if ( (_input == 0) ||
	 (_input->getTokenCode() != table_of_token_codes::token_mem_block))
      ERROR_EXIT(129,"Incorrect input Token type, expected token_mem_block!\n");
    // change current input by new input
    if (input) DecRef(input);
    input = _input->convertTo<TokenMemoryBlock*>();
    IncRef(input);
    // compute current bunch
    unsigned int bunch_size = input->getUsedSize() / input_size;
    this->bunch_size = bunch_size;
    // and resize the output to fit the bunch
    output->resize(bunch_size * output_size);
    // get memory blocks for tokens and weights
    FloatGPUMirroredMemoryBlock *input_ptr       = input->getMemBlock();
    FloatGPUMirroredMemoryBlock *output_ptr      = output->getMemBlock();
    FloatGPUMirroredMemoryBlock *weights_mat_ptr = weights_matrix->getPtr();
    // weights factor depends on dropout parameter
    float weights_factor = 1.0f;
    if (!during_training) weights_factor = 1.0f - inputs->drop_factor;
    //
    if (bunch_size == 1) {
      // vector x matrix product
      doSgemv(CblasColMajor, transpose_weights,
	      weights_matrix->getOutputSize(), weights_matrix->getInputSize(),
	      weights_factor, weights_mat_ptr, weights_matrix->getOutputSize(),
	      input_ptr, bunch_size, // conf.max_bunch_size
	      1.0f, output_ptr, bunch_size, // conf.max_bunch_size,
	      0, 0, 0, // inputs->getOffset(), outputs->getOffset()
	      use_cuda);
    } // if bunch_size==1
    else {
      // matrix x matrix product
      // C = \alpha op(A) op(B) + \beta C
      // input * weights = output
      doSgemm(CblasColMajor,
	      CblasNoTrans, NEGATE_CBLAS_TRANSPOSE(transpose_weights),
	      bunch_size, output_size, input_size, // conf.cur_bunch_size,
	      weights_factor, input_ptr, bunch_size, // conf.max_bunch_size,
	      weights_mat_ptr, weights_matrix->getOutputSize(),
	      // beta = 1.0f, C matrix contains BIAS and probably other layer
	      // computations
	      1.0f, output_ptr, bunch_size, // conf.max_bunch_size,
	      0, 0, 0, // inputs->getOffset(), 0, outputs->getOffset(),
	      use_cuda); // conf.use_cuda_flag);
    } // if bunch_size==1 ... else
    return output;
  }
  
  Token *DotProductANNComponent::doBackprop(Token *_error_input) {
    // error checking
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_mem_block))
      ERROR_EXIT(129,"Incorrect input error Token type, expected token_mem_block!\n");
    // change current input by new input
    if (error_input) DecRef(error_input);
    error_input = _error_input->convertTo<TokenMemoryBlock*>();
    IncRef(error_input);
    // compute current bunch
    unsigned int bunch_size = error_input->getUsedSize() / output_size;
    if (bunch_size != this->bunch_size)
      ERROR_EXIT(129, "Different bunches found at doForward and doBackprop\n");
    // and resize the output to fit the bunch
    error_output->resize(bunch_size * input_size);
    //
    FloatGPUMirroredMemoryBlock *error_input_ptr = error_input->getMemBlock();
    FloatGPUMirroredMemoryBlock *weights_mat_ptr = weights_matrix->getPtr();
    if (bunch_size > 1) {
      // C = alpha * A * B + beta * C
      doSgemm(CblasColMajor,
	      CblasNoTrans, NEGATE_CBLAS_TRANSPOSE(transpose_weights),
	      bunch_size, num_inputs, num_outputs, // conf.cur_bunch_size,
	      1.0f, error_input_ptr, bunch_size, // conf.max_bunch_size,
	      weights_mat_ptr, weights_matrix->getOutputSize(),
	      1.0f, error_output, bunch_size, // conf.max_bunch_size,
	      0, 0, 0, // error_input_shift, 0, error_output_shift,
	      use_cuda);
    }
    else {
      doSgemv(CblasColMajor, NEGATE_CBLAS_TRANSPOSE(transpose_weights),
	      weights_matrix->getOutputSize(),
	      weights_matrix->getInputSize(),
	      1.0f, weights_mat_ptr, weights_matrix->getOutputSize(),
	      error_input_ptr, bunch_size, // conf.max_bunch_size,
	      1.0f, error_output, bunch_size, // conf.max_bunch_size,
	      0, 0, 0, // 0, error_input_shift, error_output_shift,
	      use_cuda);
    }
    return error_output;
  }
  
  void DotProductANNComponent::
  computeBPUpdateOnPrevVectors(FloatGPUMirroredMemoryBlock *prev_weights_mat_ptr,
			       FloatGPUMirroredMemoryBlock *input,
			       const unsigned int input_shift,
			       FloatGPUMirroredMemoryBlock *error_input,
			       const unsigned int error_input_shift,
			       float beta) {
    // backprop learning rule:
    // PREV_W = alpha * ERRORS + PREV_W
    const unsigned int references = weights_matrix->getNumReferences();
    // prev_w[i,j] = -learning_rate*1/sqrt(N*bsize) * ERROR_INPUT[j] + prev_w[i,j]
    const float norm_learn_rate =
      -(1.0f/sqrtf(static_cast<float>(references*conf.cur_bunch_size))) *
      learning_rate;
    if (bunch_size > 1) {
      doSgemm(CblasColMajor, CblasTrans, CblasNoTrans,
	      weights_matrix->getOutputSize()
	      weights_matrix->getInputSize(),
	      bunch_size, // conf.cur_bunch_size, // dimensiones
	      norm_learn_rate,                          // alpha
	      (transpose_weights == CblasNoTrans)?error_input:input,                              // A
	      bunch_size, // conf.max_bunch_size,                      // A stride
	      (transpose_weights == CblasNoTrans)?input:error_input,                                    // B
	      bunch_size,                      // B stride
	      beta,                                     // beta
	      prev_weights_mat_ptr,                     // C
	      weights_matrix->getOutputSize(),                              // C stride
	      0, 0, 0, // input_shift, 0,        // desplazamientos
	      use_cuda);
    } // if bunch_size > 1 ... else
    else {
      if (beta < 1.0f)
	doSscal(weights_matrix->getNumWeights(),
		beta, prev_weights_mat_ptr, 0, 1,
		use_cuda);
      doSger(CblasColMajor,
	     weights_matrix->getOutputSize(),
	     weights_matrix->getInputSize(),
	     norm_learn_rate,
	     (transpose_weights == CblasNoTrans)?error_input:input, 0, bunch_size, // error_input_shift, conf.max_bunch_size,
	     (transpose_weights == CblasNoTrans)?input:error_input, 0, bunch_size, // input_shift, conf.max_bunch_size,
	     prev_weights_mat_ptr, 0, weights_matrix->getOutputSize(),
	     use_cuda);
    } // if bunch_size > 1 ... else
  }
  
  // The DotProductANNComponent
  void DotProductANNComponent::doUpdate() {
    assert(learning_rate > 0.0f &&
	   "Learning rate needs to be fixed with setOption method!!!");
    
    // Foces weights_matrix to update internal counts for a backward step
    weights_matrix->beginUpdate();
    
    FloatGPUMirroredMemoryBlock *weights_mat_ptr = weights_matrix->getPtr();
    FloatGPUMirroredMemoryBlock *prev_weights_mat_ptr =
      weights_matrix->getPrevPtr();
    FloatGPUMirroredMemoryBlock *input        = inputs->getMemBlock();
    FloatGPUMirroredMemoryBlock *error_input  = outputs->getMemBlock();
    FloatGPUMirroredMemoryBlock *error_output = inputs->getMemBlock();
    
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
    
    computeBPUpdateOnPrevVectors(prev_weights_mat_ptr,
				 input, 0, // input_shift,
				 error_input, 0, // output_shift,
				 beta_parameter_for_cblas_bp);
    
    // Forces to update counts and swap vectors if necessary at this backward
    // step
    if (weights_matrix->endUpdate()) {
      // TODO: max norm penalty
    }
  }
  
  void DotProductANNComponent::reset() {
    if (error_output != 0) doVectorSetToZero(error_output->getMemBlock(),
					     error_output->getMaxSize(),
					     0, 0, use_cuda);
    if (output != 0) doVectorSetToZero(output->getMemBlock(),
				       output->getMaxSize(),
				       0, 0, use_cuda);
    if (input) DecRef(input); input = 0;
    if (error_input) DecRef(error_input); error_input = 0;
  }

  ANNComponent *DotProductANNComponent::clone() {
    DotProductANNComponent *component = new
      DotProductANNComponent(name, weights_name,
			     input_size, output_size,
			     (transpose_weights == CblasTrans));
    component->learning_rate  = learning_rate;
    component->momentum       = momentum;
    component->weight_decay   = weight_decay;
    component->c_weight_decay = c_weight_decay;
    component->neuron_squared_length_upper_bound = neuron_squared_length_upper_bound;
    return component;
  }

  void DotProductANNComponent::setOption(const char *name, double value) {
    mSetOption("learning_rate", learning_rate);
    mSetOption("momentum", momentum);
    if (strcmp("weight_decay", name) == 0) {
      weight_decay   = static_cast<float>(value);
      c_weight_decay = 1.0f - weight_decay;
      return;
    }
    mSetOption("neuron_squared_length_upper_bound",
	       neuron_squared_length_upper_bound);
    ERROR_EXIT1(140, "The option to be set does not exist: %s.\n", name);
  }
  
  bool DotProductANNComponent::hasOption(const char *name) {
    mHasOption("learning_rate");
    mHasOption("momentum");
    mHasOption("weight_decay");
    mHasOption("neuron_squared_length_upper_bound");
    return false;
  }
  
  double DotProductANNComponent::getOption(const char *name) {
    mGetOption("learning_rate", learning_rate);
    mGetOption("momentum", momentum);
    // the weight decay is always fixed to 0
    mGetOption("weight_decay", weight_decay);
    mGetOption("neuron_squared_length_upper_bound", neuron_squared_length_upper_bound);
    return ANNComponent::getOption(name);
  }
  
  void DotProductANNComponent::build(unsigned int _input_size,
				     unsigned int _output_size,
				     hash<string,Connections*> &weights_dict,
				     hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size, weights_dict, components_dict);
    //
    if (input_size == 0 || output_size == 0)
      ERROR_EXIT(141, "Impossible to compute input/output "
		 "sizes for this component\n");
    unsigned int weights_input_size  = input_size;;
    unsigned int weights_output_size = output_size;
    ////////////////////////////////////////////////////////////////////
    if (weights_matrix != 0) DecRef(weights_matrix);
    if (transpose_weights) swap(weights_input_size, weights_output_size)
    Connections *&w = weights_dict[weights_name];
    if (w != 0) {
      weights_matrix = w;
      if (!weights_matrix->checkInputOutputSizes(weights_input_size,
						 weights_output_size))
	ERROR_EXIT2(256,"The weights matrix input/output sizes are not correct, "
		    "expected %d,%d.\n",
		    weights_input_size, weights_output_size);
    }
    else {
      weights_matrix = new Connections(weights_input_size,
				       weights_output_size);
      w = weights_matrix;
    }
    // TODO: compute fan-in
    // outputs->increaseFanIn(inputs->numNeurons());
    weights_matrixr->countReference();
    IncRef(weights_matrix);
  }

  void copyWeights(hash<string,Connections*> &weights_dict) {
    if (weights_matrix == 0)
      ERROR_EXIT(100, "Component not built, impossible execute copyWeights\n");
    Connections *&w = weights_dict[weights_name];
    if (w != 0 && w != weights_matrix)
      ERROR_EXIT1(101, "Weights dictionary contains %s weights name which is "
		  "not shared with weights_matrix attribute\n",
		  weights_name.c_str());
    else if (w == 0) w = weights_matrix;
  }  
  //////////////////////////////////////////////////////////////////////////
}
