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
#include "swap.h"
#include "dot_product_component.h"
#include "wrapper.h"
#include "token_base.h"
#include "token_vector.h"
#include "token_memory_block.h"
#include "table_of_token_codes.h"

using april_utils::swap;

namespace ANN {

  ///////////////////////////////////////////
  // DotProductANNComponent implementation //
  ///////////////////////////////////////////
  
  DotProductANNComponent::DotProductANNComponent(const char *name,
						 const char *weights_name,
						 unsigned int input_size,
						 unsigned int output_size,
						 bool transpose_weights) :
    ANNComponent(name, weights_name, input_size, output_size),
    input(0),
    error_input(0),
    output(0),
    error_output(0),
    weights_matrix(0),
    num_updates_from_last_prune(0),
    learning_rate(-1.0f),
    momentum(0.0f),
    weight_decay(0.0f),
    c_weight_decay(1.0f),
    max_norm_penalty(-1.0f) {
    if (weights_name == 0) generateDefaultWeightsName();
    this->transpose_weights = (transpose_weights) ? CblasTrans : CblasNoTrans;
  }
  
  DotProductANNComponent::~DotProductANNComponent() {
    if (weights_matrix) DecRef(weights_matrix);
    if (input) DecRef(input);
    if (error_input) DecRef(error_input);
    if (output) DecRef(output);
    if (error_output) DecRef(error_output);
  }
  
  // The DotProductANNComponent
  Token *DotProductANNComponent::doForward(Token *_input, bool during_training) {
    assert(weights_matrix != 0);
    FloatGPUMirroredMemoryBlock *weights_mat_ptr = weights_matrix->getPtr();
    // error checking
    if (_input == 0)
      ERROR_EXIT(129,"Null Token received!\n");
    switch(_input->getTokenCode()) {
    case table_of_token_codes::token_mem_block: {
      sparse_input = false;
      // change current input by new input
      AssignRef(input,_input);
      TokenMemoryBlock *input_mem_token=input->convertTo<TokenMemoryBlock*>();
      // compute current bunch
      bunch_size = input_mem_token->getUsedSize() / input_size;
      if (input_mem_token->getUsedSize() % input_size != 0)
	ERROR_EXIT2(128, "Input memory block (size %d) is not multiple of %d\n",
		    input_mem_token->getUsedSize(), input_size);
      // new output to fit the bunch
      AssignRef(output,new TokenMemoryBlock(bunch_size * output_size));
      // get memory blocks for tokens and weights
      FloatGPUMirroredMemoryBlock *input_ptr       = input_mem_token->getMemBlock();
      FloatGPUMirroredMemoryBlock *output_ptr      = output->getMemBlock();
      //
      if (bunch_size == 1) {
	// vector x matrix product
	doSgemv(CblasColMajor, transpose_weights,
		weights_matrix->getOutputSize(), weights_matrix->getInputSize(),
		1.0f, weights_mat_ptr, weights_matrix->getOutputSize(),
		input_ptr, 1,
		0.0f, output_ptr, 1,
		0, 0, 0,
		use_cuda);
      } // if bunch_size==1
      else {
	// matrix x matrix product
	// C = \alpha op(A) op(B) + \beta C
	// input * weights = output
	doSgemm(CblasColMajor,
		CblasNoTrans, NEGATE_CBLAS_TRANSPOSE(transpose_weights),
		bunch_size, output_size, input_size, // conf.cur_bunch_size,
		1.0f, input_ptr, bunch_size, // conf.max_bunch_size,
		weights_mat_ptr, weights_matrix->getOutputSize(),
		// beta = 1.0f, C matrix contains BIAS and probably other layer
		// computations
		0.0f, output_ptr, bunch_size, // conf.max_bunch_size,
		0, 0, 0, // inputs->getOffset(), 0, outputs->getOffset(),
		use_cuda); // conf.use_cuda_flag);
      } // if bunch_size==1 ... else
      break;
    }
    case table_of_token_codes::vector_float_sparse: {
      TokenBunchVector *aux = new TokenBunchVector();
      aux->push_back(_input);
      _input = aux; // is not necessary to do incref(aux) or decref(_input)
      // the incref is done at line 127

      // continues in the next case
    }
    case table_of_token_codes::vector_Tokens: {
      sparse_input = true;
      AssignRef(input, _input);
      TokenBunchVector *input_vector_token=input->convertTo<TokenBunchVector*>();
      bunch_size = input_vector_token->size();
      if (bunch_size == 0) ERROR_EXIT(128, "Found bunch_size==0\n");
      AssignRef(output,new TokenMemoryBlock(bunch_size * output_size));
      output->setToZero(use_cuda);
      FloatGPUMirroredMemoryBlock *output_ptr = output->getMemBlock();
      unsigned int w_lda  = output_size;
      unsigned int w_step = 1;
      if (transpose_weights == CblasTrans) {
	w_lda  = 1;
	w_step = output_size;
      }
      for (unsigned int b=0; b<bunch_size; ++b) {
	Token *current = (*input_vector_token)[b];
	if (current->getTokenCode()!=table_of_token_codes::vector_float_sparse)
	  ERROR_EXIT(128,"Incorrect token type, expected vector_float_sparse\n");
	TokenSparseVectorFloat *sparse_token;
	sparse_token = current->convertTo<TokenSparseVectorFloat*>();
	for (unsigned int k=0; k<sparse_token->size(); ++k) {
	  unsigned int pos     = (*sparse_token)[k].first;
	  float value          = (*sparse_token)[k].second;
	  unsigned int w_shift = pos*w_lda;
	  if (pos >= input_size)
	    ERROR_EXIT(128, "Overflow at sparse vector input pos\n");
	  doSaxpy(output_size,
		  value,
		  weights_mat_ptr, w_shift, w_step,
		  output_ptr, b, bunch_size, use_cuda);
	}
      }
      break;
    }
    default:
      ERROR_EXIT1(128, "Incorrect token type: %d\n", _input->getTokenCode());
    };
    return output;
  }
  
  Token *DotProductANNComponent::doBackprop(Token *_error_input) {
    // error checking
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_mem_block))
      ERROR_EXIT(129,"Incorrect input error Token type, expected token_mem_block!\n");
    // change current input by new input
    AssignRef(error_input,_error_input->convertTo<TokenMemoryBlock*>());
    if (sparse_input) {
      // If input is parse, the component needs to be an input of the ANN,
      // therefore the input is probably SO LARGE, and computing the backprop
      // will lead in HIGH computational cost ;) Because of this, the components
      // returns a NULL gradient pointer
      if (error_output) { DecRef(error_output); error_output = 0; }
      return 0;
    }
    // compute current bunch
    unsigned int bunch_size = error_input->getUsedSize() / output_size;
    if (bunch_size != this->bunch_size)
      ERROR_EXIT(129, "Different bunches found at doForward and doBackprop\n");
    // new error output to fit the bunch
    AssignRef(error_output,new TokenMemoryBlock(bunch_size * input_size));
    //
    FloatGPUMirroredMemoryBlock *error_input_ptr  = error_input->getMemBlock();
    FloatGPUMirroredMemoryBlock *error_output_ptr = error_output->getMemBlock();
    FloatGPUMirroredMemoryBlock *weights_mat_ptr  = weights_matrix->getPtr();
    if (bunch_size > 1) {
      // C = alpha * A * B + beta * C
      doSgemm(CblasColMajor,
	      CblasNoTrans, transpose_weights,
	      bunch_size, input_size, output_size,
	      1.0f, error_input_ptr, bunch_size,
	      weights_mat_ptr, weights_matrix->getOutputSize(),
	      0.0f, error_output_ptr, bunch_size,
	      0, 0, 0,
	      use_cuda);
    }
    else {
      doSgemv(CblasColMajor, NEGATE_CBLAS_TRANSPOSE(transpose_weights),
	      weights_matrix->getOutputSize(),
	      weights_matrix->getInputSize(),
	      1.0f, weights_mat_ptr, weights_matrix->getOutputSize(),
	      error_input_ptr, 1,
	      0.0f, error_output_ptr, 1,
	      0, 0, 0,
	      use_cuda);
    }
    return error_output;
  }
  
  void DotProductANNComponent::
  computeBPUpdateOnPrevVectors(FloatGPUMirroredMemoryBlock *prev_weights_mat_ptr,
			       Token *input_token,
			       FloatGPUMirroredMemoryBlock *error_input,
			       float beta) {
    // backprop learning rule:
    // PREV_W = alpha * ERRORS + PREV_W
    const unsigned int references = weights_matrix->getNumReferences();
    // prev_w[i,j] = -learning_rate*1/sqrt(N*bsize) * ERROR_INPUT[j] + prev_w[i,j]
    const float norm_learn_rate =
      -(1.0f/sqrtf(static_cast<float>(references*bunch_size))) *
      learning_rate;
    if (sparse_input) {
      TokenBunchVector *input_vector_token;
      input_vector_token  = input_token->convertTo<TokenBunchVector*>();
      unsigned int w_lda  = output_size;
      unsigned int w_step = 1;
      if (transpose_weights == CblasTrans) {
	w_lda  = 1;
	w_step = output_size;
      }
      for (unsigned int b=0; b<bunch_size; ++b) {
	Token *current = (*input_vector_token)[b];
	TokenSparseVectorFloat *sparse_token;
	sparse_token = current->convertTo<TokenSparseVectorFloat*>();
	for (unsigned int k=0; k<sparse_token->size(); ++k) {
	  unsigned int pos     = (*sparse_token)[k].first;
	  float value          = (*sparse_token)[k].second;
	  unsigned int w_shift = pos*w_lda;
	  if (pos >= input_size)
	    ERROR_EXIT(128, "Overflow at sparse vector input pos\n");
	  doSaxpy(output_size,
		  norm_learn_rate*value,
		  error_input, b, bunch_size,
		  prev_weights_mat_ptr, w_shift, w_step,
		  use_cuda);
	}
      }
    } // if sparse_input
    else {
      TokenMemoryBlock *input_mem_token=input_token->convertTo<TokenMemoryBlock*>();
      FloatGPUMirroredMemoryBlock *input=input_mem_token->getMemBlock();
      if (bunch_size > 1) {
	doSgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		weights_matrix->getOutputSize(),
		weights_matrix->getInputSize(),
		bunch_size,                               // dimensiones
		norm_learn_rate,                          // alpha
		(transpose_weights == CblasNoTrans)?error_input:input, // A
		bunch_size,                                            // A stride
		(transpose_weights == CblasNoTrans)?input:error_input, // B
		bunch_size,                                            // B stride
		beta,                                     // beta
		prev_weights_mat_ptr,                     // C
		weights_matrix->getOutputSize(),          // C stride
		0, 0, 0,                                  // offsets
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
	       (transpose_weights == CblasNoTrans)?error_input:input, 0, bunch_size,
	       (transpose_weights == CblasNoTrans)?input:error_input, 0, bunch_size,
	       prev_weights_mat_ptr, 0, weights_matrix->getOutputSize(),
	       use_cuda);
      } // if bunch_size > 1 ... else
    } // if sparse_input ... else
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
    FloatGPUMirroredMemoryBlock *error_input_ptr  = error_input->getMemBlock();
    
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
				 input,
				 error_input_ptr,
				 beta_parameter_for_cblas_bp);
    
    // Forces to update counts and swap vectors if necessary at this backward
    // step
    if (weights_matrix->endUpdate()) {
      ++num_updates_from_last_prune;
      if (num_updates_from_last_prune > MAX_UPDATES_WITHOUT_PRUNE) {
	num_updates_from_last_prune = 0;
	weights_matrix->pruneSubnormalAndCheckNormal();
      }
      if (max_norm_penalty > 0.0) {
	for (unsigned int i=0; i<output_size; ++i) {
	  float norm2 = doSnrm2(input_size,
				weights_matrix->getPtr(), i, output_size,
				use_cuda);
	  if (norm2 > max_norm_penalty) {
	    float scal = max_norm_penalty/norm2;
	    doSscal(input_size, scal,
		    weights_matrix->getPtr(), i, output_size,
		    use_cuda);
	  } // if norm2 > max_norm_penalty
	} // for (i=0; i<output_size; ++i)
      } // if max_norm_penalty > 0.0
    }
  }
  
  void DotProductANNComponent::reset() {
    if (input)        DecRef(input);
    if (error_input)  DecRef(error_input);
    if (output)       DecRef(output);
    if (error_output) DecRef(error_output);
    input	 = 0;
    error_input	 = 0;
    output	 = 0;
    error_output = 0;
  }
  
  ANNComponent *DotProductANNComponent::clone() {
    DotProductANNComponent *component = new
      DotProductANNComponent(name.c_str(), weights_name.c_str(),
			     input_size, output_size,
			     (transpose_weights == CblasTrans));
    component->input_size     = input_size;
    component->output_size    = output_size;
    component->learning_rate  = learning_rate;
    component->momentum       = momentum;
    component->weight_decay   = weight_decay;
    component->c_weight_decay = c_weight_decay;
    component->max_norm_penalty = max_norm_penalty;
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
    mSetOption("max_norm_penalty",
	       max_norm_penalty);
    ERROR_EXIT1(140, "The option to be set does not exist: %s.\n", name);
  }
  
  bool DotProductANNComponent::hasOption(const char *name) {
    mHasOption("learning_rate");
    mHasOption("momentum");
    mHasOption("weight_decay");
    mHasOption("max_norm_penalty");
    return false;
  }
  
  double DotProductANNComponent::getOption(const char *name) {
    mGetOption("learning_rate", learning_rate);
    mGetOption("momentum", momentum);
    // the weight decay is always fixed to 0
    mGetOption("weight_decay", weight_decay);
    mGetOption("max_norm_penalty", max_norm_penalty);
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
    if (transpose_weights == CblasTrans)
      swap(weights_input_size, weights_output_size);
    Connections *&w = weights_dict[weights_name];
    if (w != 0) {
      AssignRef(weights_matrix, w);
      if (!weights_matrix->checkInputOutputSizes(weights_input_size,
						 weights_output_size))
	ERROR_EXIT2(256,"The weights matrix input/output sizes are not correct, "
		    "expected %d,%d.\n",
		    weights_input_size, weights_output_size);
    }
    else {
      if (weights_matrix == 0) {
	weights_matrix = new Connections(weights_input_size,
					 weights_output_size);
	IncRef(weights_matrix);
      }
      w = weights_matrix;
    }
    // TODO: compute fan-in
    // outputs->increaseFanIn(inputs->numNeurons());
    weights_matrix->countReference();
  }

  void DotProductANNComponent::copyWeights(hash<string,Connections*> &weights_dict) {
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
