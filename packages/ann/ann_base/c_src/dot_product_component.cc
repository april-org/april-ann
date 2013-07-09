/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
    if (weights_name == 0) generateDefaultWeightsName("w");
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
    if (weights_matrix == 0) ERROR_EXIT1(129, "Not built component %s\n",
					 name.c_str());
    MatrixFloat *weights_mat = weights_matrix->getPtr();
    // error checking
    if (_input == 0) ERROR_EXIT1(129,"Null Token received! [%s]\n",
				 name.c_str());
    // Three tokens are allowed: matrix, sparse, vector of sparse
    switch(_input->getTokenCode()) {
    case table_of_token_codes::token_matrix: {
      sparse_input = false;
      // change current input by new input
      AssignRef(input,_input);
      TokenMatrixFloat *input_mat_token=input->convertTo<TokenMatrixFloat*>();
      MatrixFloat *input_mat=input_mat_token->getMatrix();
      ASSERT_MATRIX(input_mat);
      assert(input_mat->getDimSize(1) == static_cast<int>(input_size));
      if (input_mat->getStrideSize(0) > 1) {
	input_mat = input_mat->clone();
	AssignRef(input,new TokenMatrixFloat(input_mat));
      }
#ifdef USE_CUDA
      input_mat->setUseCuda(use_cuda);
#endif
      unsigned int bunch_size = input_mat->getDimSize(0);
      // new output to fit the bunch
      MatrixFloat *output_mat;
      int dims[2] = { static_cast<int>(bunch_size),
		      static_cast<int>(output_size) };
      output_mat = new MatrixFloat(2, dims, CblasColMajor);
      AssignRef(output,new TokenMatrixFloat(output_mat));
#ifdef USE_CUDA
      output_mat->setUseCuda(use_cuda);
#endif
      if (bunch_size == 1) {
	// vector x matrix product
	output_mat->gemv(transpose_weights,
			 1.0f, weights_mat,
			 input_mat,
			 0.0f);
      } // if bunch_size==1
      else {
	// matrix x matrix product
	// C = \alpha op(A) op(B) + \beta C
	// input * weights = output
	output_mat->gemm(CblasNoTrans,
			 NEGATE_CBLAS_TRANSPOSE(transpose_weights),
			 1.0f, input_mat, weights_mat,
			 0.0f);
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
      assert(input_vector_token->size() > 0);
      // new output to fit the bunch
      MatrixFloat *output_mat;
      int dims[2] = {static_cast<int>(input_vector_token->size()),
		     static_cast<int>(output_size)};
      output_mat = new MatrixFloat(2, dims, CblasColMajor);
      AssignRef(output,new TokenMatrixFloat(output_mat));
#ifdef USE_CUDA
      output_mat->setUseCuda(use_cuda);
#endif      
      output_mat->zeros();
      unsigned int w_lda  = output_size;
      unsigned int w_step = 1;
      if (transpose_weights == CblasTrans) {
	w_lda  = 1;
	w_step = output_size;
      }
      
      // FIXME: Improve this code using sub-matrices instead of direct raw
      // access to matrix data
      FloatGPUMirroredMemoryBlock *output_ptr = output_mat->getRawDataAccess();
      FloatGPUMirroredMemoryBlock *weights_mat_ptr = weights_mat->getRawDataAccess();
      for (unsigned int b=0; b<input_vector_token->size(); ++b) {
	Token *current = (*input_vector_token)[b];
	if (current->getTokenCode()!=table_of_token_codes::vector_float_sparse)
	  ERROR_EXIT1(128,"Incorrect token type, expected vector_float_sparse [%s]\n",
		      name.c_str());
	TokenSparseVectorFloat *sparse_token;
	sparse_token = current->convertTo<TokenSparseVectorFloat*>();
	for (unsigned int k=0; k<sparse_token->size(); ++k) {
	  unsigned int pos     = (*sparse_token)[k].first;
	  float value          = (*sparse_token)[k].second;
	  unsigned int w_shift = pos*w_lda;
	  if (pos >= input_size)
	    ERROR_EXIT1(128, "Overflow at sparse vector input pos [%s]\n",
			name.c_str());
	  doSaxpy(output_size,
		  value,
		  weights_mat_ptr, w_shift, w_step,
		  output_ptr, b, input_vector_token->size(), use_cuda);
	}
      }
      break;
    }
    default:
      ERROR_EXIT2(128, "Incorrect token type: %d [%s]\n", _input->getTokenCode(),
		  name.c_str());
    };
    return output;
  }
  
  Token *DotProductANNComponent::doBackprop(Token *_error_input) {
    // error checking
    if ( (_error_input == 0) ||
	 (_error_input->getTokenCode() != table_of_token_codes::token_matrix))
      ERROR_EXIT1(129,"Incorrect input error Token type, expected token_matrix! [%s]\n",
		  name.c_str());
    // change current input by new input
    AssignRef(error_input,_error_input->convertTo<TokenMatrixFloat*>());
    if (sparse_input) {
      // If input is parse, the component needs to be an input of the ANN,
      // therefore the input is probably SO LARGE, and computing the backprop
      // will lead in HIGH computational cost ;) Because of this, the components
      // returns a NULL gradient pointer
      if (error_output) { DecRef(error_output); error_output = 0; }
      return 0;
    }
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
    output_mat->setUseCuda(use_cuda);
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
  }
  
  void DotProductANNComponent::
  computeBPUpdateOnPrevVectors(MatrixFloat *prev_weights_mat,
			       Token *input_token,
			       MatrixFloat *error_input_mat,
			       float beta) {
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
      unsigned int w_lda  = output_size;
      unsigned int w_step = 1;
      if (transpose_weights == CblasTrans) {
	w_lda  = 1;
	w_step = output_size;
      }
      FloatGPUMirroredMemoryBlock *error_input = error_input_mat->getRawDataAccess();
      FloatGPUMirroredMemoryBlock *prev_weights_mat_ptr = prev_weights_mat->getRawDataAccess();
      for (unsigned int b=0; b<bunch_size; ++b) {
	Token *current = (*input_vector_token)[b];
	TokenSparseVectorFloat *sparse_token;
	sparse_token = current->convertTo<TokenSparseVectorFloat*>();
	for (unsigned int k=0; k<sparse_token->size(); ++k) {
	  unsigned int pos     = (*sparse_token)[k].first;
	  float value          = (*sparse_token)[k].second;
	  unsigned int w_shift = pos*w_lda;
	  if (pos >= input_size)
	    ERROR_EXIT1(128, "Overflow at sparse vector input pos [%s]\n",
			name.c_str());
	  doSaxpy(output_size,
		  norm_learn_rate*value,
		  error_input, b, bunch_size,
		  prev_weights_mat_ptr, w_shift, w_step,
		  use_cuda);
	}
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
  }
  
  // The DotProductANNComponent
  void DotProductANNComponent::doUpdate() {
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
      if (max_norm_penalty > 0.0) {
	// we need to use the pointer because after endUpdate() method the
	// pointers will be swapped or changed
	weights_mat = weights_matrix->getPtr();
	MatrixFloat::sliding_window window(weights_mat, 0, 0, 0, 0, 0);
	while(!window.isEnd()) {
	  MatrixFloat *submat = window.getMatrix();
	  float norm2 = submat->norm2();
	  if (norm2 > max_norm_penalty) {
	    float scal_factor = max_norm_penalty/norm2;
	    submat->scal(scal_factor);
	  }
	  delete submat;
	  window.next();
	}
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
  
  bool DotProductANNComponent::hasOption(const char *name) {
    mHasOption(LEARNING_RATE_STRING);
    mHasOption(MOMENTUM_STRING);
    mHasOption(WEIGHT_DECAY_STRING);
    mHasOption(MAX_NORM_PENALTY_STRING);
    return false;
  }
  
  double DotProductANNComponent::getOption(const char *name) {
    mGetOption(LEARNING_RATE_STRING, learning_rate);
    mGetOption(MOMENTUM_STRING, momentum);
    // the weight decay is always fixed to 0
    mGetOption(WEIGHT_DECAY_STRING, weight_decay);
    mGetOption(MAX_NORM_PENALTY_STRING, max_norm_penalty);
    return ANNComponent::getOption(name);
  }
  
  void DotProductANNComponent::build(unsigned int _input_size,
				     unsigned int _output_size,
				     hash<string,Connections*> &weights_dict,
				     hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    //
    if (input_size == 0 || output_size == 0)
      ERROR_EXIT1(141, "Impossible to compute input/output "
		  "sizes for this component [%s]\n",
		  name.c_str());
    unsigned int weights_input_size  = input_size;;
    unsigned int weights_output_size = output_size;
    ////////////////////////////////////////////////////////////////////
    if (transpose_weights == CblasTrans)
      swap(weights_input_size, weights_output_size);
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
    // TODO: compute fan-in
    // outputs->increaseFanIn(inputs->numNeurons());
    weights_matrix->countReference();
  }

  void DotProductANNComponent::copyWeights(hash<string,Connections*> &weights_dict) {
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
  }  

  char *DotProductANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.dot_product{ name='%s',weights='%s',"
		  "input=%d,output=%d,transpose=%s }",
		  name.c_str(), weights_name.c_str(),
		  input_size, output_size,
		  (transpose_weights==CblasTrans)?"true":"false");
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
  //////////////////////////////////////////////////////////////////////////
}
