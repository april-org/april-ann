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

  /////////////////////////////////////
  // DotProductAction implementation //
  /////////////////////////////////////
  
  DotProductAction::DotProductAction(const ANNConfiguration &conf,
				     ActivationUnits *inputs,
				     ActivationUnits *outputs,
				     Connections *weights_matrix,
				     bool transpose_weights) :
    Action(conf),
    inputs(inputs), outputs(outputs), weights_matrix(weights_matrix),
    num_inputs(inputs->numNeurons()),
    num_outputs(outputs->numNeurons()),
    conf(conf),
    learning_rate(-1.0f),
    momentum(0.0f),
    weight_decay(0.0f),
    c_weight_decay(1.0f),
    neuron_squared_length_upper_bound(-1.0f),
    transpose_weights(transpose_weights) {
    if (!transpose_weights) {
      if (!weights_matrix->checkInputOutputSizes(inputs, outputs))
	ERROR_EXIT(256, "The input/output sizes are not correct.\n");
      outputs->increaseFanIn(inputs->numNeurons());
    }
    else
      if (!weights_matrix->checkInputOutputSizes(outputs, inputs)) {
	ERROR_EXIT(256, "The input/output sizes are not correct.\n");
	inputs->increaseFanIn(outputs->numNeurons());
      }
    weights_matrix->countReference();
    IncRef(inputs);
    IncRef(outputs);
    IncRef(weights_matrix);
  }
  
  DotProductAction::~DotProductAction() {
    DecRef(inputs);
    DecRef(outputs);
    DecRef(weights_matrix);
  }
  
  // The DotProductAction
  void DotProductAction::doForward(bool during_training) {
    FloatGPUMirroredMemoryBlock *input_ptr       = inputs->getPtr();
    FloatGPUMirroredMemoryBlock *output_ptr      = outputs->getPtr();
    FloatGPUMirroredMemoryBlock *weights_mat_ptr = weights_matrix->getPtr();
    float weights_factor = 1.0f;
    if (!during_training) weights_factor = 1.0f - inputs->drop_factor;
    
    // if input is sparse, then zero input values are not multiplied
    if (inputs->isSparse()) {
      if (!transpose_weights) {
	const float *input_float_ptr = input_ptr->getPPALForRead();
	unsigned int w_shift = 0;
	for (unsigned int i=0; i<num_inputs; ++i, w_shift+=num_outputs) {
	  for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	    float v = input_float_ptr[b];
	    if ( v != 0.0f ) {
	      doSaxpy(num_outputs,
		      weights_factor*v,
		      weights_mat_ptr, w_shift, 1,
		      output_ptr, b, conf.max_bunch_size, conf.use_cuda_flag);
	    }
	  }
	  input_float_ptr += conf.max_bunch_size;
	}
      } // if !transposed weights
      else {
	const float *input_float_ptr = input_ptr->getPPALForRead();
	for (unsigned int i=0; i<num_inputs; ++i) {
	  for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	    float v = input_float_ptr[b];
	    if ( v != 0.0f ) {
	      doSaxpy(num_outputs,
		      weights_factor*v,
		      weights_mat_ptr, i, num_outputs,
		      output_ptr, b, conf.max_bunch_size, conf.use_cuda_flag);
	    }
	  }
	  input_float_ptr += conf.max_bunch_size;
	}
      } // if !transposed weights .. else
    } // if isSparse
    else {
      if (conf.cur_bunch_size == 1) {
	// vector x matrix product
	if (!transpose_weights) {
	  doSgemv(CblasColMajor, CblasNoTrans,
		  num_outputs, num_inputs,
		  weights_factor, weights_mat_ptr, num_outputs,
		  input_ptr, conf.max_bunch_size,
		  1.0f, output_ptr, conf.max_bunch_size,
		  0, inputs->getOffset(), outputs->getOffset(),
		  conf.use_cuda_flag);
	} // if !transposed weights
	else {
	  doSgemv(CblasColMajor, CblasTrans,
		  num_inputs, num_outputs,
		  weights_factor, weights_mat_ptr, num_inputs,
		  input_ptr, conf.max_bunch_size,
		  1.0f, output_ptr, conf.max_bunch_size,
		  0, inputs->getOffset(), outputs->getOffset(),
		  conf.use_cuda_flag);
	} // if !transposed weights ... else
      } // if bunch_size==1
      else {
	// matrix x matrix product
	// C = \alpha op(A) op(B) + \beta C
	// input * weights = output
	if (!transpose_weights) {
	  doSgemm(CblasColMajor, CblasNoTrans, CblasTrans,
		  conf.cur_bunch_size, num_outputs, num_inputs,
		  weights_factor, input_ptr, conf.max_bunch_size,
		  weights_mat_ptr, num_outputs,
		  // beta = 1.0f, C matrix contains BIAS and probably other layer
		  // computations
		  1.0f, output_ptr, conf.max_bunch_size,
		  inputs->getOffset(), 0, outputs->getOffset(),
		  conf.use_cuda_flag);
	} // if !transposed weights
	else {
	  doSgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		  conf.cur_bunch_size, num_outputs, num_inputs,
		  weights_factor,
		  input_ptr, conf.max_bunch_size,
		  weights_mat_ptr, num_inputs,
		  // beta = 1.0f, C matrix contains BIAS and probably other layer
		  // computations
		  1.0f, output_ptr, conf.max_bunch_size,
		  inputs->getOffset(), 0, outputs->getOffset(),
		  conf.use_cuda_flag);
	} // if !transposed weights ... else
      } // if bunch_size==1 ... else
    } // if isSparse ... else
  }

  void DotProductAction::doBackprop() {
    FloatGPUMirroredMemoryBlock *output_error = inputs->getErrorVectorPtr();
    if (output_error != 0) {
      const unsigned int output_error_shift        = inputs->getOffset();
      FloatGPUMirroredMemoryBlock *input_error     = outputs->getErrorVectorPtr();
      const unsigned int input_error_shift         = outputs->getOffset();
      FloatGPUMirroredMemoryBlock *weights_mat_ptr = weights_matrix->getPtr();
      if (conf.cur_bunch_size > 1) {
	// C = alpha * A * B + beta * C
	if (!transpose_weights)
	  doSgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		  conf.cur_bunch_size, num_inputs, num_outputs,
		  1.0f, input_error, conf.max_bunch_size,
		  weights_mat_ptr, num_outputs,
		  1.0f, output_error, conf.max_bunch_size,
		  input_error_shift, 0, output_error_shift,
		  conf.use_cuda_flag);
	else
	  doSgemm(CblasColMajor, CblasNoTrans, CblasTrans,
		  conf.cur_bunch_size, num_inputs, num_outputs,
		  1.0f, input_error, conf.max_bunch_size,
		  weights_mat_ptr, num_inputs,
		  1.0f, output_error, conf.max_bunch_size,
		  input_error_shift, 0, output_error_shift,
		  conf.use_cuda_flag);
      }
      else {
	if (!transpose_weights)
	  doSgemv(CblasColMajor, CblasTrans,
		  num_outputs, num_inputs,
		  1.0f, weights_mat_ptr, num_outputs,
		  input_error, conf.max_bunch_size,
		  1.0f, output_error, conf.max_bunch_size,
		  0, input_error_shift, output_error_shift,
		  conf.use_cuda_flag);
	else {
	  doSgemv(CblasColMajor, CblasNoTrans,
		  num_inputs, num_outputs,
		  1.0f, weights_mat_ptr, num_inputs,
		  input_error, conf.max_bunch_size,
		  1.0f, output_error, conf.max_bunch_size,
		  0, input_error_shift, output_error_shift,
		  conf.use_cuda_flag);
	}
      }
    } // if output_error != 0
    
    if (neuron_squared_length_upper_bound > 0.0f) {
      if (conf.use_cuda_flag)
	ERROR_EXIT(128,"Max-norm penalty CUDA version not implemented yet!!!\n");
      if (!transpose_weights) {
	// TODO: Implement this in CBLAS and CUDA
	FloatGPUMirroredMemoryBlock *sql_sums   = outputs->getSquaredLengthSums();
	FloatGPUMirroredMemoryBlock *weights_mat_ptr = weights_matrix->getPtr();
	
	float *squared_length_sums = sql_sums->getPPALForReadAndWrite();
	const float *w = weights_mat_ptr->getPPALForRead();
	
	for (unsigned int j=0; j<num_outputs; ++j) {
	  unsigned int k = j;
	  // compute squared length, adding previous squared lengths computed
	  // over the same neuron
	  float squared_length = squared_length_sums[j];
	  for (unsigned int i=0; i<num_inputs; ++i) {
	    squared_length += w[k]*w[k];
	    k += num_outputs;
	  }
	  squared_length_sums[j] = squared_length;
	}
      }
      else {
	// TODO: Implement this in CBLAS and CUDA
	FloatGPUMirroredMemoryBlock *sql_sums   = outputs->getSquaredLengthSums();
	FloatGPUMirroredMemoryBlock *weights_mat_ptr = weights_matrix->getPtr();
	
	float *squared_length_sums = sql_sums->getPPALForReadAndWrite();
	const float *w = weights_mat_ptr->getPPALForRead();
	
	unsigned int k = 0;
	for (unsigned int j=0; j<num_outputs; ++j) {
	  // compute squared length, adding previous squared lengths computed
	  // over the same neuron
	  float squared_length = squared_length_sums[j];
	  for (unsigned int i=0; i<num_inputs; ++i) {
	    squared_length += w[k]*w[k];
	    ++k;
	  }
	  squared_length_sums[j] = squared_length;
	}
      }
    }
  }
  
  void DotProductAction::
  computeBPUpdateOnPrevVectors(FloatGPUMirroredMemoryBlock *prev_weights_mat_ptr,
			       FloatGPUMirroredMemoryBlock *input,
			       const unsigned int input_shift,
			       FloatGPUMirroredMemoryBlock *input_error,
			       const unsigned int input_error_shift,
			       float beta) {
    // backprop learning rule:
    // PREV_W = alpha * ERRORS + PREV_W
    const unsigned int references = weights_matrix->getNumReferences();
    // prev_w[i,j] = -learning_rate*1/sqrt(N*bsize) * ERROR_INPUT[j] + prev_w[i,j]
    const float norm_learn_rate =
      -(1.0f/sqrtf(static_cast<float>(references*conf.cur_bunch_size))) *
      //-(1.0f/static_cast<float>(references*conf.cur_bunch_size)) *
      //-(1.0f/sqrtf(static_cast<float>(references))) *
      learning_rate;
      
    if (inputs->isSparse()) {
      if (beta < 1.0f)
	doSscal((num_inputs * num_outputs),
		beta,
		prev_weights_mat_ptr, 0, 1,
		conf.use_cuda_flag);
      if (!transpose_weights) {
	const float *input_float_ptr = input->getPPALForRead() + input_shift;
	unsigned int w_shift = 0;
	for (unsigned int i=0; i<num_inputs; ++i, w_shift+=num_outputs) {
	  for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	    float v = input_float_ptr[b];
	    if ( v != 0.0f ) {
	      doSaxpy(num_outputs,
		      norm_learn_rate*v,
		      input_error, b+input_error_shift, conf.max_bunch_size,
		      prev_weights_mat_ptr, w_shift, 1,
		      conf.use_cuda_flag);
	    }
	  }
	  input_float_ptr += conf.max_bunch_size;
	}
      } // if !transposed weights
      else {
	const float *input_float_ptr = input->getPPALForRead() + input_shift;
	for (unsigned int i=0; i<num_inputs; ++i) {
	  for (unsigned int b=0; b<conf.cur_bunch_size; ++b) {
	    float v = input_float_ptr[b];
	    if ( v != 0.0f ) {
	      doSaxpy(num_outputs,
		      norm_learn_rate*v,
		      input_error, b+input_error_shift, conf.max_bunch_size,
		      prev_weights_mat_ptr, i, num_outputs,
		      conf.use_cuda_flag);
	    }
	  }
	  input_float_ptr += conf.max_bunch_size;
	}
      } // if !transposed weights ... else
    } // if isSparse
    else {
      if (conf.cur_bunch_size > 1) {
	if (!transpose_weights) {
	  doSgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		  num_outputs, num_inputs, conf.cur_bunch_size, // dimensiones
		  norm_learn_rate,                          // alpha
		  input_error,                              // A
		  conf.max_bunch_size,                      // A stride
		  input,                                    // B
		  conf.max_bunch_size,                      // B stride
		  beta,                                     // beta
		  prev_weights_mat_ptr,                     // C
		  num_outputs,                              // C stride
		  input_error_shift, input_shift, 0,        // desplazamientos
		  conf.use_cuda_flag);
	} // if !transposed weights
	else
	  doSgemm(CblasColMajor, CblasTrans, CblasNoTrans,
		  num_inputs, num_outputs, conf.cur_bunch_size, // dimensiones
		  norm_learn_rate,                          // alpha
		  input,                                    // B
		  conf.max_bunch_size,                      // B stride
		  input_error,                              // A
		  conf.max_bunch_size,                      // A stride
		  beta,                                     // beta
		  prev_weights_mat_ptr,                     // C
		  num_inputs,                               // C stride
		  input_shift, input_error_shift, 0,        // desplazamientos
		  conf.use_cuda_flag);
      } // if bunch_size > 1 ... else
      else {
	if (beta < 1.0f)
	  doSscal((num_inputs * num_outputs),
		  beta,
		  prev_weights_mat_ptr, 0, 1,
		  conf.use_cuda_flag);
	if (!transpose_weights)
	  doSger(CblasColMajor,
		 num_outputs, num_inputs,
		 norm_learn_rate,
		 input_error, input_error_shift, conf.max_bunch_size,
		 input, input_shift, conf.max_bunch_size,
		 prev_weights_mat_ptr, 0, num_outputs,
		 conf.use_cuda_flag);
	else
	  doSger(CblasColMajor,
		 num_inputs, num_outputs,
		 norm_learn_rate,
		 input, input_shift, conf.max_bunch_size,
		 input_error, input_error_shift, conf.max_bunch_size,
		 prev_weights_mat_ptr, 0, num_inputs,
		 conf.use_cuda_flag);
      }
    } // if isSparse ... else
  }
  
  // The DotProductAction
  void DotProductAction::doUpdate() {
    assert(learning_rate > 0.0f &&
	   "Learning rate/momentum/weight decay needs to be fixed with "
	   "setOption method!!!");
    
    // Foces weights_matrix to update internal counts for a backward step
    weights_matrix->beginUpdate();
    
    FloatGPUMirroredMemoryBlock *weights_mat_ptr = weights_matrix->getPtr();
    FloatGPUMirroredMemoryBlock *prev_weights_mat_ptr =
      weights_matrix->getPrevPtr();
    FloatGPUMirroredMemoryBlock *input        = inputs->getPtr();
    FloatGPUMirroredMemoryBlock *input_error  = outputs->getErrorVectorPtr();
    FloatGPUMirroredMemoryBlock *output_error = inputs->getErrorVectorPtr();

    const unsigned int input_shift  = inputs->getOffset();
    const unsigned int output_shift = outputs->getOffset();
    
    float beta_parameter_for_cblas_bp = 1.0f;
    // Momentum computation
    if (weights_matrix->isFirstUpdateCall()) {      
      if (momentum > 0.0f) {
	// prev_w[i,j] = momentum * (w[i,j] - prev_w[i,j])
	weights_matrix->computeMomentumOnPrevVector(momentum,
						    conf.use_cuda_flag);
	weights_matrix->computeWeightDecayOnPrevVector(c_weight_decay,
						       conf.use_cuda_flag);
      }
      else {
	weights_matrix->copyToPrevVector(conf.use_cuda_flag);
	beta_parameter_for_cblas_bp = c_weight_decay;
      }
    } // if (weights_matrix->needsToComputeMomentum()) {
    
    computeBPUpdateOnPrevVectors(prev_weights_mat_ptr,
				 input, input_shift,
				 input_error, output_shift,
				 beta_parameter_for_cblas_bp);
    
    // Forces to update counts and swap vectors if necessary at this backward
    // step
    if (weights_matrix->endUpdate()) {
      if (neuron_squared_length_upper_bound > 0.0f) {
	if (!transpose_weights) {
	  // TODO: Implement this in CBLAS and CUDA
	  
	  FloatGPUMirroredMemoryBlock *sql_sums   = outputs->getSquaredLengthSums();
	  FloatGPUMirroredMemoryBlock *w_ptr      = weights_matrix->getPtr();
	  FloatGPUMirroredMemoryBlock *prev_w_ptr =
	    weights_matrix->getPrevPtr();
	
	  const float *squared_length_sums = sql_sums->getPPALForRead();
	  float *w      = weights_mat_ptr->getPPALForReadAndWrite();
	  float *prev_w = prev_weights_mat_ptr->getPPALForReadAndWrite();
	
	  for (unsigned int j=0; j<num_outputs; ++j) {
	    // compute squared length, adding previous squared lengths computed
	    // over the same neuron
	    float squared_length = squared_length_sums[j];
	    if (squared_length > neuron_squared_length_upper_bound) {
	      float ratio    = sqrtf(neuron_squared_length_upper_bound/squared_length);
	      unsigned int k = j;
	      for (unsigned int i=0; i<num_inputs; ++i) {
		w[k]      *= ratio;
		prev_w[k] *= ratio;
		k += num_outputs;
	      }
	    }
	  }
	}
	else {
	  // TODO: Implement this in CBLAS and CUDA
	
	  FloatGPUMirroredMemoryBlock *sql_sums   = outputs->getSquaredLengthSums();
	  FloatGPUMirroredMemoryBlock *w_ptr      = weights_matrix->getPtr();
	  FloatGPUMirroredMemoryBlock *prev_w_ptr =
	    weights_matrix->getPrevPtr();
	
	  const float *squared_length_sums = sql_sums->getPPALForRead();
	  float *w      = weights_mat_ptr->getPPALForReadAndWrite();
	  float *prev_w = prev_weights_mat_ptr->getPPALForReadAndWrite();
	
	  unsigned int k = 0;
	  for (unsigned int j=0; j<num_outputs; ++j) {
	    // compute squared length, adding previous squared lengths computed
	    // over the same neuron
	    float squared_length = squared_length_sums[j];
	    if (squared_length > neuron_squared_length_upper_bound) {
	      float ratio    = sqrtf(neuron_squared_length_upper_bound/squared_length);
	      for (unsigned int i=0; i<num_inputs; ++i) {
		w[k]      *= ratio;
		prev_w[k] *= ratio;
		k++;
	      }
	    }
	  }
	}
      }

    }
  }

  Action *DotProductAction::clone(hash<void*,void*> &clone_dict,
				  const ANNConfiguration &conf) {
    DotProductAction *action = new
      DotProductAction(conf,
		       static_cast<ActivationUnits*>(clone_dict[inputs]),
		       static_cast<ActivationUnits*>(clone_dict[outputs]),
		       static_cast<Connections*>(clone_dict[weights_matrix]),
		       transpose_weights);
    action->learning_rate  = learning_rate;
    action->momentum       = momentum;
    action->weight_decay   = weight_decay;
    action->c_weight_decay = c_weight_decay;
    action->neuron_squared_length_upper_bound = neuron_squared_length_upper_bound;
    return action;
  }

  void DotProductAction::setOption(const char *name, double value) {
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
  
  bool DotProductAction::hasOption(const char *name) {
    mHasOption("learning_rate");
    mHasOption("momentum");
    mHasOption("weight_decay");
    mHasOption("neuron_squared_length_upper_bound");
    return false;
  }
  
  double DotProductAction::getOption(const char *name) {
    mGetOption("learning_rate", learning_rate);
    mGetOption("momentum", momentum);
    // the weight decay is always fixed to 0
    mGetOption("weight_decay", weight_decay);
    mGetOption("neuron_squared_length_upper_bound", neuron_squared_length_upper_bound);
    ERROR_EXIT(140, "The option to be get does not exist.\n");
  }

  void DotProductAction::transferFanInToConnections() {
    if (!transpose_weights)
      weights_matrix->setFanIn(outputs->getFanIn());
    else
      weights_matrix->setFanIn(inputs->getFanIn());
  }
  
  //////////////////////////////////////////////////////////////////////////
}
