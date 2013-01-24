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
#include "trainsuper.h"
#include "forward_bias_action.h"

namespace ANN {

  //////////////////////////////////////
  // ForwardBiasAction implementation //
  //////////////////////////////////////
  
  ForwardBiasAction::ForwardBiasAction(const ANNConfiguration &conf,
				       ActivationUnits *outputs,
				       Connections *bias_vector) :
    Action(conf),
    outputs(outputs), bias_vector(bias_vector),
    learning_rate(-1.0f),
    momentum(0.0f) {
    if (!bias_vector->checkInputOutputSizes(0, outputs))
      ERROR_EXIT(256, "The input/output sizes are not correct.\n");
    bias_vector->countReference();
    IncRef(outputs);
    IncRef(bias_vector);
  }
  
  ForwardBiasAction::~ForwardBiasAction() {
    DecRef(outputs);
    DecRef(bias_vector);
  }
  
  // The ForwardBiasAction copies the bias vector at the outputs vector in the
  // method doForward.
  void ForwardBiasAction::doForward() {
    FloatGPUMirroredMemoryBlock *output_ptr      = outputs->getPtr();
    FloatGPUMirroredMemoryBlock *bias_vector_ptr = bias_vector->getPtr();
    doScopyLoop(outputs->numNeurons(),
		bias_vector_ptr, 1,
		output_ptr, conf.max_bunch_size,
		conf.cur_bunch_size, 1,
		conf.use_cuda_flag);
  }
  
  // The ForwardBiasAction doBackward method tell the bias_vector to compute
  // momentum if it is necessary, and doBackward computes also the
  // backpropagation update of the bias_vector.
  void ForwardBiasAction::doBackward() {
    assert(learning_rate > 0.0f &&
	   "Learning rate/momentum/weight decay needs to be fixed with "
	   "setOption method!!!");
    
    // Foces bias_vector to update internal counts for a backward step
    bias_vector->beginUpdate();
    
    FloatGPUMirroredMemoryBlock *bias_ptr      = bias_vector->getPtr();
    FloatGPUMirroredMemoryBlock *prev_bias_ptr = bias_vector->getPrevPtr();
    FloatGPUMirroredMemoryBlock *input_error   = outputs->getErrorVectorPtr();
    
    // Momentum computation
    if (bias_vector->isFirstUpdateCall()) {
      if (momentum > 0.0f) {
	// prev_w[i,j] = momentum * (w[i,j] - prev_w[i,j])
	bias_vector->computeMomentumOnPrevVector(momentum, conf.use_cuda_flag);
	bias_vector->computeWeightDecayOnPrevVector(1.0f,
						    conf.use_cuda_flag);
      }
      else bias_vector->copyToPrevVector(conf.use_cuda_flag);
    } // if (bias_vector->needsToComputeMomentum()) {

    // backprop learning rule:
    // PREV_W = alpha * ERRORS + PREV_W
    const unsigned int references = bias_vector->getNumReferences();
    // prev_w[i,j] = -learning_rate*1/sqrt(N*bsize) * ERROR_INPUT[j] + prev_w[i,j]
    const float norm_learn_rate =
      //-(1.0f/sqrtf(static_cast<float>(references*conf.cur_bunch_size))) *
      -(1.0f/sqrtf(static_cast<float>(references))) *
      learning_rate;
    
    // bias update: prev_bias[j] = prev_bias[j] + \sum_b norm_learn_rate * ERROR_INPUT[b,j]
    doSaxpyLoop(outputs->numNeurons(),
		norm_learn_rate,
		input_error, conf.max_bunch_size,
		prev_bias_ptr, 1,
		conf.cur_bunch_size, 1,
		conf.use_cuda_flag);

    // Forces to update counts at this backward step
    bias_vector->endUpdate();
  }

  Action *ForwardBiasAction::clone(hash<void*,void*> &clone_dict,
				   const ANNConfiguration &conf) {
    ForwardBiasAction *action = new
      ForwardBiasAction(conf,
			static_cast<ActivationUnits*>(clone_dict[outputs]),
			static_cast<Connections*>(clone_dict[bias_vector]));
    action->learning_rate  = learning_rate;
    action->momentum       = momentum;
    return action;
  }

  void ForwardBiasAction::setOption(const char *name, double value) {
    mSetOption("learning_rate", learning_rate);
    mSetOption("momentum", momentum);
    // the weight decay is always fixed to 0, but it does not throw error
    // message
    if (strcmp("weight_decay", name) == 0) return;
    ERROR_EXIT(140, "The option to be set does not exist.\n");
  }
  
  bool ForwardBiasAction::hasOption(const char *name) {
    mHasOption("learning_rate");
    mHasOption("momentum");
    mHasOption("weight_decay");
    return false;
  }
  
  double ForwardBiasAction::getOption(const char *name) {
    mGetOption("learning_rate", learning_rate);
    mGetOption("momentum", momentum);
    // the weight decay is always fixed to 0
    mGetOption("weight_decay", 0.0f);
    ERROR_EXIT(140, "The option to be get does not exist.\n");
  }
  
  //////////////////////////////////////////////////////////////////////////
}
