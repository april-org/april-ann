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
#include <cstdio>
#include <cstring>
#include "constString.h"
#include "error_print.h"
#include "trainsuper.h"
#include "mlp.h"
#include "bias_connection.h"
#include "all_all_connection.h"
#include "dot_product_action.h"
#include "forward_bias_action.h"
#include "activations_action.h"

namespace ANN {

  MLP::MLP(ANNConfiguration configuration) : ANNBase(configuration),
					     learning_rate(0.0f),
					     momentum(0.0f),
					     weight_decay(0.0f),
					     c_weight_decay(1.0f),
					     neuron_squared_length_upper_bound(-1.0f),
					     error_func(0)
  {
    setErrorFunction(new MSE());
  }

  MLP::~MLP() {
    if (error_func) DecRef(error_func);
  }

  void MLP::setErrorFunction(ErrorFunction *error_func) {
    if (this->error_func) DecRef(this->error_func);
    this->error_func = error_func;
    IncRef(this->error_func);
    conf.error_function_logistic_mandatory = error_func->logisticMandatory();
  }

  // FIXME: showActivations instead of showNetworkAtts??
  void MLP::showNetworkAtts()
  {
  }

  void MLP::doForward(bool during_training)
  {
    //printf ("FORWARD START\n");
    for (unsigned int i = 0; i < actions.size(); i++)
      actions[i]->doForward(during_training);
    //printf ("FORWARD END\n");
  }

  void MLP::doBackward()
  {
    //printf ("BACKWARD START\n");
    for (int i = actions.size() - 1; i >= 0; i--)
      actions[i]->doBackprop();
    for (int i = actions.size() - 1; i >= 0; i--)
      actions[i]->doUpdate();
    //printf ("BACKWARD END\n");
  }

  // FIXME: this is showOutputs, not showActivations
  void MLP::showActivations()
  {
    for (unsigned int i=0; i<activations.size(); ++i) {
      const float *output_ptr = activations[i]->getPtr()->getPPALForRead();
      printf("Activations layer %d:\n", i);
      for (unsigned int j = 0; j < activations[i]->size(); j++) {
	for (unsigned int k=0; k<conf.max_bunch_size; ++k) {
	  printf("%g ", *output_ptr);
	  output_ptr++;
	}
	printf ("\n");
      }
      printf ("\n");
    }

    for (unsigned int i=0; i<activations.size(); ++i) {
      if (activations[i]->getErrorVectorPtr()) {
	const float *output_ptr = activations[i]->getErrorVectorPtr()->getPPALForRead();
	printf("Errors layer %d:\n", i);
	for (unsigned int j = 0; j < activations[i]->size(); j++) {
	  for (unsigned int k=0; k<conf.max_bunch_size; ++k) {
	    printf("%g ", *output_ptr);
	    output_ptr++;
	  }
	  printf ("\n");
	}
	printf ("\n");
      }
    }
  }

  void MLP::showWeights()
  {
    for (unsigned int i=0; i<connections.size(); i++) {
      printf("W[%d] = ", i);
      const float *w = connections[i]->getPtr()->getPPALForRead();
      for (unsigned int k=0; k<connections[i]->getNumWeights(); ++k)
	printf("%f ", w[k]);
      printf("\n");
    }
  }

  // from trainable supervised
  void MLP::setOption(const char *name, double value)
  {
    //synchronize every option from the network
    //
    // NOTE: option is only updated in connections since training
    // algorithm is placed there
    for (unsigned int i = 0; i < actions.size(); i++)
      if (actions[i]->hasOption(name))
	actions[i]->setOption(name, value);
  
    mSetOption("learning_rate", learning_rate);
    mSetOption("momentum", momentum);
    // Caso especial, no podemos usar esto: mSetOption("weight_decay", weight_decay);
    if (strcmp("weight_decay", name) == 0) {
      weight_decay   = value;
      c_weight_decay = 1.0 - weight_decay;
      return;
    }
    mSetOption("neuron_squared_length_upper_bound",
	       neuron_squared_length_upper_bound);
    mSetOption("dropout", dropout);
    ERROR_EXIT(140, "The option to be set does not exist.\n");
  }

  bool MLP::hasOption(const char *name)
  {
    mHasOption("learning_rate");
    mHasOption("momentum");
    mHasOption("weight_decay");
    mHasOption("neuron_squared_length_upper_bound");
    mHasOption("dropout");
    return false;
  }

  double MLP::getOption(const char *name)
  {
    mGetOption("learning_rate", learning_rate);
    mGetOption("momentum", momentum);
    mGetOption("weight_decay", weight_decay);
    mGetOption("neuron_squared_length_upper_bound", neuron_squared_length_upper_bound);
    mGetOption("dropout", dropout);
    ERROR_EXIT(140, "The option to be get does not exist.\n");
  }

  void MLP::loadModel(const char *filename)
  {
  }

  void MLP::saveModel(const char *filename)
  {
  }
  
  void MLP::beginValidateBatch() {
    //printf ("BEGIN VALIDATE\n");
    cur_bunch_pos	   = 0;
    num_patterns_processed = 0;
    resetPatternErrorsAuxiliarVector();
  }

  // le da un patron para calcular el error, PERO sin entrenar
  void MLP::validatePattern(float *input, float *target_output) {

    ++num_patterns_processed;
  
    // feed MLP with singular input
    setInput(input, cur_bunch_pos);
    // and singular desired output
    setDesiredOutput(target_output, cur_bunch_pos);
    ++cur_bunch_pos;
    if (cur_bunch_pos >= conf.cur_bunch_size) {
      doForward();
      cur_bunch_pos = 0;
      
      error_func->computePatternErrorFunction(output_neurons, 
					      desired_output,
					      output_errors,
					      pattern_errors,
					      total_num_outputs,
					      conf);
    }
  }

  float MLP::endValidateBatch() {
    if (cur_bunch_pos != 0) {
      unsigned int old_bunch = conf.cur_bunch_size;
      conf.cur_bunch_size = cur_bunch_pos;
      doForward();

      error_func->computePatternErrorFunction(output_neurons,
					      desired_output,
					      output_errors,
					      pattern_errors,
					      total_num_outputs,
					      conf);
      conf.cur_bunch_size = old_bunch;
    }
    return error_func->computeBatchErrorFunction(patternErrorsSum(),
						 num_patterns_processed);
  }

  void MLP::beginTrainingBatch()
  {
    //printf ("BEGIN TRAINING\n");
    cur_bunch_pos	   = 0;
    num_patterns_processed = 0;
    resetPatternErrorsAuxiliarVector();
  }

  void MLP::trainPattern(float *input, float *target_output)
  {
    ++num_patterns_processed;
  
    setInput(input, cur_bunch_pos);
    setDesiredOutput(target_output, cur_bunch_pos);
    ++cur_bunch_pos;
    if (cur_bunch_pos >= conf.cur_bunch_size) {
      doTraining();
      cur_bunch_pos = 0;
    }
    // FIXME explain and put 10000 into a constant with a proper name
    if (num_patterns_processed % 10000 == 0) pruneSubnormalAndCheckNormal();
  }

  void MLP::doTraining()
  {
    resetErrorVectors();
    
    doForward(true);

    error_func->computePatternErrorFunction(output_neurons,
					    desired_output,
					    output_errors,
					    pattern_errors,
					    total_num_outputs,
					    conf);
    
    doBackward();
  }

  float MLP::endTrainingBatch()
  {
    if (cur_bunch_pos != 0) {	
      unsigned int old_bunch = conf.cur_bunch_size;
      conf.cur_bunch_size = cur_bunch_pos;
      doTraining();
      conf.cur_bunch_size = old_bunch;
    }
    pruneSubnormalAndCheckNormal();
    return error_func->computeBatchErrorFunction(patternErrorsSum(),
						 num_patterns_processed);
  }

  MLP *MLP::clone() {
    MLP *copy = new MLP(getConfReference());
    copy->setErrorFunction(error_func->clone());
    cloneTopologyTo(copy);
    copy->learning_rate  = learning_rate;
    copy->momentum       = momentum;
    copy->weight_decay   = weight_decay;
    copy->c_weight_decay = c_weight_decay;
    copy->neuron_squared_length_upper_bound = neuron_squared_length_upper_bound;
    return copy;
  }

  void MLP::randomizeWeights(MTRand *rnd, float low, float high,
			     bool use_fanin) {
    for (unsigned int i = 0; i < actions.size(); i++)
      actions[i]->transferFanInToConnections();
    for (unsigned int i=0; i<connections.size(); ++i)
      connections[i]->randomizeWeights(rnd, low, high, use_fanin);
  }

  void MLP::pushBackAllAllLayer(ActivationUnits    *inputs,
				ActivationUnits    *outputs,
				ActivationFunction *actf,
				Connections       **weights,
				bool                transpose_weights,
				bool                has_bias,
				Connections       **bias) {
    Action *action;
    if (has_bias) {
      // Add action bias
      if (*bias == 0) {
	*bias = new BiasConnections(outputs->numNeurons());
	registerConnections(*bias);
      }
      action = new ForwardBiasAction(getConfReference(), outputs, *bias);
      registerAction(action);
    }
    if (*weights == 0) {
      *weights = new AllAllConnections(inputs->numNeurons(), outputs->numNeurons());
      registerConnections(*weights);
    }
    action = new DotProductAction(getConfReference(),
				  inputs, outputs,
				  *weights, transpose_weights);
    registerAction(action);
    if (actf == 0) actf = new LinearActivationFunction();
    action = new ActivationsAction(getConfReference(),
				   outputs, actf);
    registerAction(action);
  }

}
