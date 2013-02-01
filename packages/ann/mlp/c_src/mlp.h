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
#ifndef MLP_H
#define MLP_H
#include "ann.h"
#include "actunit.h"
#include "all_all_connection.h"
#include "action.h"
#include "errorfunc.h"

namespace ANN {

  class MLP : public ANNBase {
  protected:
    // required to compute MSE and so on:
    unsigned int num_patterns_processed;
    // to fill the bunch:
    unsigned int cur_bunch_pos;

    float learning_rate, momentum, weight_decay, c_weight_decay;
    ErrorFunction *error_func;
    
    void  beginTrainingBatch();
    void  trainPattern(float *input, float *target_output);
    void  doTraining();
    float endTrainingBatch();

    void beginValidateBatch();
    // Calculates the error for a given pattern (no training)
    void validatePattern(float *input, float *target_output);
    float endValidateBatch();

  public:

    MLP(ANNConfiguration conf);
    ~MLP();

    void setErrorFunction(ErrorFunction *error_func);
    
    void setOption(const char *name, double value);
    bool hasOption(const char *name);
    double getOption(const char *name);
  
    virtual void loadModel(const char *filename);
    virtual void saveModel(const char *filename);
  
    void showNetworkAtts();
    void doForward();
    void doBackward();
    void showActivations();
    void showWeights();
  
    MLP *clone();
    virtual void randomizeWeights(MTRand *rnd, float low, float high,
				  bool use_fanin);

    void pushBackAllAllLayer(ActivationUnits    *inputs,
			     ActivationUnits    *outputs,
			     ActivationFunction *actf,
			     Connections       **weights, // if *ptr=0 reserves new
			     bool                transpose_weights,
			     bool                has_bias,
			     Connections       **bias); // if *ptr=0 reserves new
    
  };

}

#endif // MLP_H
