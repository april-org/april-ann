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
#ifndef DOTPRODUCTACTION_H
#define DOTPRODUCTACTION_H  

#include "action.h"
#include "actunit.h"
#include "connection.h"

namespace ANN {
  class DotProductAction : public Action {
    ActivationUnits      *inputs;
    ActivationUnits      *outputs;
    Connections          *weights_matrix;
    const unsigned int num_inputs, num_outputs;
    const ANNConfiguration &conf;
    float learning_rate, momentum, weight_decay, c_weight_decay;

    void
    backpropagateErrors(FloatGPUMirroredMemoryBlock *weights_mat_ptr,
			FloatGPUMirroredMemoryBlock *output_error,
			const unsigned int output_error_shift,
			FloatGPUMirroredMemoryBlock *input_error,
			const unsigned int input_error_shift);
    
    void
    computeBPUpdateOnPrevVectors(FloatGPUMirroredMemoryBlock *prev_weights_mat_ptr,
				 FloatGPUMirroredMemoryBlock *input,
				 const unsigned int input_shift,
				 FloatGPUMirroredMemoryBlock *input_error,
				 const unsigned int input_error_shift,
				 float beta);

  public:
    DotProductAction(const ANNConfiguration &conf,
		     ActivationUnits *inputs,
		     ActivationUnits *outputs,
		     Connections *weights_matrix);
    virtual ~DotProductAction();
    virtual void doForward();
    virtual void doBackward();
    virtual Action *clone(hash<void*,void*> &clone_dict,
			  const ANNConfiguration &conf);
    virtual void setOption(const char *name, double value);
    virtual bool hasOption(const char *name);
    virtual double getOption(const char *name);
  };
}

#endif // DOTPRODUCTACTION_H
