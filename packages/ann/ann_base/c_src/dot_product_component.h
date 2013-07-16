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
#ifndef DOTPRODUCTANNCOMPONENT_H
#define DOTPRODUCTANNCOMPONENT_H  

#include "token_matrix.h"
#include "cblas_headers.h"
#include "ann_component.h"
#include "connection.h"

namespace ANN {
  class DotProductANNComponent : public ANNComponent {
    Token            *input;
    TokenMatrixFloat *error_input, *output, *error_output;
    Connections *weights_matrix;
    unsigned int num_updates_from_last_prune;
    bool sparse_input;
    
    /// learning parameters
    float learning_rate, momentum, weight_decay, c_weight_decay;
    float max_norm_penalty;
    CBLAS_TRANSPOSE transpose_weights;
    
    void computeBP(MatrixFloat *weights_mat,
		   Token *input_token,
		   MatrixFloat *error_input_mat,
		   const float alpha,
		   const float beta);

  protected:

    virtual void computeGradients(MatrixFloat*& weight_grads);

  public:
    DotProductANNComponent(const char *name=0, const char *weights_name=0,
			   unsigned int input_size  = 0,
			   unsigned int output_size = 0,
			   bool transpose_weights   = false);
    virtual ~DotProductANNComponent();
    virtual Token *getInput() { return input; }
    virtual Token *getOutput() { return output; }
    virtual Token *getErrorInput() { return error_input; }
    virtual Token *getErrorOutput() { return error_output; }
    virtual Token *doForward(Token* input, bool during_training);
    virtual Token *doBackprop(Token *input_error);
    virtual void   doUpdate();
    virtual void   reset();
    virtual ANNComponent *clone();
    virtual void setOption(const char *name, double value);
    virtual bool hasOption(const char *name);
    virtual double getOption(const char *name);
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);
    virtual void copyWeights(hash<string,Connections*> &weights_dict);
    virtual void resetConnections() {
      if (weights_matrix) weights_matrix->reset();
    }

    virtual char *toLuaString();
    
    bool transposed() { return transpose_weights == CblasTrans; }
  };
}

#endif // DOTPRODUCTANNCOMPONENT_H
