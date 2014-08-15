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
#ifndef JOINCOMPONENT_H
#define JOINCOMPONENT_H

#include "unused_variable.h"
#include "vector.h"
#include "ann_component.h"
#include "token_vector.h"
#include "token_matrix.h"

namespace ANN {

  /// A container component which distributes its input over the contained
  /// components. So, the input is spliced, and the input size must be the sum
  /// of components input sizes. The output is a join of all the components
  /// outputs.

  class JoinANNComponent : public ANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(JoinANNComponent);
    
    april_utils::vector<ANNComponent*> components;
    // Token pointers which contains exactly the same that was received
    basics::Token *input, *error_output;

    // This token is always a MatrixFloat
    basics::TokenMatrixFloat *output, *error_input;

    // Auxiliar Token pointers to prepare data from and for contained components
    basics::TokenBunchVector *input_vector,  *error_input_vector;
    basics::TokenBunchVector *output_vector, *error_output_vector;

    bool segmented_input;

    // private auxiliar methods
    void buildInputBunchVector(basics::TokenBunchVector *&vector_token,
			       basics::Token *token);
    void buildErrorInputBunchVector(basics::TokenBunchVector *&vector_token,
				    basics::Token *token);
    basics::TokenMatrixFloat *buildMatrixFloatToken(basics::TokenBunchVector *token,
                                                    bool is_output);
    basics::TokenMatrixFloat *buildMatrixFloatToken(basics::Token *token,
                                                    bool is_output);
    
  public:
    JoinANNComponent(const char *name=0);
    virtual ~JoinANNComponent();
    
    void addComponent(ANNComponent *component);
    
    virtual basics::Token *getInput() { return input; }
    virtual basics::Token *getOutput() { return output; }
    virtual basics::Token *getErrorInput() { return error_input; }
    virtual basics::Token *getErrorOutput() { return error_output; }

    virtual void precomputeOutputSize(const april_utils::vector<unsigned int> &input_size,
				      april_utils::vector<unsigned int> &output_size) {
      UNUSED_VARIABLE(input_size);
      UNUSED_VARIABLE(output_size);
      ERROR_EXIT(128,
		 "Impossible to precomputeOutputSize in JoinANNComponent\n");
    }
    
    virtual basics::Token *doForward(basics::Token* input, bool during_training);

    virtual basics::Token *doBackprop(basics::Token *input_error);
    
    virtual void reset(unsigned int it=0);
    
    virtual ANNComponent *clone();
    
    virtual void setUseCuda(bool v);
    
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       basics::MatrixFloatSet *weights_dict,
		       april_utils::hash<april_utils::string,ANNComponent*> &components_dict);
    
    virtual void copyWeights(basics::MatrixFloatSet *weights_dict);

    virtual void copyComponents(april_utils::hash<april_utils::string,ANNComponent*> &components_dict);
    
    virtual ANNComponent *getComponent(april_utils::string &name);
    virtual void computeAllGradients(basics::MatrixFloatSet *weight_grads_dict);
    virtual void debugInfo() {
      ANNComponent::debugInfo();
      for (unsigned int i=0; i<components.size(); ++i)
	components[i]->debugInfo();
    }

    virtual char *toLuaString();
  };
}

#endif // JOINCOMPONENT_H
