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
    
    AprilUtils::vector<ANNComponent*> components;
    // Token pointers which contains exactly the same that was received
    Basics::Token *input, *error_output;

    // This token is always a MatrixFloat
    Basics::TokenMatrixFloat *output, *error_input;

    // Auxiliar Token pointers to prepare data from and for contained components
    Basics::TokenBunchVector *input_vector,  *error_input_vector;
    Basics::TokenBunchVector *output_vector, *error_output_vector;

    bool segmented_input;

    // private auxiliar methods
    void buildInputBunchVector(Basics::TokenBunchVector *&vector_token,
			       Basics::Token *token);
    void buildErrorInputBunchVector(Basics::TokenBunchVector *&vector_token,
				    Basics::Token *token);
    Basics::TokenMatrixFloat *buildMatrixFloatToken(Basics::TokenBunchVector *token,
                                                    bool is_output);
    Basics::TokenMatrixFloat *buildMatrixFloatToken(Basics::Token *token,
                                                    bool is_output);
    
  public:
    JoinANNComponent(const char *name=0);
    virtual ~JoinANNComponent();
    
    void addComponent(ANNComponent *component);
    
    virtual Basics::Token *getInput() { return input; }
    virtual Basics::Token *getOutput() { return output; }
    virtual Basics::Token *getErrorInput() { return error_input; }
    virtual Basics::Token *getErrorOutput() { return error_output; }
    //
    virtual void setInput(Basics::Token *tk) {
      AssignRef(input, tk);
    }
    virtual void setOutput(Basics::Token *tk) {
      AssignRef(output, tk->convertTo<Basics::TokenMatrixFloat*>());
    }
    virtual void setErrorInput(Basics::Token *tk) {
      AssignRef(error_input, tk->convertTo<Basics::TokenMatrixFloat*>());
    }
    virtual void setErrorOutput(Basics::Token *tk) {
      AssignRef(error_output, tk);
    }
    
    virtual void copyState(AprilUtils::LuaTable &dict) {
      ANNComponent::copyState(dict);
      for (unsigned int i=0; i<components.size(); ++i) {
        components[i]->copyState(dict);
      }
    }

    virtual void setState(AprilUtils::LuaTable &dict) {
      ANNComponent::setState(dict);
      for (unsigned int i=0; i<components.size(); ++i) {
        components[i]->setState(dict);
      }
    }

    virtual void precomputeOutputSize(const AprilUtils::vector<unsigned int> &input_size,
				      AprilUtils::vector<unsigned int> &output_size) {
      UNUSED_VARIABLE(input_size);
      UNUSED_VARIABLE(output_size);
      ERROR_EXIT(128,
		 "Impossible to precomputeOutputSize in JoinANNComponent\n");
    }
    
    virtual Basics::Token *doForward(Basics::Token* input, bool during_training);

    virtual Basics::Token *doBackprop(Basics::Token *input_error);
    
    virtual void reset(unsigned int it=0);
    
    virtual ANNComponent *clone();
    
    virtual void setUseCuda(bool v);
    
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       AprilUtils::LuaTable &weights_dict,
		       AprilUtils::LuaTable &components_dict);
    
    virtual void copyWeights(AprilUtils::LuaTable &weights_dict);

    virtual void copyComponents(AprilUtils::LuaTable &components_dict);
    
    virtual ANNComponent *getComponent(AprilUtils::string &name);
    virtual void computeAllGradients(AprilUtils::LuaTable &weight_grads_dict);
    virtual void debugInfo() {
      ANNComponent::debugInfo();
      for (unsigned int i=0; i<components.size(); ++i)
	components[i]->debugInfo();
    }

    virtual char *toLuaString();
  };
}

#endif // JOINCOMPONENT_H
