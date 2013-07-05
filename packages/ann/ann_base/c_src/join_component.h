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

#include "vector.h"
#include "ann_component.h"
#include "token_vector.h"
#include "token_matrix.h"

using april_utils::vector;

namespace ANN {

  class JoinANNComponent : public ANNComponent {
    vector<ANNComponent*> components;
    // Token pointers which contains exactly the same that was received
    Token *input, *error_output;

    // This token is always a MatrixFloat
    TokenMatrixFloat *output, *error_input;

    // Auxiliar Token pointers to prepare data from and for contained components
    TokenBunchVector *input_vector,  *error_input_vector;
    TokenBunchVector *output_vector, *error_output_vector;

    bool segmented_input;

    // private auxiliar methods
    void buildInputBunchVector(TokenBunchVector *&vector_token,
			       Token *token);
    void buildErrorInputBunchVector(TokenBunchVector *&vector_token,
				    Token *token);
    TokenMatrixFloat *buildMatrixFloatToken(TokenBunchVector *token,
					    bool is_output);
    TokenMatrixFloat *buildMatrixFloatToken(Token *token,
					    bool is_output);
    
  public:
    JoinANNComponent(const char *name=0);
    virtual ~JoinANNComponent();
    
    void addComponent(ANNComponent *component);
    
    virtual Token *getInput() { return input; }
    virtual Token *getOutput() { return output; }
    virtual Token *getErrorInput() { return error_input; }
    virtual Token *getErrorOutput() { return error_output; }
    
    virtual Token *doForward(Token* input, bool during_training);

    virtual Token *doBackprop(Token *input_error);
    
    virtual void doUpdate();

    virtual void reset();
    
    virtual ANNComponent *clone();
    
    virtual void setUseCuda(bool v);
    
    virtual void setOption(const char *name, double value);

    virtual bool hasOption(const char *name);
    
    virtual double getOption(const char *name);
    
    virtual void build(unsigned int input_size,
		       unsigned int output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);
    
    virtual void copyWeights(hash<string,Connections*> &weights_dict);

    virtual void copyComponents(hash<string,ANNComponent*> &components_dict);
    
    virtual ANNComponent *getComponent(string &name);
    virtual void resetConnections() {
      for (unsigned int i=0; i<components.size(); ++i)
	components[i]->resetConnections();
    }
    virtual void debugInfo() {
      ANNComponent::debugInfo();
      for (unsigned int i=0; i<components.size(); ++i)
	components[i]->debugInfo();
    }

    virtual char *toLuaString();
  };
}

#endif // JOINCOMPONENT_H
