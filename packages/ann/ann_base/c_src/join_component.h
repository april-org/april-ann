/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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

using april_utils::vector;

namespace ANN {

  class JoinANNComponent : public ANNComponent {
    vector<ANNComponent*> components;
    // Token pointers which contains exactly the same that was received
    Token *input, *error_input, *error_output;

    // This token is always a MemoryBlock
    TokenMemoryBlock *output;

    // Auxiliar Token pointers to prepare data from and for contained components
    TokenBunchVector *input_vector,  *error_input_vector;
    TokenBunchVector *output_vector, *error_output_vector;

    // private auxiliar methods
    void buildInputBunchVector(TokenBunchVector *&vector_token,
			       Token *token);
    void buildErrorInputBunchVector(TokenBunchVector *&vector_token,
				    Token *token);
    void buildMemoryBlockToken(TokenMemoryBlock *&mem_block_token,
			       TokenBunchVector *token);
    void buildMemoryBlockToken(TokenMemoryBlock *&mem_block_token,
			       Token *token);
    
  public:
    JoinANNComponent(const char *name);
    virtual ~JoinANNComponent();
    
    void addComponent(ANNComponent *component);
    
    virtual const Token *getInput() const { return input; }
    virtual const Token *getOutput() const { return output; }
    virtual const Token *getErrorInput() const { return error_input; }
    virtual const Token *getErrorOutput() const { return error_output; }
    
    virtual Token *doForward(Token* input, bool during_training);

    virtual Token *doBackprop(Token *input_error);
    
    virtual void reset();
    
    virtual ANNComponent *clone();

    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       hash<string,Connections*> &weights_dict,
		       hash<string,ANNComponent*> &components_dict);
  };
}

#endif // JOINCOMPONENT_H
