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
#ifndef PCAWHITENINGCOMPONENT_H
#define PCAWHITENINGCOMPONENT_H

#include "sparse_matrixFloat.h"
#include "matrixFloat.h"
#include "vector.h"
#include "ann_component.h"
#include "token_vector.h"
#include "token_matrix.h"
#include "dot_product_component.h"
#include "matrixFloatSet.h"

namespace ANN {

  class PCAWhiteningANNComponent : public ANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(PCAWhiteningANNComponent);
    
  protected:
    basics::MatrixFloat *U; //< bi-dimensional
    basics::SparseMatrixFloat *S; //< sparse diagonal matrix in CSR
    basics::MatrixFloat *U_S_epsilon; //< matrix for dot_product_component
    float epsilon;  //< regularization
    DotProductANNComponent dot_product_encoder; //< Applies the transformation
    basics::MatrixFloatSet matrix_set; //< Auxiliary for dot_product_encoder build
    unsigned int takeN;
    
  public:
    PCAWhiteningANNComponent(basics::MatrixFloat *U,
			     basics::SparseMatrixFloat *S,
			     float epsilon=0.0f,
			     unsigned int takeN=0,
			     const char *name=0);
    virtual ~PCAWhiteningANNComponent();
    
    virtual basics::Token *getInput() { return dot_product_encoder.getInput(); }
    virtual basics::Token *getOutput() { return dot_product_encoder.getOutput(); }
    virtual basics::Token *getErrorInput() { return dot_product_encoder.getErrorInput(); }
    virtual basics::Token *getErrorOutput() { return dot_product_encoder.getErrorOutput(); }
    
    virtual basics::Token *doForward(basics::Token* input, bool during_training);
    
    virtual basics::Token *doBackprop(basics::Token *input_error);
    
    virtual void reset(unsigned int it=0);
    
    virtual ANNComponent *clone();

    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       basics::MatrixFloatSet *weights_dict,
		       april_utils::hash<april_utils::string,ANNComponent*> &components_dict);

    virtual char *toLuaString();
    unsigned int getTakeN() const { return takeN; }
  };
}

#endif // PCAWHITENINGCOMPONENT_H
