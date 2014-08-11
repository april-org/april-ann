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
#ifndef ZCAWHITENINGCOMPONENT_H
#define ZCAWHITENINGCOMPONENT_H

#include "sparse_matrixFloat.h"
#include "matrixFloat.h"
#include "vector.h"
#include "ann_component.h"
#include "token_vector.h"
#include "token_matrix.h"
#include "pca_whitening_component.h"
#include "dot_product_component.h"

using april_utils::vector;

namespace ANN {

  class ZCAWhiteningANNComponent : public PCAWhiteningANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(ZCAWhiteningANNComponent);
    
    DotProductANNComponent dot_product_decoder; //< Applies the reconstruction from PCA rotated data
  public:
    ZCAWhiteningANNComponent(MatrixFloat *U,
			     SparseMatrixFloat *S,
			     float epsilon=0.0f,
			     unsigned int takeN=0,
			     const char *name=0);
    virtual ~ZCAWhiteningANNComponent();
    
    virtual Token *doForward(Token* input, bool during_training);
    
    virtual Token *doBackprop(Token *input_error);
    
    virtual Token *getInput() { return PCAWhiteningANNComponent::getInput(); }
    virtual Token *getOutput() { return dot_product_decoder.getOutput(); }
    virtual Token *getErrorInput() { return dot_product_decoder.getErrorInput(); }
    virtual Token *getErrorOutput() { return PCAWhiteningANNComponent::getErrorOutput(); }


    virtual ANNComponent *clone();
    
    virtual char *toLuaString();
  };
}

#endif // ZCAWHITENINGCOMPONENT_H
