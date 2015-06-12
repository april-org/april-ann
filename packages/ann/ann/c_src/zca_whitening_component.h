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

namespace ANN {

  class ZCAWhiteningANNComponent : public PCAWhiteningANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(ZCAWhiteningANNComponent);
    
    DotProductANNComponent dot_product_decoder; //< Applies the reconstruction from PCA rotated data
  public:
    ZCAWhiteningANNComponent(Basics::MatrixFloat *U,
			     Basics::SparseMatrixFloat *S,
			     float epsilon=0.0f,
			     unsigned int takeN=0,
			     const char *name=0);
    virtual ~ZCAWhiteningANNComponent();
    
    virtual Basics::Token *doForward(Basics::Token* input, bool during_training);
    
    virtual Basics::Token *doBackprop(Basics::Token *input_error);
    
    virtual Basics::Token *getInput() { return PCAWhiteningANNComponent::getInput(); }
    virtual Basics::Token *getOutput() { return dot_product_decoder.getOutput(); }
    virtual Basics::Token *getErrorInput() { return dot_product_decoder.getErrorInput(); }
    virtual Basics::Token *getErrorOutput() { return PCAWhiteningANNComponent::getErrorOutput(); }
    //
    virtual void setInput(Basics::Token *tk) {
      PCAWhiteningANNComponent::setInput(tk);
    }
    virtual void setOutput(Basics::Token *tk) {
      dot_product_decoder.setOutput(tk);
    }
    virtual void setErrorInput(Basics::Token *tk) {
      dot_product_decoder.setErrorInput(tk);
    }
    virtual void setErrorOutput(Basics::Token *tk) {
      PCAWhiteningANNComponent::setErrorOutput(tk);
    }

    // FIXME: Is it needd an implementation of copyState and setState?
    /*
      virtual void copyState(AprilUtils::LuaTable &dict) {
      dot_product->copyState(dict);
      bias->copyState(dict);
      }
      virtual void setState(AprilUtils::LuaTable &dict) {
      dot_product->setState(dict);
      bias->setState(dict);
      }
    */
    

    virtual ANNComponent *clone();
    
    virtual char *toLuaString();
    
    virtual const char *luaCtorName() const;
    virtual int exportParamsToLua(lua_State *L);
  };
}

#endif // ZCAWHITENINGCOMPONENT_H
