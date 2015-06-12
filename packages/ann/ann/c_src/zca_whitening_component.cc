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
#include "unused_variable.h"
#include "error_print.h"
#include "table_of_token_codes.h"
#include "token_vector.h"
#include "token_matrix.h"
#include "zca_whitening_component.h"
#include "utilMatrixFloat.h"

using namespace AprilIO;
using namespace AprilMath;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilUtils;
using namespace Basics;

#define WEIGHTS_NAME "U"

namespace ANN {
  
  ZCAWhiteningANNComponent::ZCAWhiteningANNComponent(MatrixFloat *U,
						     SparseMatrixFloat *S,
						     float epsilon,
						     unsigned int takeN,
						     const char *name) :
    PCAWhiteningANNComponent(U,S,epsilon,takeN,name),
    dot_product_decoder(0, WEIGHTS_NAME,
			getOutputSize(), getInputSize(),
			false)
  {
    output_size = input_size;
    MatrixFloat *aux_U = U;
    if (takeN != 0) {
      int coords[2] = { 0,0 };
      int sizes[2] = { U->getDimSize(0), static_cast<int>(takeN) };
      aux_U = new MatrixFloat(this->U, coords, sizes, true);
    }
    matrix_set.put(WEIGHTS_NAME, aux_U);
    AprilUtils::LuaTable components_dict;
    dot_product_decoder.build(0, 0, matrix_set, components_dict);
    // avoid problems with DecRef in LuaTable
    IncRef(&dot_product_decoder);
  }
  
  ZCAWhiteningANNComponent::~ZCAWhiteningANNComponent() {
  }
  
  Token *ZCAWhiteningANNComponent::doForward(Token* _input,
					     bool during_training) {
    Token *rotated = PCAWhiteningANNComponent::doForward(_input, during_training);
    return dot_product_decoder.doForward(rotated, during_training);
  }
  
  Token *ZCAWhiteningANNComponent::doBackprop(Token *_error_input) {
    Token *rotated_error = dot_product_decoder.doBackprop(_error_input);
    return PCAWhiteningANNComponent::doBackprop(rotated_error);
  }
  
  ANNComponent *ZCAWhiteningANNComponent::clone() {
    ZCAWhiteningANNComponent *component = new ZCAWhiteningANNComponent(U, S,
								       epsilon,
								       takeN,
								       name.c_str());
    return component;
  }
  
  char *ZCAWhiteningANNComponent::toLuaString() {
    SharedPtr<CStringStream> stream(new CStringStream());
    AprilUtils::LuaTable options;
    options.put("ascii", false);
    stream->printf("ann.components.zca_whitening{ name='%s', U=matrix.fromString[[",
                   name.c_str());
    U->write(stream.get(), options);
    stream->put("]], S=matrix.sparse.fromString[[");
    S->write(stream.get(), options);
    stream->printf("]], epsilon=%g, takeN=%u, }", epsilon, getTakeN());
    stream->put("\0",1); // forces a \0 at the end of the buffer
    return stream->releaseString();
  }

  const char *ZCAWhiteningANNComponent::luaCtorName() const {
    return "ann.components.zca_whitening";
  }
  
  int ZCAWhiteningANNComponent::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable t(L);
    t["name"] = name.c_str();
    t["U"] = U;
    t["S"] = S;
    t["epsilon"] = epsilon;
    t["takeN"] = takeN;
    t.pushTable(L);
    return 1;
  }
}
