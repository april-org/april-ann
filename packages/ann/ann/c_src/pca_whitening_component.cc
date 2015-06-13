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
#include "error_print.h"
#include "pca_whitening_component.h"
#include "table_of_token_codes.h"
#include "token_vector.h"
#include "token_matrix.h"
#include "unused_variable.h"
#include "utilMatrixFloat.h"

using namespace AprilIO;
using namespace AprilMath;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilUtils;
using namespace Basics;

#define WEIGHTS_NAME "U_S_epsilon"

namespace ANN {
  
  PCAWhiteningANNComponent::PCAWhiteningANNComponent(MatrixFloat *U,
						     SparseMatrixFloat *S,
						     float epsilon,
						     unsigned int takeN,
						     const char *name) :
    ANNComponent(name, 0,
		 static_cast<unsigned int>(U->getDimSize(0)),
		 (takeN==0)?(static_cast<unsigned int>(S->getDimSize(0))):(takeN)),
    U(U), S(S), epsilon(epsilon),
    dot_product_encoder(0, WEIGHTS_NAME,
			getInputSize(), getOutputSize(),
			true),
    takeN(takeN) {
    if (U->getNumDim() != 2)
      ERROR_EXIT(128, "Needs a bi-dimensional matrix as U argument\n");
    if ( !S->isDiagonal() )
      ERROR_EXIT(128, "Needs a sparse diagonal matrix as S argument\n");
    if (static_cast<int>(takeN) > S->getDimSize(0))
      ERROR_EXIT(128, "Taking more components than size of S matrix\n");
    if (takeN != 0) {
      int coords[2] = { 0,0 };
      int sizes[2] = { U->getDimSize(0), static_cast<int>(takeN) };
      U_S_epsilon = new MatrixFloat(this->U, coords, sizes, true);
    }
    else U_S_epsilon = this->U->clone();
    IncRef(this->U);
    IncRef(this->S);
    IncRef(U_S_epsilon);
    // regularization
    MatrixFloat *aux_mat = 0;
    SparseMatrixFloat::const_iterator Sit(this->S->begin());
    for (int i=0; i<U_S_epsilon->getDimSize(1); ++i, ++Sit) {
      april_assert(Sit != this->S->end());
      aux_mat = U_S_epsilon->select(1, i, aux_mat);
      matScal(aux_mat, 1/sqrtf( (*Sit) + epsilon ) );
    }
    delete aux_mat;
    //
    matrix_set.put(WEIGHTS_NAME, U_S_epsilon);
    AprilUtils::LuaTable components_dict;
    dot_product_encoder.build(0, 0, matrix_set, components_dict);
    // avoid problems with DecRef in LuaTable
    IncRef(&dot_product_encoder);
  }
  
  PCAWhiteningANNComponent::~PCAWhiteningANNComponent() {
    DecRef(U);
    DecRef(S);
    DecRef(U_S_epsilon);
  }
  
  Token *PCAWhiteningANNComponent::doForward(Token* _input, bool during_training) {
    return dot_product_encoder.doForward(_input, during_training);
  }

  Token *PCAWhiteningANNComponent::doBackprop(Token *_error_input) {
    return dot_product_encoder.doBackprop(_error_input);
  }
  
  void PCAWhiteningANNComponent::reset(unsigned int it) {
    dot_product_encoder.reset(it);
  }
  
  ANNComponent *PCAWhiteningANNComponent::clone() {
    PCAWhiteningANNComponent *component = new PCAWhiteningANNComponent(U, S,
								       epsilon,
								       takeN,
								       name.c_str());
    return component;
  }
  
  void PCAWhiteningANNComponent::build(unsigned int _input_size,
				       unsigned int _output_size,
				       AprilUtils::LuaTable &weights_dict,
				       AprilUtils::LuaTable &components_dict) {
    // TODO: CHECK INPUT OUTPUT SIZES
    UNUSED_VARIABLE(_input_size);
    UNUSED_VARIABLE(_output_size);
    UNUSED_VARIABLE(weights_dict);
    UNUSED_VARIABLE(components_dict);
  }
  
  const char *PCAWhiteningANNComponent::luaCtorName() const {
    return "ann.components.pca_whitening";
  }
  int PCAWhiteningANNComponent::exportParamsToLua(lua_State *L) {
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
