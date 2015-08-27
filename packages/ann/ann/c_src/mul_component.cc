/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Adrian Palacios Corella, Francisco
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
#include "april_assert.h"
#include "mul_component.h"
#include "unused_variable.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  MulANNComponent::MulANNComponent(unsigned int size,
                                   bool scalar,
                                   const char *name,
                                   const char *weights_name,
                                   MatrixFloat *matrix) :
    VirtualMatrixANNComponent(name, weights_name, size, size),
    scalar(scalar),
    mul_vector(matrix)
  {
    setInputContiguousProperty(true);
    if (weights_name == 0) generateDefaultWeightsName("m");
  }

  MulANNComponent::~MulANNComponent() {
  }

  void MulANNComponent::applyCmul(MatrixFloat *dest, MatrixFloat *w) {
    if (w->size() == 1) {
      matScal(dest, *(w->begin()));
    }
    else {
      if (dest->size() == w->size()) {
        matCmul(dest, w);
      }
      else {
        april_assert(dest->size() > w->size());
        // addition of mul vector at output
        AprilUtils::SharedPtr<MatrixFloat> dest_slice;
        for (int i=0; i<dest->getDimSize(0); ++i) {
          dest_slice = dest->select(0, i, dest_slice.get());
          matCmul(dest_slice.get(), w);
        }
      }
    }
  }

  MatrixFloat *MulANNComponent::privateDoForward(MatrixFloat* input,
                                                 bool during_training) {
    UNUSED_VARIABLE(during_training);
    if (input->getNumDim() < 2)
      ERROR_EXIT2(128, "A 2-dimensional matrix is expected, found %d. "
		  "[%s]", input->getNumDim(), name.c_str());
    if (mul_vector.empty()) ERROR_EXIT1(129, "Not built component %s\n",
                                        name.c_str());
    // transfer of input to output
    MatrixFloat *output = input->clone();
    applyCmul(output, mul_vector.get());
    //
    return output;
  }

  /// In MulANNComponent this method is a multiplication by the weights vector
  MatrixFloat *MulANNComponent::privateDoBackprop(MatrixFloat *error_input)
  {
    if (error_input->getNumDim() < 2)
      ERROR_EXIT2(128, "A 2-dimensional matrix is expected, found %d. "
		  "[%s]", error_input->getNumDim(), name.c_str());
    if (mul_vector.empty()) ERROR_EXIT1(129, "Not built component %s\n",
                                        name.c_str());
    // transfer of input to output
    MatrixFloat *error_output = error_input->clone();
    applyCmul(error_output, mul_vector.get());
    //
    return error_output;
  }

  void MulANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
    // reset scalar counter
    mul_vector->resetSharedCount();
  }
  
  void MulANNComponent::computeGradients(const char *name,
                                          AprilUtils::LuaTable &weight_grads_dict) {
    // count one use of the vector
    mul_vector->addToSharedCount();
    MatrixFloat *grads_mat = weight_grads_dict.opt<MatrixFloat*>(name, 0);
    if (grads_mat == 0) {
      grads_mat = mul_vector->cloneOnlyDims();
      matZeros(grads_mat);
      weight_grads_dict.put(name, grads_mat);
    }
    else if (!grads_mat->sameDim(mul_vector.get())) {
      ERROR_EXIT(128, "Incorrect weights matrix dimensions\n");
    }
#ifdef USE_CUDA
    grads_mat->setUseCuda(use_cuda);
#endif
    MatrixFloat *error_input_mat = getErrorInputMatrix();
    AprilUtils::SharedPtr<MatrixFloat> aux = error_input_mat->clone();
    MatrixFloat *input_mat = getInputMatrix();
    applyCmul(aux.get(), input_mat);
    if (scalar) {
      *(grads_mat->begin()) = matSum(aux.get());
    }
    else {
      AprilUtils::SharedPtr<MatrixFloat> grads_mat_row;
      int dims[2] = { 1, aux->getDimSize(1) };
      grads_mat_row = grads_mat->rewrap(dims, 2);
      matSum(aux.get(), 0, grads_mat_row.get());
    }
  }
    
  ANNComponent *MulANNComponent::clone(AprilUtils::LuaTable &copies) {
    UNUSED_VARIABLE(copies);
    MulANNComponent *component = new MulANNComponent(input_size,
                                                     scalar,
                                                     name.c_str(),
                                                     weights_name.c_str());
    return component;
  }
  
  void MulANNComponent::build(unsigned int _input_size,
                              unsigned int _output_size,
                              AprilUtils::LuaTable &weights_dict,
                              AprilUtils::LuaTable &components_dict) {
    ANNComponent::build(_input_size, _output_size,
			weights_dict, components_dict);
    //
    if (!scalar && input_size == 0 && output_size == 0)
      ERROR_EXIT1(141, "Impossible to compute input/output "
		  "sizes for this component [%s]\n",
		  name.c_str());
    if (input_size == 0) input_size = output_size;
    else if (output_size == 0) output_size= input_size;
    else if (input_size != output_size)
           ERROR_EXIT1(142, "MulANNComponent input/output sizes must be equal [%s]\n",
                       name.c_str());
    
    ////////////////////////////////////////////////////////////////////
    MatrixFloat *w = weights_dict.opt<MatrixFloat*>(weights_name, 0);
    // printf("%s :: %p %p\n", weights_name.c_str(), w, mul_vector);
    if (w != 0) {
      mul_vector = w;
      // printf("COPY OF MUL FROM HASH %s\n", weights_name.c_str());
      if (w->getNumDim() != 1) {
        ERROR_EXIT1(256, "Needs a one dimensional matrix [%s]\n",
                    name.c_str());
      }
      if (!scalar && w->getDimSize(0) != static_cast<int>(input_size)) {
        ERROR_EXIT2(256, "Needs a one dimensional matrix with size %d [%s]\n",
                    input_size, name.c_str());
      }
    }
    else {
      if (mul_vector.empty()) {
        if (scalar) mul_vector = new MatrixFloat(1, 1);
        else mul_vector = new MatrixFloat(1, input_size);
      }
      weights_dict.put(weights_name, mul_vector.get());
    }
  }

  void MulANNComponent::copyWeights(AprilUtils::LuaTable &weights_dict) {
    if (!mul_vector)
      ERROR_EXIT1(100, "Component not built, impossible execute copyWeights [%s]\n",
		  name.c_str());
    MatrixFloat *w = weights_dict.opt<MatrixFloat*>(weights_name, 0);
    if (w != 0 && w != mul_vector.get())
      ERROR_EXIT2(101, "Weights dictionary contains %s weights name which is "
		  "not shared with mul_vector attribute [%s]\n",
		  weights_name.c_str(),
		  name.c_str());
    else if (w == 0) {
      weights_dict.put(weights_name, mul_vector.get());
    }
  }

  const char *MulANNComponent::luaCtorName() const {
    return "ann.components.mul";
  }
  
  int MulANNComponent::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable t(L);
    t["size"]    = input_size;
    t["scalar"]  = scalar;
    t["name"]    = name;
    t["weights"] = weights_name;
    t["matrix"]  = mul_vector.get();
    t.pushTable(L);
    return 1;
  }
}
