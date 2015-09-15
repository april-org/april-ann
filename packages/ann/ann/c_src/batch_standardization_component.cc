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
#include "batch_standardization_component.h"
#include "unused_variable.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  BatchStandardizationANNComponent::
  BatchStandardizationANNComponent(float alpha, float epsilon,
                                   unsigned int size,
                                   const char *name,
                                   MatrixFloat *mean,
                                   MatrixFloat *inv_std) :
    VirtualMatrixANNComponent(name, 0, size, size),
    alpha(alpha), epsilon(epsilon),
    mean(mean),
    inv_std(inv_std),
  {
    setInputContiguousProperty(true);
  }

  BatchStandardizationANNComponent::~BatchStandardizationANNComponent() {
  }

  MatrixFloat *BatchStandardizationANNComponent::
  privateDoForward(MatrixFloat* input,
                   bool during_training) {
    if (input->getNumDim() < 2)
      ERROR_EXIT2(128, "A 2-dimensional matrix is expected, found %d. "
		  "[%s]", input->getNumDim(), name.c_str());
    if (mean.empty()) ERROR_EXIT1(129, "Not built component %s\n", name.c_str());
    MatrixFloat *output;
    // compute running mean and inv_std
    if (during_training) {
      output = input->clone();
      float inv_N = 1.0f/static_cast<float>(input->getDimSize(0));
      AprilUtils::SharedPtr<MatrixFloat> batch_mean = matScal(matSum(input, 0),inv_N);
      AprilUtils::SharedPtr<MatrixFloat> output_row;
      for (int i=0; i<output->getDimSize(0); ++i) {
        output_row = output->select(0, i, output_row.get());
        matAxpy(output_row.get(), -1.0f, batch_mean.get());
      }
      AprilUtils::SharedPtr<MatrixFloat> batch_inv_std =
        matPow( matScalarAdd( matSqrt( matScal( matSum( matPow(output,2.0f), // x^2
                                                        0), // sum(x,0)
                                                inv_N ) // x/N
                                       ), // sqrt(x)
                              epsilon ), // x + eps
                -1.0f); // x^-1
      matAxpy( matScal(mean.get(), 1.0f - alpha), momentum, batch_mean.get() );
      matAxpy( matScal(inv_std.get(), 1.0f - alpha), momentum, batch_inv_std.get() );
    }
    // apply transformation
    matCopy(output, input.get());
    AprilUtils::SharedPtr<MatrixFloat> output_row;
    for (int i=0; i<output->getDimSize(0); ++i) {
      output_row = output->select(0, i, output_row.get());
      matAxpy(output_row.get(), -1.0f, mean.get()); // x = x - mean
      matCmul(output_row.get(), inv_std.get()); // x = x / std
    }
    return output;
  }

  /// In BatchStandardizationANNComponent this method is a multiplication by the weights vector
  MatrixFloat *BatchStandardizationANNComponent::privateDoBackprop(MatrixFloat *error_input)
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

  void BatchStandardizationANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
    // reset scalar counter
    mul_vector->resetSharedCount();
  }
  
  void BatchStandardizationANNComponent::computeGradients(const char *name,
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
    
  ANNComponent *BatchStandardizationANNComponent::clone(AprilUtils::LuaTable &copies) {
    UNUSED_VARIABLE(copies);
    BatchStandardizationANNComponent *component =
      new BatchStandardizationANNComponent(input_size,
                                           name.c_str(),
                                           weights_name.c_str());
    return component;
  }
  
  void BatchStandardizationANNComponent::build(unsigned int _input_size,
                                               unsigned int _output_size,
                                               AprilUtils::LuaTable &weights_dict,
                                               AprilUtils::LuaTable &components_dict) {
    ANNComponent::build(_input_size, _output_size,
                        weights_dict, components_dict);
    //
    if (input_size == 0 && output_size == 0)
      ERROR_EXIT1(141, "Impossible to compute input/output "
		  "sizes for this component [%s]\n",
		  name.c_str());
    if (input_size == 0) input_size = output_size;
    else if (output_size == 0) output_size = input_size;
    else if (input_size != output_size)
      ERROR_EXIT1(142, "BatchStandardizationANNComponent input/output sizes must be equal [%s]\n",
                  name.c_str());
    
    ////////////////////////////////////////////////////////////////////
    if (!mean.empty()) {
      if (mean->size() != static_cast<int>(input_size)) {
        ERROR_EXIT2(256, "Expected size %d, found %d in build [%s]\n",
                    mean->size(), input_size, name.c_str());
      }
    }
    else {
      mean = matZeros(new MatrixFloat(1, input_size));
    }
    if (!inv_std.empty()) {
      if (inv_std->size() != static_cast<int>(input_size)) {
        ERROR_EXIT2(256, "Expected size %d, found %d in build [%s]\n",
                    inv_std->size(), input_size, name.c_str());
      }
    }
    else {
      inv_std = matOnes(new MatrixFloat(1, input_size));
    }
  }

  const char *BatchStandardizationANNComponent::luaCtorName() const {
    return "ann.components.batch_standardization";
  }
  
  int BatchStandardizationANNComponent::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable t(L);
    t["size"]    = input_size;
    t["name"]    = name;
    t["weights"] = weights_name;
    t["mean"]    = mean.get();
    t["inv_std"] = inv_std.get();
    t.pushTable(L);
    return 1;
  }
}
