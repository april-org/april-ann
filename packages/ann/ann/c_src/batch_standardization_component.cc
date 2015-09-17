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

static AprilUtils::SharedPtr<MatrixFloat> computeMean(MatrixFloat *m) {
  float inv_N = 1.0f/static_cast<float>(m->getDimSize(0));
  AprilUtils::SharedPtr<MatrixFloat> mean;
  mean = matSum(m, 0);
  matScal(mean.get(), inv_N);
  return mean;
}
  
static void applyCenter(MatrixFloat *m, MatrixFloat *c, float a=-1.0f) {
  AprilUtils::SharedPtr<MatrixFloat> m_row;
  for (int i=0; i<m->getDimSize(0); ++i) {
    m_row = m->select(0, i, m_row.get());
    matAxpy(m_row.get(), a, c);
  }
}

static void applyScale(MatrixFloat *m, MatrixFloat *s) {
  AprilUtils::SharedPtr<MatrixFloat> m_row;
  for (int i=0; i<m->getDimSize(0); ++i) {
    m_row = m->select(0, i, m_row.get());
    matCmul(m_row.get(), s);
  }
}

static void applyCenterScale(MatrixFloat *m, MatrixFloat *c, MatrixFloat *s) {
  AprilUtils::SharedPtr<MatrixFloat> m_row;
  for (int i=0; i<m->getDimSize(0); ++i) {
    m_row = m->select(0, i, m_row.get());
    matAxpy(m_row.get(), -1.0f, c);
    matCmul(m_row.get(), s);
  }
}

namespace ANN {
  
  BatchStandardizationANNComponent::
  BatchStandardizationANNComponent(float alpha, float epsilon,
                                   unsigned int size,
                                   const char *name,
                                   MatrixFloat *mean,
                                   MatrixFloat *inv_std) :
    VirtualMatrixANNComponent(name, 0, size, size),
    alpha(alpha), epsilon(epsilon),
    running_mean(mean), running_inv_std(inv_std)
  {
    setInputContiguousProperty(true);
  }

  BatchStandardizationANNComponent::~BatchStandardizationANNComponent() {
  }

  MatrixFloat *BatchStandardizationANNComponent::
  privateDoForward(MatrixFloat* input, bool during_training) {
    if (input->getNumDim() < 2)
      ERROR_EXIT2(128, "A 2-dimensional matrix is expected, found %d. "
		  "[%s]", input->getNumDim(), name.c_str());
    if (running_mean.empty()) {
      ERROR_EXIT1(129, "Not built component %s\n", name.c_str());
    }
    MatrixFloat *output = input->clone();
    if (!during_training) {
      applyCenterScale(output, running_mean.get(), running_inv_std.get());
    }
    // compute running mean and inv_std
    else {
      if (input->getDimSize(0) < 2) {
        ERROR_EXIT(128, "Batch std. needs more than one sample per bunch\n");
      }
      mean = computeMean(output);
      applyCenter(output, mean.get());
      centered = output->clone(); // copy for backprop method
      // apply square to output matrix
      matPow(output,2.0f);
      // compute inverted standard deviation for current batch
      inv_std = computeMean(output);
      matScalarAdd(inv_std.get(), epsilon);
      matSqrt(inv_std.get());
      matPow(inv_std.get(), -1.0f);
      // update running mean and inv_std matrices
      matScal(running_mean.get(), 1.0f - alpha);
      matAxpy(running_mean.get(), alpha, mean.get());
      matScal(running_inv_std.get(), 1.0f - alpha);
      matAxpy(running_inv_std.get(), alpha, inv_std.get());
      // copy centered into output matrix and apply scale
      matCopy(output, centered.get());
      applyScale(output, inv_std.get());
    }
    return output;
  }

  /// In BatchStandardizationANNComponent this method is a multiplication by the weights vector
  MatrixFloat *BatchStandardizationANNComponent::privateDoBackprop(MatrixFloat *error_input)
  {
    if (error_input->getNumDim() < 2)
      ERROR_EXIT2(128, "A 2-dimensional matrix is expected, found %d. "
		  "[%s]", error_input->getNumDim(), name.c_str());
    if (mean.empty()) {
      ERROR_EXIT1(129, "Requires forward execution [%s]\n",name.c_str());
    }
    // transfer of input to output
    MatrixFloat *error_output = error_input->clone();
    matCmul(error_output, centered.get());
    AprilUtils::SharedPtr<MatrixFloat> gmean = computeMean(error_output);
    AprilUtils::SharedPtr<MatrixFloat> row;
    for (int i=0; i<error_output->getDimSize(0); ++i) {
      row = error_output->select(0, i, row.get());
      matCopy(row.get(), gmean.get());
    }
    matCmul(error_output, centered.get());
    matScal(error_output, -1.0f);
    applyScale(error_output, inv_std.get());
    applyScale(error_output, inv_std.get());
    
    matAxpy(error_output, 1.0f, error_input);
    gmean = computeMean(error_input);
    applyCenterScale(error_output, gmean.get(), inv_std.get());
    //
    return error_output;
  }

  void BatchStandardizationANNComponent::privateReset(unsigned int it) {
    UNUSED_VARIABLE(it);
    mean.reset();
    inv_std.reset();
    centered.reset();
  }
  
  void BatchStandardizationANNComponent::computeGradients(const char *name,
                                                          AprilUtils::LuaTable &weight_grads_dict) {
    UNUSED_VARIABLE(name);
    UNUSED_VARIABLE(weight_grads_dict);
  }
    
  ANNComponent *BatchStandardizationANNComponent::clone(AprilUtils::LuaTable &copies) {
    UNUSED_VARIABLE(copies);
    BatchStandardizationANNComponent *component =
      new BatchStandardizationANNComponent(alpha, epsilon, input_size,
                                           name.c_str(),
                                           !running_mean.empty() ? running_mean->clone() : 0,
                                           !running_inv_std.empty() ? running_inv_std->clone() : 0);
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
    if (!running_mean.empty()) {
      if (running_mean->size() != static_cast<int>(input_size)) {
        ERROR_EXIT3(256, "Expected size %d, found %d in build [%s]\n",
                    running_mean->size(), input_size, name.c_str());
      }
    }
    else {
      running_mean = matZeros(new MatrixFloat(1, input_size));
    }
    if (!running_inv_std.empty()) {
      if (running_inv_std->size() != static_cast<int>(input_size)) {
        ERROR_EXIT3(256, "Expected size %d, found %d in build [%s]\n",
                    running_inv_std->size(), input_size, name.c_str());
      }
    }
    else {
      running_inv_std = matOnes(new MatrixFloat(1, input_size));
    }
  }

  const char *BatchStandardizationANNComponent::luaCtorName() const {
    return "ann.components.batch_standardization";
  }
  
  int BatchStandardizationANNComponent::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable t(L);
    t["alpha"]   = alpha;
    t["epsilon"] = epsilon;
    t["size"]    = input_size;
    t["name"]    = name;
    t["mean"]    = running_mean;
    t["inv_std"] = running_inv_std;
    t.pushTable(L);
    return 1;
  }
}
