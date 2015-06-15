/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2015, Francisco Zamora-Martinez
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
#include "activation_function_kernels.h"
#include "cblas_headers.h"
#include "error_print.h"
#include "prelu_actf_component.h"
#include "unused_variable.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::BLAS;
using namespace AprilMath::MatrixExt::Boolean;
using namespace AprilMath::MatrixExt::Initializers;
using namespace AprilMath::MatrixExt::Misc;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilMath::MatrixExt::Reductions;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  PReLUActfANNComponent::PReLUActfANNComponent(bool scalar, unsigned int size,
                                               const char *name,
                                               const char *weights_name,
                                               MatrixFloat *matrix) :
    ActivationFunctionANNComponent(name),
    weights(matrix), size(size), scalar(scalar) {
    if (weights_name) {
      this->weights_name = string(weights_name);
    }
    else {
      generateDefaultWeightsName("a");
    }
  }
  PReLUActfANNComponent::~PReLUActfANNComponent() { }

  void PReLUActfANNComponent::applyActivation(MatrixFloat *input_units,
                                              MatrixFloat *output_units) {
    if (scalar) {
      Kernels::applyLeakyReLU(output_units, input_units, (*weights)(0,0));
    }
    else {
      Kernels::applyPReLU(output_units, input_units, weights.get());
    }
  }

  void PReLUActfANNComponent::multiplyDerivatives(MatrixFloat *input_units,
                                                  MatrixFloat *output_units,
                                                  MatrixFloat *input_errors,
                                                  MatrixFloat *output_errors) {
    UNUSED_VARIABLE(output_units);
    if (scalar) {
      Kernels::applyLeakyReLUDerivative(output_errors, input_units, (*weights)(0,0));
    }
    else {
      Kernels::applyPReLUDerivative(output_errors, input_units, weights.get());
    }
    matCmul(output_errors, input_errors);
  }
  
  void PReLUActfANNComponent::computeGradients(const char *weights_name,
                                               AprilUtils::LuaTable &weight_grads) {
    weights->addToSharedCount();
    MatrixFloat *grads_mat = weight_grads.opt<MatrixFloat*>(weights_name, 0);
    if (grads_mat == 0) {
      grads_mat = weights->cloneOnlyDims();
      matZeros(grads_mat);
      weight_grads.put(weights_name, grads_mat);
    }
    else if (!grads_mat->sameDim(weights.get())) {
      ERROR_EXIT(128, "Incorrect weights matrix dimensions\n");
    }
#ifdef USE_CUDA
    grads_mat->setUseCuda(use_cuda);
#endif
    MatrixFloat *input = getInput()->convertTo<TokenMatrixFloat*>()->getMatrix();
    MatrixFloat *error = getErrorInput()->convertTo<TokenMatrixFloat*>()->getMatrix();
    //
    AprilUtils::SharedPtr<Basics::MatrixBool> lt_zero;
    lt_zero = new MatrixBool(input->getNumDim(), input->getDimPtr());
    lt_zero = matLT(input, 0.0f, lt_zero.get());
    AprilUtils::SharedPtr<Basics::MatrixFloat> error_units;
    error_units = matConvertTo(lt_zero.get(), error_units.get());
    error_units = matCmul(matCmul(error_units.get(), input), error);
    if (scalar) {
      (*grads_mat)(0,0) += matSum(error_units.get());
    }
    else {
      // accumulated over previous gradients
      matSum(error_units.get(), 0, grads_mat, true);
    }
  }

  void PReLUActfANNComponent::reset(unsigned int it) {
    ANNComponent::reset(it);
    weights->resetSharedCount();
  }
  
  ANNComponent *PReLUActfANNComponent::clone() {
    PReLUActfANNComponent *obj = new PReLUActfANNComponent(scalar, size,
                                                           name.c_str(),
                                                           weights_name.c_str());
    return obj;
  }

  void PReLUActfANNComponent::build(unsigned int _input_size,
                                    unsigned int _output_size,
                                    AprilUtils::LuaTable &weights_dict,
                                    AprilUtils::LuaTable &components_dict) {
    ActivationFunctionANNComponent::build(_input_size, _output_size, weights_dict, components_dict);
    unsigned int _size = getInputSize();
    if (this->size != 0 && _size != 0 && this->size != _size) {
      ERROR_EXIT(128, "Incompatible sizes in build call\n");
    }
    if (_size != 0) this->size = _size;
    this->input_size = this->output_size = this->size;
    if (this->size == 0) ERROR_EXIT(256, "Unable to allocate prelu weights\n");
    unsigned int weights_size = (scalar) ? (1) : (this->size);
    //
    { // block for w variable
      MatrixFloat *w = weights_dict.opt<MatrixFloat*>(getWeightsName(), 0);
      if (w != 0) {
        weights = w;
      }
      else {
        if (weights == 0) {
          weights = Connections::build(1, weights_size);
        }
        weights_dict.put<MatrixFloat*>(getWeightsName(), weights.get());
      }
    } // end of block for w variable
    //
    if (weights->size() != static_cast<int>(weights_size)) {
      ERROR_EXIT1(257, "Unexpected matrix size [%s]\n", name.c_str());
    }
  }

  const char *PReLUActfANNComponent::luaCtorName() const {
    return "ann.components.actf.prelu";
  }
  int PReLUActfANNComponent::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable t(L);
    t["name"] = name.c_str();
    t["weights"] = weights_name.c_str();
    t["size"] = size;
    t["scalar"] = scalar ? "true" : "false";
    t["matrix"] = weights.get();
    t.pushTable(L);
    return 1;
  }
}
