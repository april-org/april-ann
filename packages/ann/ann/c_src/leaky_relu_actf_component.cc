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
#include "leaky_relu_actf_component.h"
#include "unused_variable.h"

using namespace AprilMath;
using namespace AprilMath::MatrixExt::Operations;
using namespace AprilUtils;
using namespace Basics;

namespace ANN {

  LeakyReLUActfANNComponent::LeakyReLUActfANNComponent(float leak,
                                                       const char *name) :
    ActivationFunctionANNComponent(name), leak(leak) { }
  LeakyReLUActfANNComponent::~LeakyReLUActfANNComponent() { }

  void LeakyReLUActfANNComponent::applyActivation(MatrixFloat *input_units,
                                                  MatrixFloat *output_units) {
    Kernels::applyLeakyReLU(output_units, input_units, leak);
  }

  void LeakyReLUActfANNComponent::multiplyDerivatives(MatrixFloat *input_units,
                                                      MatrixFloat *output_units,
                                                      MatrixFloat *input_errors,
                                                      MatrixFloat *output_errors) {
    UNUSED_VARIABLE(output_units);
    Kernels::applyLeakyReLUDerivative(output_errors, input_units, leak);
    matCmul(output_errors, input_errors);
  }

  ANNComponent *LeakyReLUActfANNComponent::clone(AprilUtils::LuaTable &copies) {
    UNUSED_VARIABLE(copies);
    LeakyReLUActfANNComponent *obj = new LeakyReLUActfANNComponent(leak,
                                                                   name.c_str());
    return obj;
  }

  const char *LeakyReLUActfANNComponent::luaCtorName() const {
    return "ann.components.actf.leaky_relu";
  }
  int LeakyReLUActfANNComponent::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable t(L);
    t["name"] = name;
    t["leak"] = leak;
    t.pushTable(L);
    return 1;
  }
}
