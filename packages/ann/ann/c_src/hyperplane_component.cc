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
#include "hyperplane_component.h"

using namespace Basics;
using namespace AprilUtils;
using namespace AprilMath;

namespace ANN {

  HyperplaneANNComponent::HyperplaneANNComponent(const char *name) :
    ANNComponent(name, 0, 0, 0), dot_product(), bias() {
  }
  
  HyperplaneANNComponent::HyperplaneANNComponent(const char *name,
						 const char *dot_product_name,
						 const char *bias_name,
						 const char *dot_product_weights_name,
						 const char *bias_weights_name,
						 unsigned int input_size,
						 unsigned int output_size,
						 bool transpose_weights,
                                                 MatrixFloat *wmatrix,
                                                 MatrixFloat *bmatrix) :
    ANNComponent(name, 0, input_size, output_size),
    dot_product(new DotProductANNComponent(dot_product_name,
					   dot_product_weights_name,
					   input_size, output_size,
					   transpose_weights,
                                           wmatrix)),
    bias(new BiasANNComponent(output_size, bias_name,
                              bias_weights_name, bmatrix)) {
    IncRef(dot_product);
    IncRef(bias);
  }
  
  HyperplaneANNComponent::~HyperplaneANNComponent() {
    if (dot_product != 0) DecRef(dot_product);
    if (bias != 0) DecRef(bias);
  }

  Token *HyperplaneANNComponent::getInput() {
    return dot_product->getInput();
  }

  Token *HyperplaneANNComponent::getOutput() {
    return bias->getOutput();
  }
  
  Token *HyperplaneANNComponent::getErrorInput() {
    return bias->getErrorInput();
  }

  Token *HyperplaneANNComponent::getErrorOutput() {
    return dot_product->getErrorOutput();
  }
    
  Token *HyperplaneANNComponent::doForward(Token* input, bool during_training) {
    Token *output = dot_product->doForward(input, during_training);
    output = bias->doForward(output, during_training);
    return output;
  }

  Token *HyperplaneANNComponent::doBackprop(Token *input_error) {
    Token *output = bias->doBackprop(input_error);
    output = dot_product->doBackprop(output);
    return output;
  }
    
  void HyperplaneANNComponent::reset(unsigned int it) {
    dot_product->reset(it);
    bias->reset(it);
  }

  void HyperplaneANNComponent::computeAllGradients(AprilUtils::LuaTable &weight_grads_dict) {
    dot_product->computeAllGradients(weight_grads_dict);
    bias->computeAllGradients(weight_grads_dict);
  }
    
  ANNComponent *HyperplaneANNComponent::clone(AprilUtils::LuaTable &copies) {
    HyperplaneANNComponent *obj;
    obj = new HyperplaneANNComponent(name.c_str());
    obj->dot_product = dynamic_cast<DotProductANNComponent*>(dot_product->clone(copies));
    obj->bias        = dynamic_cast<BiasANNComponent*>(bias->clone(copies));
    obj->input_size  = input_size;
    obj->output_size = output_size;
    IncRef(obj->dot_product);
    IncRef(obj->bias);
    return obj;
  }
  
  void HyperplaneANNComponent::setUseCuda(bool v) {
    ANNComponent::setUseCuda(v);
    dot_product->setUseCuda(v);
    bias->setUseCuda(v);
  }
  
  void HyperplaneANNComponent::build(unsigned int _input_size,
				     unsigned int _output_size,
				     AprilUtils::LuaTable &weights_dict,
				     AprilUtils::LuaTable &components_dict) {
    ANNComponent::build(_input_size, _output_size, weights_dict, components_dict);
    //////////////////////////////////////////////////////////////
    if (input_size == 0 || output_size == 0)
      ERROR_EXIT1(141, "Impossible to compute input/output "
		  "sizes for this component [%s]\n",
		  name.c_str());
    //
    dot_product->build(input_size, output_size, weights_dict, components_dict);
    bias->build(output_size, output_size, weights_dict, components_dict);
  }
  
  void HyperplaneANNComponent::copyWeights(AprilUtils::LuaTable &weights_dict) {
    dot_product->copyWeights(weights_dict);
    bias->copyWeights(weights_dict);
  }

  void HyperplaneANNComponent::copyComponents(AprilUtils::LuaTable &components_dict) {
    ANNComponent::copyComponents(components_dict);
    dot_product->copyComponents(components_dict);
    bias->copyComponents(components_dict);
  }
    
  ANNComponent *HyperplaneANNComponent::getComponent(string &name) {
    ANNComponent *component = ANNComponent::getComponent(name);
    if (component == 0) {
      component = dot_product->getComponent(name);
      if (component == 0) component = bias->getComponent(name);
    }
    return component;
  }

  const char *HyperplaneANNComponent::luaCtorName() const {
    return "ann.components.hyperplane";
  }
  int HyperplaneANNComponent::exportParamsToLua(lua_State *L) {
    AprilUtils::LuaTable t(L), weights(L);
    dot_product->copyWeights(weights);
    bias->copyWeights(weights);
    t["name"] = name;
    t["dot_product_name"] = dot_product->getName();
    t["bias_name"] = bias->getName();
    t["dot_product_weights"] = dot_product->getWeightsName();
    t["bias_weights"] = bias->getWeightsName();
    t["input"] = input_size;
    t["output"] = output_size;
    t["transpose"] = dot_product->transposed();
    t["wmatrix"] = weights[dot_product->getWeightsName()].get<MatrixFloat*>();
    t["bmatrix"] = weights[bias->getWeightsName()].get<MatrixFloat*>();
    t.pushTable(L);
    return 1;
  }
}
