/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#include "hyperplane_component.h"

namespace ANN {

  HyperplaneANNComponent::HyperplaneANNComponent(const char *name) :
    ANNComponent(name), dot_product(0), bias(0) {
  }
  
  HyperplaneANNComponent::HyperplaneANNComponent(const char *name,
						 const char *dot_product_name,
						 const char *bias_name,
						 const char *dot_product_weights_name,
						 const char *bias_weights_name,
						 unsigned int input_size,
						 unsigned int output_size,
						 bool transpose_weights) :
    ANNComponent(name, input_size, output_size),
    dot_product(new DotProductANNComponent(dot_product_name,
					   dot_product_weights_name,
					   input_size, output_size,
					   transpose_weights)),
    bias(new BiasANNComponent(bias_name, bias_weights_name)) {
    IncRef(dot_product);
    IncRef(bias);
  }
  
  HyperplaneANNComponent::~HyperplaneANNComponent() {
    if (dot_product != 0) DecRef(dot_product);
    if (bias != 0) DecRef(bias);
  }

  const Token *HyperplaneANNComponent::getInput() const {
    return dot_product->getInput();
  }

  const Token *HyperplaneANNComponent::getOutput() const {
    return bias->getOutput();
  }
  
  const Token *HyperplaneANNComponent::getErrorInput() const {
    return bias->getErrorInput();
  }

  const Token *HyperplaneANNComponent::getErrorOutput() const {
    return dot_product->getErrorOutput();
  }
    
  Token *HyperplaneANNComponent::doForward(Token* input, bool during_training) {
    return bias->doForward(dot_product->doForward(input,
						  during_training),
			   during_training);
  }

  Token *HyperplaneANNComponent::doBackprop(Token *input_error) {
    return dot_product->doBackprop(bias->doBackprop(input_error));
  }
    
  void HyperplaneANNComponent::doUpdate() {
    bias->doUpdate();
    dot_product->doUpdate();
  }

  void HyperplaneANNComponent::reset() {
    dot_product->reset();
    bias->reset();
  }
    
  ANNComponent *HyperplaneANNComponent::clone() {
    HyperplaneANNComponent *obj;
    obj = new HyperplaneANNComponent(name.c_str());
    obj->dot_product = dot_product->clone();
    obj->bias        = bias->clone();
    IncRef(obj->dot_product);
    IncRef(obj->bias);
    return obj;
  }
  
  void HyperplaneANNComponent::setUseCuda(bool v) {
    ANNComponent::setUseCuda(v);
    dot_product->setUseCuda(v);
    bias->setUseCuda(v);
  }
  
  void HyperplaneANNComponent::setOption(const char *name, double value) {
    dot_product->setOption(name, value);
    bias->setOption(name, value);
  }

  bool HyperplaneANNComponent::hasOption(const char *name) {
    return dot_product->hasOption(name) || bias->hasOption(name);
  }
    
  double HyperplaneANNComponent::getOption(const char *name) {
    if (dot_product->hasOption(name)) return dot_product->getOption(name);
    if (bias->hasOption(name)) return bias->getOption(name);
    return ANNComponent::getOption(name);
  }
    
  void HyperplaneANNComponent::build(unsigned int _input_size,
				     unsigned int _output_size,
				     hash<string,Connections*> &weights_dict,
				     hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size, weights_dict, components_dict);
    //////////////////////////////////////////////////////////////
    if (input_size == 0 || output_size == 0)
      ERROR_EXIT(141, "Impossible to compute input/output "
		 "sizes for this component\n");
    //
    dot_product->build(input_size, output_size, weights_dict, components_dict);
    bias->build(output_size, output_size, weights_dict, components_dict);
  }
  
  void HyperplaneANNComponent::copyWeights(hash<string,Connections*> &weights_dict) {
    dot_product->copyWeights(weights_dict);
    bias->copyWeights(weights_dict);
  }

  void HyperplaneANNComponent::copyComponents(hash<string,ANNComponent*> &components_dict) {
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

}
