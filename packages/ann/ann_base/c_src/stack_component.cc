/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
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
#include "stack_component.h"

namespace ANN {

  StackANNComponent::StackANNComponent(const char *name) :
    ANNComponent(name) { }
  
  StackANNComponent::~StackANNComponent() {
    for (unsigned int i=0; i<components.size(); ++i)
      DecRef(components[i]);
  }

  void StackANNComponent::pushComponent(ANNComponent *component) {
    IncRef(component);
    components.push_back(component);
  }

  ANNComponent *StackANNComponent::topComponent() {
    return components.back();
  }

  void StackANNComponent::popComponent() {
    DecRef(components.back());
    components.pop_back();
    if (components.size() > 0)
      output_size = components.back()->getOutputSize();
    else output_size = 0;
  }

  Token *StackANNComponent::getInput() {
    return components[0]->getInput();
  }

  Token *StackANNComponent::getOutput() {
    return components.back()->getOutput();
  }
  
  Token *StackANNComponent::getErrorInput() {
    return components.back()->getErrorInput();
  }

  Token *StackANNComponent::getErrorOutput() {
    return components[0]->getErrorOutput();
  }
    
  Token *StackANNComponent::doForward(Token* input, bool during_training) {
    Token *aux_token = input;
    for (unsigned int c=0; c<components.size(); ++c)
      aux_token = components[c]->doForward(aux_token, during_training);
    return aux_token;
  }

  Token *StackANNComponent::doBackprop(Token *input_error) {
    Token *aux_token = input_error;
    for (unsigned int c=components.size(); c>0; --c)
      aux_token = components[c-1]->doBackprop(aux_token);
    return aux_token;
  }
    
  void StackANNComponent::doUpdate() {
    for (unsigned int c=components.size(); c>0; --c)
      components[c-1]->doUpdate();
  }

  void StackANNComponent::reset() {
    for (unsigned int c=0; c<components.size(); ++c)
      components[c]->reset();
  }
  
  void StackANNComponent::computeAllGradients(hash<string,MatrixFloat*>
					      &weight_grads_dict) {
    for (unsigned int c=0; c<components.size(); ++c)
      components[c]->computeAllGradients(weight_grads_dict);
  }

  ANNComponent *StackANNComponent::clone() {
    StackANNComponent *obj = new StackANNComponent(name.c_str());
    for (unsigned int c=0; c<components.size(); ++c)
      obj->pushComponent(components[c]->clone());
    obj->input_size  = input_size;
    obj->output_size = output_size;
    return obj;
  }
  
  void StackANNComponent::setUseCuda(bool v) {
    ANNComponent::setUseCuda(v);
    for (unsigned int c=0; c<components.size(); ++c)
      components[c]->setUseCuda(v);
  }
  
  void StackANNComponent::setOption(const char *name, double value) {
    for (unsigned int c=0; c<components.size(); ++c)
      components[c]->setOption(name, value);
  }

  bool StackANNComponent::hasOption(const char *name) {
    bool ret = false;
    for (unsigned int c=0; c<components.size() && !ret; ++c)
      ret = components[c]->hasOption(name);
    return ret;
  }
    
  double StackANNComponent::getOption(const char *name) {
    for (unsigned int c=0; c<components.size(); ++c) {
      if (components[c]->hasOption(name))
	return components[c]->getOption(name);
    }
    return ANNComponent::getOption(name);
  }
    
  void StackANNComponent::build(unsigned int _input_size,
				unsigned int _output_size,
				hash<string,Connections*> &weights_dict,
				hash<string,ANNComponent*> &components_dict) {
    ANNComponent::build(_input_size, _output_size, weights_dict, components_dict);
    //////////////////////////////////////////////////////////////
    if (components.size() == 0)
      ERROR_EXIT1(128, "StackANNComponent needs one or more components, "
		  "use pushComponent method [%s]\n", name.c_str());
    //
    unsigned int current_input_size  = input_size;
    unsigned int current_output_size = 0;
    for (unsigned int c=0; c<components.size(); ++c) {
      if (c < components.size()-1)
	current_output_size = components[c+1]->getInputSize();
      components[c]->build(current_input_size, current_output_size,
			   weights_dict, components_dict);
      current_input_size  = components[c]->getOutputSize();
      current_output_size = 0;
    }
    if (input_size  == 0) input_size  = components[0]->getInputSize();
    if (output_size == 0) output_size = components.back()->getOutputSize();
    else if (output_size != components.back()->getOutputSize())
      ERROR_EXIT3(141, "StackANNComponent output size is not correct: "
		  "%d != %d [%s]\n", output_size,
		  components.back()->getOutputSize(), name.c_str());
    /*
      if (input_size  == 0 || output_size == 0)
      ERROR_PRINT3("# WARNING: Impossible to compute input/output "
      "sizes for this component input=%d output=%d [%s]\n",
      input_size, output_size, name.c_str());
    */
  }
  
  void StackANNComponent::copyWeights(hash<string,Connections*> &weights_dict) {
    for (unsigned int c=0; c<components.size(); ++c)
      components[c]->copyWeights(weights_dict);
  }

  void StackANNComponent::copyComponents(hash<string,ANNComponent*> &components_dict) {
    ANNComponent::copyComponents(components_dict);
    for (unsigned int c=0; c<components.size(); ++c)
      components[c]->copyComponents(components_dict);
  }
    
  ANNComponent *StackANNComponent::getComponent(string &name) {
    ANNComponent *component = ANNComponent::getComponent(name);
    for (unsigned int c=0; c<components.size() && component==0; ++c)
      component = components[c]->getComponent(name);
    return component;
  }

  char *StackANNComponent::toLuaString() {
    buffer_list buffer;
    buffer.printf("ann.components.stack{ name='%s' }", name.c_str());
    for (unsigned int i=0; i<components.size(); ++i) {
      char *aux = components[i]->toLuaString();
      buffer.printf(":push(%s)", aux);
      delete[] aux;
    }
    return buffer.to_string(buffer_list::NULL_TERMINATED);
  }
}
