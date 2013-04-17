/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
    ANNComponent(name, 0) { }
  StackANNComponent::~StackANNComponent() { }

  const Token *StackANNComponent::getInput() const {
    return components[0]->getInput();
  }

  const Token *StackANNComponent::getOutput() const {
    return components.back()->getInput();
  }
  
  const Token *StackANNComponent::getErrorInput() const {
    return components.back()->getErrorInput();
  }

  const Token *StackANNComponent::getErrorOutput() const {
    return components.back()->getErrorOutput();
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
      aux_token = components[c-1]->doBackprop(aux_token, during_training);
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
    
  virtual ANNComponent *clone() {
    StackANNComponent *obj = new StackANNComponent(name);
    for (unsigned int c=0; c<components.size(); ++c)
      obj->components.push_back(components[c]->clone());
  }
    
  void setOption(const char *name, double value) {
    for (unsigned int c=0; c<components.size(); ++c)
      components[c]->setOption(name, value);
  }

  bool hasOption(const char *name) {
    bool ret = false;
    for (unsigned int c=0; c<components.size() && !ret; ++c)
      ret = components[c]->hasOption(name);
    return ret;
  }
    
  double getOption(const char *name) {
    for (unsigned int c=0; c<components.size(); ++c) {
      if (components[c]->hasOption(name))
	return components[c]->getOption(name);
    }
    ERROR_EXIT1(140, "The option %s does not exist.\n", name);
  }
    
  void build(unsigned int _input_size,
	     unsigned int _output_size,
	     hash<string,Connections*> &weights_dict,
	     hash<string,ANNComponent*> &components_dict) {
    ANNComponent(_input_size, _output_size, weights_dict, components_dict);
    //////////////////////////////////////////////////////////////
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
    if (output_size == 0) output_size = current_output_size;
    if (input_size == 0 || output_size == 0)
      ERROR_EXIT(141, "Impossible to compute input/output "
		 "sizes for this component\n");
    if (current_output_size != output_size)
      ERROR_EXIT(141, "StackANNComponent output size are not correct\n");
  }
    
  virtual void copyWeights(hash<string,Connections*> &weights_dict);

  virtual void copyComponents(hash<string,ANNComponent*> &weights_dict);
    
  virtual ANNComponent *getComponent(string &name);
}
