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
#include "const_component.h"

using Basics::Token;

namespace ANN {
  
  ConstANNComponent::ConstANNComponent(ANNComponent *component,
				       const char *name) :
    ANNComponent(name, 0,
		 component->getInputSize(),
		 component->getOutputSize()),
    component(component) {
    if (!component->getIsBuilt()) {
      ERROR_EXIT(128, "Needs a built component!\n");
    }
    component->copyWeights(component_weights);
  }
  
  ConstANNComponent::~ConstANNComponent() { }
  
  Token *ConstANNComponent::doForward(Token* _input, bool during_training) {
    return component->doForward(_input, during_training);
  }

  Token *ConstANNComponent::doBackprop(Token *_error_input) {
    return component->doBackprop(_error_input);
  }
  
  void ConstANNComponent::reset(unsigned int it) {
    component->reset(it);
  }
  
  ANNComponent *ConstANNComponent::clone() {
    return new ConstANNComponent(component->clone(), name.c_str());
  }
  
  void ConstANNComponent::build(unsigned int _input_size,
                                unsigned int _output_size,
                                AprilUtils::LuaTable &weights_dict,
                                AprilUtils::LuaTable &components_dict) {
    ANNComponent::build(_input_size, _output_size, weights_dict, components_dict);
  }
  
  char *ConstANNComponent::toLuaString() {
    AprilUtils::SharedPtr<AprilIO::CStringStream>
      stream(new AprilIO::CStringStream());
    char *component_str = component->toLuaString();
    AprilUtils::string component_weights_str( component_weights.toLuaString() );
    stream->printf("ann.components.const{ name='%s', component=%s:build{ weights=%s } }",
                   name.c_str(),
		   component_str,
                   component_weights_str.c_str());
    stream->put("\0",1); // forces a \0 at the end of the buffer
    delete[] component_str;
    delete[] component_weights_str;
    return stream->releaseString();
  }
}
